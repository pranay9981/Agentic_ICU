from __future__ import annotations

import csv
import math
import sys
import unittest
from pathlib import Path
from typing import List

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_icu.api.dependencies import get_workflow
from agentic_icu.api.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_patient_window(patient_id: str, max_rows: int = 24) -> list[dict]:
    patient_path = ROOT / "data" / "raw" / f"{patient_id}.psv"
    window = []
    with patient_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            values = {}
            for key, value in row.items():
                if key == "SepsisLabel" or value in (None, ""):
                    continue
                try:
                    numeric_value = float(value)
                except ValueError:
                    continue
                if math.isfinite(numeric_value):
                    values[key] = numeric_value
            window.append({"values": values})
    return window


# ── Unit: ClinicalReasoner ────────────────────────────────────────────────────

class TestClinicalReasoner(unittest.TestCase):
    def setUp(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy, ClinicalReasoner
        from agentic_icu.domain.contracts import ModelAgentResult, SignalQualityResult

        self.policy = AlertPolicy()
        self.reasoner = ClinicalReasoner(self.policy)
        self.ModelAgentResult = ModelAgentResult
        self.SignalQualityResult = SignalQualityResult

    def _sq_clear(self):
        return self.SignalQualityResult(signal_valid=True, suppression_recommendation=False, suppression_mode="none")

    def _sq_full_suppress(self):
        return self.SignalQualityResult(signal_valid=False, suppression_recommendation=True, suppression_mode="full")

    def _sq_partial_suppress(self):
        return self.SignalQualityResult(
            signal_valid=True, suppression_recommendation=True, suppression_mode="partial",
            artifact_type="soft_range_violation", artifact_affected_features=["Temp"],
        )

    def _agent(self, score: float) -> "ModelAgentResult":
        dt = 0.5
        tr = score / dt
        return self.ModelAgentResult(
            status="available", score=score, risk_band="high" if tr >= 1 else "low",
            detail=f"score {score:.3f}", decision_threshold=dt, threshold_ratio=tr,
        )

    def test_full_suppression_blocks_alert(self) -> None:
        decision, *_ = self.reasoner.decide(
            self._sq_full_suppress(), self._agent(0.99), self._agent(0.99)
        )
        self.assertFalse(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.suppressed_artifact_alert_type)

    def test_high_alert_triggers_on_extreme_sequence(self) -> None:
        decision, *_ = self.reasoner.decide(
            self._sq_clear(), self._agent(0.95), self._agent(0.1)
        )
        self.assertTrue(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.high_alert_type)

    def test_stable_when_below_thresholds(self) -> None:
        decision, *_ = self.reasoner.decide(
            self._sq_clear(), self._agent(0.1), self._agent(0.05)
        )
        self.assertFalse(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.stable_alert_type)

    def test_medium_alert_on_sequence_threshold(self) -> None:
        decision, *_ = self.reasoner.decide(
            self._sq_clear(), self._agent(0.6), self._agent(0.05)
        )
        self.assertTrue(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.medium_alert_type)

    def test_partial_suppression_reduces_scores_consistently(self) -> None:
        """After partial suppression the returned agent results must carry the penalized score,
        consistent threshold_ratio, and a suppression-adjusted detail string."""
        high_vitals = self._agent(0.95)
        decision, logs, adj_vitals, adj_lab, _ = self.reasoner.decide(
            self._sq_partial_suppress(), high_vitals, self._agent(0.1)
        )
        suppressed_score = 0.95 * self.policy.partial_suppression_factor  # 0.665
        # Returned adjusted object must carry the penalized score
        self.assertAlmostEqual(adj_vitals.score, suppressed_score, places=5)
        # Detail must reflect the adjustment (not the original)
        self.assertIn("suppression-adjusted", adj_vitals.detail)
        # threshold_ratio must be consistent: new_score / decision_threshold
        expected_ratio = suppressed_score / high_vitals.decision_threshold
        self.assertAlmostEqual(adj_vitals.threshold_ratio, expected_ratio, places=5)
        # Suppression log must appear
        suppression_log = next((l for l in logs if "suppression" in l.message.lower()), None)
        self.assertIsNotNone(suppression_log)
        # Decision should still fire (0.665 >= medium threshold 0.55)
        self.assertTrue(decision.alert_triggered)

    def test_resp_high_alert_when_sepsis_stable(self) -> None:
        from agentic_icu.domain.contracts import ModelAgentResult
        resp_result = ModelAgentResult(
            status="available", score=0.9, risk_band="high",
            detail="resp score 0.9", decision_threshold=0.4, threshold_ratio=2.25,
        )
        decision, *_ = self.reasoner.decide(
            self._sq_clear(), self._agent(0.1), self._agent(0.05), resp_result
        )
        self.assertTrue(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.resp_high_alert_type)

    def test_no_available_scores_returns_models_unavailable(self) -> None:
        from agentic_icu.domain.contracts import ModelAgentResult
        unavailable = ModelAgentResult(status="unavailable", detail="no model")
        decision, *_ = self.reasoner.decide(self._sq_clear(), unavailable, unavailable)
        self.assertFalse(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.models_unavailable_alert_type)

    def test_resp_high_alert_when_sepsis_models_unavailable(self) -> None:
        """When both sepsis models are unavailable but resp score is high, a resp alert must still fire."""
        from agentic_icu.domain.contracts import ModelAgentResult
        unavailable = ModelAgentResult(status="unavailable", detail="no model")
        resp_result = ModelAgentResult(
            status="available", score=0.9, risk_band="high",
            detail="resp score 0.9", decision_threshold=0.4, threshold_ratio=2.25,
        )
        decision, *_ = self.reasoner.decide(self._sq_clear(), unavailable, unavailable, resp_result)
        self.assertTrue(decision.alert_triggered)
        self.assertEqual(decision.alert_type, self.policy.resp_high_alert_type)

    def test_alert_policy_rejects_threshold_above_one(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        with self.assertRaises(ValueError):
            AlertPolicy(high_alert_extreme_sequence_score_threshold=1.5)

    def test_alert_policy_rejects_negative_threshold(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        with self.assertRaises(ValueError):
            AlertPolicy(medium_alert_sequence_score_threshold=-0.1)

    def test_alert_policy_accepts_valid_probability_thresholds(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        policy = AlertPolicy(
            high_alert_extreme_sequence_score_threshold=0.9,
            medium_alert_sequence_score_threshold=0.5,
        )
        self.assertEqual(policy.high_alert_extreme_sequence_score_threshold, 0.9)

    def test_alert_policy_accepts_none_thresholds(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        policy = AlertPolicy(high_alert_max_score_threshold=None)
        self.assertIsNone(policy.high_alert_max_score_threshold)

    def test_alert_policy_rejects_suppression_factor_above_one(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        with self.assertRaises(ValueError):
            AlertPolicy(partial_suppression_factor=1.5)

    def test_alert_policy_rejects_suppression_factor_below_zero(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        with self.assertRaises(ValueError):
            AlertPolicy(partial_suppression_factor=-0.1)

    def test_alert_policy_accepts_valid_suppression_factor(self) -> None:
        from agentic_icu.agents.reasoner import AlertPolicy
        policy = AlertPolicy(partial_suppression_factor=0.5)
        self.assertEqual(policy.partial_suppression_factor, 0.5)


# ── Unit: RuntimePreprocessor ─────────────────────────────────────────────────

class TestRuntimePreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        from agentic_icu.preprocessing.windowing import RuntimePreprocessor
        from agentic_icu.config import settings

        self.preprocessor = RuntimePreprocessor(
            train_statistics_path=settings.train_statistics_path,
            pipeline_config_path=settings.pipeline_config_path,
        )
        self.preprocessor.load()

    def _make_records(self, n: int = 24) -> list:
        from agentic_icu.domain.contracts import ObservationRecord
        return [
            ObservationRecord(values={
                "HR": 80.0, "O2Sat": 97.0, "SBP": 120.0, "DBP": 70.0,
                "MAP": 87.0, "Resp": 16.0, "Temp": 37.0, "FiO2": 0.21,
                "ICULOS": float(i + 1),
            })
            for i in range(n)
        ]

    def test_sequence_tensor_uses_72h_window(self) -> None:
        """Sequence tensor must be shaped (72, input_size) not (24, input_size)."""
        records = self._make_records(24)
        tensor = self.preprocessor.build_sequence_tensor(records)
        self.assertEqual(tensor.shape[0], self.preprocessor.sequence_hours)
        self.assertGreater(self.preprocessor.sequence_hours, self.preprocessor.observation_hours)

    def test_tabular_features_includes_composite_features(self) -> None:
        """Composite clinical features added in Phase 1.3 must be present."""
        records = self._make_records(24)
        features = self.preprocessor.build_tabular_features(records)
        for key in ("shock_index", "pulse_pressure", "qsofa_score", "HR__slope_6h", "map_computed"):
            self.assertIn(key, features, f"Missing composite feature: {key}")

    def test_tabular_feature_count_matches_training(self) -> None:
        """Feature count must be 292 (matching xgboost_metrics.json feature_count)."""
        records = self._make_records(24)
        features = self.preprocessor.build_tabular_features(records)
        self.assertEqual(len(features), 292, f"Got {len(features)} features, expected 292")

    def test_sequence_tensor_shape_with_short_window(self) -> None:
        """Short windows must be padded to sequence_hours."""
        records = self._make_records(5)
        tensor = self.preprocessor.build_sequence_tensor(records)
        self.assertEqual(tensor.shape[0], self.preprocessor.sequence_hours)

    def test_shock_index_is_correct(self) -> None:
        from agentic_icu.domain.contracts import ObservationRecord
        records = [ObservationRecord(values={"HR": 100.0, "SBP": 200.0, "DBP": 80.0, "ICULOS": 1.0})]
        features = self.preprocessor.build_tabular_features(records)
        self.assertAlmostEqual(features["shock_index"], 0.5, places=5)

    def test_iculos_fallback_uses_sequential_index_when_absent(self) -> None:
        """When ICULOS is missing from record.values, it must default to the sequential row index (1-based).
        Regression test for the dict.get() NaN-key bug fixed in windowing.py."""
        from agentic_icu.domain.contracts import ObservationRecord
        # Records deliberately omit ICULOS — the fallback must assign hour 1, 2, 3.
        records = [
            ObservationRecord(values={"HR": 80.0, "SBP": 120.0}),
            ObservationRecord(values={"HR": 82.0, "SBP": 118.0}),
            ObservationRecord(values={"HR": 84.0, "SBP": 116.0}),
        ]
        df = self.preprocessor.records_to_frame(records)
        self.assertAlmostEqual(df["ICULOS"].iloc[0], 1.0, places=5)
        self.assertAlmostEqual(df["ICULOS"].iloc[1], 2.0, places=5)
        self.assertAlmostEqual(df["ICULOS"].iloc[2], 3.0, places=5)

    def test_iculos_from_record_values_takes_precedence(self) -> None:
        """When ICULOS is present in record.values, that value must be used, not the index."""
        from agentic_icu.domain.contracts import ObservationRecord
        records = [
            ObservationRecord(values={"HR": 80.0, "ICULOS": 48.0}),
            ObservationRecord(values={"HR": 82.0, "ICULOS": 49.0}),
        ]
        df = self.preprocessor.records_to_frame(records)
        self.assertAlmostEqual(df["ICULOS"].iloc[0], 48.0, places=5)
        self.assertAlmostEqual(df["ICULOS"].iloc[1], 49.0, places=5)

    def test_load_is_idempotent(self) -> None:
        """Calling load() a second time must not raise and must return the same stats."""
        stats_before = id(self.preprocessor._stats)
        self.preprocessor.load()  # second call — should be a no-op
        self.assertEqual(id(self.preprocessor._stats), stats_before)

    def test_concurrent_load_is_safe(self) -> None:
        """Concurrent calls to load() must not corrupt state or raise."""
        import threading as _threading
        from agentic_icu.preprocessing.windowing import RuntimePreprocessor
        from agentic_icu.config import settings
        preprocessor = RuntimePreprocessor(
            train_statistics_path=settings.train_statistics_path,
            pipeline_config_path=settings.pipeline_config_path,
        )
        errors: list[Exception] = []
        def _load():
            try:
                preprocessor.load()
            except Exception as exc:
                errors.append(exc)
        threads = [_threading.Thread(target=_load) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertFalse(errors, f"Concurrent load raised: {errors}")
        self.assertIsNotNone(preprocessor._stats)

    def test_concurrent_gru_load_is_safe(self) -> None:
        """Concurrent calls to SequenceInference.load() must not corrupt model state."""
        import threading as _threading
        from agentic_icu.api.dependencies import get_workflow
        predictor = get_workflow().vitals_agent.predictor
        # Reset to unloaded state is not safe in tests — just verify idempotent second call
        errors: list[Exception] = []
        def _load():
            try:
                predictor.load()
            except Exception as exc:
                errors.append(exc)
        threads = [_threading.Thread(target=_load) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertFalse(errors, f"Concurrent GRU load raised: {errors}")
        self.assertIsNotNone(predictor._model)


# ── Unit: SignalQualityAgent ──────────────────────────────────────────────────

class TestSignalQualityAgent(unittest.TestCase):
    def setUp(self) -> None:
        from agentic_icu.agents.signal_quality import SignalQualityAgent
        self.agent = SignalQualityAgent()

    def _eval(self, window: list[dict]) -> tuple:
        result, logs = self.agent.evaluate(window)
        return result, logs

    def _row(self, **kw) -> dict:
        base = {"HR": 80.0, "SBP": 120.0, "DBP": 70.0, "O2Sat": 97.0, "Resp": 16.0, "Temp": 37.0}
        base.update(kw)
        return base

    def test_single_row_accepted_without_checks(self) -> None:
        result, _ = self._eval([self._row()])
        self.assertTrue(result.signal_valid)
        self.assertEqual(result.suppression_mode, "none")

    def test_impossible_hr_triggers_full_suppression(self) -> None:
        window = [self._row(), self._row(HR=5.0)]   # 5 bpm < 15 lower bound
        result, _ = self._eval(window)
        self.assertFalse(result.signal_valid)
        self.assertEqual(result.artifact_type, "impossible_hr")
        self.assertEqual(result.suppression_mode, "full")

    def test_hr_above_upper_bound_triggers_full_suppression(self) -> None:
        window = [self._row(), self._row(HR=350.0)]  # 350 bpm > 300 upper bound
        result, _ = self._eval(window)
        self.assertFalse(result.signal_valid)
        self.assertEqual(result.suppression_mode, "full")

    def test_impossible_sbp_triggers_full_suppression(self) -> None:
        window = [self._row(), self._row(SBP=20.0)]  # below 30 lower bound
        result, _ = self._eval(window)
        self.assertFalse(result.signal_valid)
        self.assertEqual(result.artifact_type, "impossible_sbp")

    def test_impossible_spo2_triggers_full_suppression(self) -> None:
        window = [self._row(), self._row(O2Sat=40.0)]  # below 50 lower bound
        result, _ = self._eval(window)
        self.assertFalse(result.signal_valid)
        self.assertEqual(result.artifact_type, "invalid_spo2")

    def test_improbable_hr_jump_triggers_full_suppression(self) -> None:
        # HR jump > 40 bpm between consecutive rows
        window = [self._row(HR=70.0), self._row(HR=120.0)]
        result, _ = self._eval(window)
        self.assertFalse(result.signal_valid)
        self.assertEqual(result.artifact_type, "improbable_hr_jump")

    def test_bp_inversion_triggers_full_suppression(self) -> None:
        # SBP < DBP is physiologically impossible.
        # Previous row SBP=65 → current SBP=60: drop of 5 mmHg (< 35 threshold) so
        # isolated_bp_drop does not fire; bp_inversion fires instead.
        window = [self._row(SBP=65.0, DBP=50.0), self._row(SBP=60.0, DBP=90.0)]
        result, _ = self._eval(window)
        self.assertFalse(result.signal_valid)
        self.assertEqual(result.artifact_type, "bp_inversion")

    def test_isolated_bp_drop_triggers_full_suppression(self) -> None:
        # >35 mmHg SBP drop with no HR response (HR stable within 5 bpm)
        window = [self._row(HR=80.0, SBP=160.0), self._row(HR=81.0, SBP=120.0)]
        result, _ = self._eval(window)
        self.assertFalse(result.signal_valid)
        self.assertEqual(result.artifact_type, "isolated_bp_drop")

    def test_spo2_hr_paradox_triggers_partial_suppression(self) -> None:
        # SpO2 critically low but HR barely changed → probe-off heuristic
        window = [self._row(O2Sat=97.0, HR=80.0), self._row(O2Sat=75.0, HR=81.0)]
        result, _ = self._eval(window)
        self.assertTrue(result.signal_valid)
        self.assertEqual(result.artifact_type, "spo2_hr_paradox")
        self.assertEqual(result.suppression_mode, "partial")

    def test_flatline_in_hard_block_feature_triggers_full_suppression(self) -> None:
        # 5+ consecutive identical HR values → flatline → full suppression (HR is hard-block)
        window = [self._row(HR=80.0) for _ in range(5)]
        result, _ = self._eval(window)
        self.assertFalse(result.signal_valid)
        self.assertEqual(result.artifact_type, "flatline")
        self.assertEqual(result.suppression_mode, "full")

    def test_flatline_in_soft_feature_triggers_partial_suppression(self) -> None:
        # 5+ consecutive identical Resp values → flatline → partial suppression (Resp is not hard-block).
        # O2Sat alternates so it does not trigger a hard flatline first.
        rows = [
            {"HR": float(70 + i), "SBP": float(115 + i), "DBP": 70.0,
             "O2Sat": float(96 + i % 2), "Resp": 16.0}
            for i in range(5)
        ]
        result, _ = self._eval(rows)
        self.assertTrue(result.signal_valid)
        self.assertEqual(result.artifact_type, "flatline")
        self.assertEqual(result.suppression_mode, "partial")

    def test_soft_range_violation_triggers_partial_suppression(self) -> None:
        # Temp=50°C is above the 43°C physiological upper bound → partial
        window = [self._row(), self._row(Temp=50.0)]
        result, _ = self._eval(window)
        self.assertTrue(result.signal_valid)
        self.assertEqual(result.artifact_type, "soft_range_violation")
        self.assertEqual(result.suppression_mode, "partial")

    def test_clean_window_passes_all_checks(self) -> None:
        # Vary HR, SBP, O2Sat, Resp so no flatline fires; keep jumps small so no
        # HR-jump or isolated-BP-drop check fires either.
        window = [
            self._row(HR=float(75 + i), SBP=float(115 + i),
                      O2Sat=float(96 + i % 3), Resp=float(14 + i % 3))
            for i in range(6)
        ]
        result, logs = self._eval(window)
        self.assertTrue(result.signal_valid)
        self.assertFalse(result.suppression_recommendation)
        self.assertEqual(result.suppression_mode, "none")
        self.assertTrue(any("passed" in l.message.lower() for l in logs))

    def test_spo2_paradox_uses_window_hr_stability(self) -> None:
        """SpO2/HR paradox check should fire when HR is stable over the full window, not just 2 rows."""
        # O2Sat alternates 74/75 (both < 80) to avoid flatline; HR varies by 2 BPM total.
        window = [
            self._row(HR=float(75 + i % 3), O2Sat=float(74 + i % 2))
            for i in range(6)
        ]
        result, _ = self._eval(window)
        self.assertEqual(result.artifact_type, "spo2_hr_paradox")
        self.assertEqual(result.suppression_mode, "partial")

    def test_persistent_low_spo2_triggers_partial_suppression(self) -> None:
        """SpO2 < 85% for 5+ rows with stable HR should trigger persistent_low_spo2 (partial)."""
        # O2Sat alternates 81/83 (above paradox threshold of 80, below 85) to avoid spo2_hr_paradox.
        # SBP and Resp are varied to avoid triggering the flatline check (step 8) before step 9.
        window = [
            self._row(HR=float(78 + i % 2), O2Sat=float(81 + i % 2 * 2),
                      SBP=float(115 + i), Resp=float(14 + i % 3))
            for i in range(5)
        ]
        result, _ = self._eval(window)
        self.assertEqual(result.artifact_type, "persistent_low_spo2")
        self.assertEqual(result.suppression_mode, "partial")

    def test_persistent_low_spo2_not_triggered_with_rising_hr(self) -> None:
        """Persistent low SpO2 should NOT fire if HR is rising (real deterioration, not probe-off)."""
        # HR rises 6 BPM/row → 24 BPM range over 5 rows, which exceeds the 20 BPM stability threshold.
        window = [
            self._row(HR=float(75 + i * 6), O2Sat=float(81 + i % 2 * 2))
            for i in range(5)
        ]
        result, _ = self._eval(window)
        self.assertNotEqual(result.artifact_type, "persistent_low_spo2")


# ── Unit: Calibrator application ──────────────────────────────────────────────

class TestCalibratorThreshold(unittest.TestCase):
    def test_sequence_threshold_in_calibrated_space(self) -> None:
        """decision_threshold must return calibrated value when calibrator is loaded."""
        from agentic_icu.api.dependencies import get_workflow
        wf = get_workflow()
        predictor = wf.vitals_agent.predictor
        predictor.load()
        if not predictor.calibrated:
            self.skipTest("Calibrator not loaded")
        threshold = predictor.decision_threshold
        self.assertIsNotNone(threshold)
        # Raw threshold is 0.00138 — calibrated threshold must differ and be > raw
        raw_threshold = predictor.metrics["threshold_selection"]["threshold"]
        self.assertNotAlmostEqual(threshold, raw_threshold, places=4,
            msg="Calibrated threshold should differ from raw threshold")

    def test_xgboost_threshold_in_calibrated_space(self) -> None:
        from agentic_icu.api.dependencies import get_workflow
        wf = get_workflow()
        predictor = wf.lab_agent.predictor
        predictor.load()
        if not predictor.calibrated:
            self.skipTest("Calibrator not loaded")
        threshold = predictor.decision_threshold
        self.assertIsNotNone(threshold)
        raw_threshold = predictor.metrics["threshold_selection"]["threshold"]
        # Both should be valid probabilities
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        self.assertNotEqual(threshold, raw_threshold)


# ── Integration: API ──────────────────────────────────────────────────────────

class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        get_workflow.cache_clear()
        cls.client = TestClient(app)

    def build_payload(self, patient_id: str, max_rows: int = 24) -> dict:
        window = _load_patient_window(patient_id, max_rows)
        self.assertGreaterEqual(len(window), 1)
        return {"patient_id": patient_id, "observation_window": window}

    def evaluate_patient(self, patient_id: str) -> dict:
        response = self.client.post("/evaluate", json=self.build_payload(patient_id))
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_dashboard_root_serves_html(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Agentic-ICU", response.text)

    def test_runtime_config_exposes_policy_and_thresholds(self) -> None:
        response = self.client.get("/runtime-config")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("alert_policy", payload)
        self.assertIn("model_thresholds", payload)
        # Resp thresholds must be present and match runtime_alert_policy.json
        self.assertEqual(payload["alert_policy"]["resp_high_alert_threshold"], 0.8)
        self.assertEqual(payload["alert_policy"]["resp_medium_alert_threshold"], 0.55)
        seq_thresh = payload["model_thresholds"]["sequence_threshold"]
        self.assertGreater(seq_thresh, 0.0)
        self.assertLess(seq_thresh, 1.0)

    def test_latest_alert_policy_report_returns_404_when_absent(self) -> None:
        response = self.client.get("/reports/alert-policy-latest")
        self.assertEqual(response.status_code, 404)

    def test_demo_patient_endpoint_returns_window(self) -> None:
        response = self.client.get("/demo-patient/p000018")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["patient_id"], "p000018")
        self.assertGreaterEqual(len(payload["observation_window"]), 1)

    def test_demo_patient_max_rows_upper_bound(self) -> None:
        """max_rows > MAX_WINDOW_ROWS must be rejected."""
        response = self.client.get("/demo-patient/p000018?max_rows=10000")
        self.assertEqual(response.status_code, 422)

    def test_health_reports_runtime_ready(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["preprocessing_ready"])
        self.assertTrue(payload["xgboost_ready"])
        self.assertTrue(payload["sequence_ready"])
        self.assertIn("resp_ready", payload)

    def test_evaluate_returns_real_model_scores(self) -> None:
        payload = self.evaluate_patient("p000001")
        self.assertEqual(payload["patient_id"], "p000001")
        self.assertEqual(payload["vitals_agent"]["status"], "available")
        self.assertEqual(payload["lab_agent"]["status"], "available")
        va = payload["vitals_agent"]
        la = payload["lab_agent"]
        self.assertGreaterEqual(va["score"], 0.0)
        self.assertLessEqual(va["score"], 1.0)
        self.assertGreaterEqual(la["score"], 0.0)
        self.assertLessEqual(la["score"], 1.0)
        self.assertIsNotNone(va["decision_threshold"])
        self.assertIsNotNone(la["decision_threshold"])
        # Scores and thresholds must be in same probability space (both in [0,1])
        self.assertGreater(va["decision_threshold"], 0.0)
        self.assertLess(va["decision_threshold"], 1.0)
        self.assertGreaterEqual(len(payload["reasoning_log"]), 4)

    def test_evaluate_resp_agent_present(self) -> None:
        payload = self.evaluate_patient("p000001")
        resp = payload["resp_failure_agent"]
        self.assertEqual(resp["status"], "available")
        self.assertIsNotNone(resp["score"])
        self.assertGreaterEqual(resp["score"], 0.0)
        self.assertLessEqual(resp["score"], 1.0)
        self.assertIn("Resp GRU", resp["detail"])

    def test_full_suppression_skips_model_inference(self) -> None:
        """p000011 has a signal artifact — vitals/lab must be unavailable (skipped)."""
        payload = self.evaluate_patient("p000011")
        self.assertFalse(payload["clinical_decision"]["alert_triggered"])
        self.assertEqual(payload["clinical_decision"]["alert_type"], "Suppressed Artifact")
        # With the workflow short-circuit fix, suppressed patients get status=unavailable
        self.assertEqual(payload["vitals_agent"]["status"], "unavailable")
        self.assertEqual(payload["lab_agent"]["status"], "unavailable")

    def test_shap_contributions_populated_on_evaluate(self) -> None:
        payload = self.evaluate_patient("p000018")
        lab = payload["lab_agent"]
        self.assertGreater(len(lab["feature_contributions"]), 0)
        self.assertIsInstance(lab["explanation"], str)
        self.assertGreater(len(lab["explanation"]), 0)
        rationale = payload["clinical_decision"]["rationale"]
        self.assertIn("lab signal", rationale.lower())

    def test_temporal_saliency_populated_and_not_uniform(self) -> None:
        """Saliency weights must sum to ~1.0 and not all be identical (uniform fallback = broken)."""
        payload = self.evaluate_patient("p000018")
        vitals = payload["vitals_agent"]
        t_keys = [k for k in vitals["feature_contributions"] if k.startswith("t_")]
        self.assertGreater(len(t_keys), 0)
        weights: List[float] = [vitals["feature_contributions"][k] for k in t_keys]
        self.assertFalse(
            any(math.isnan(w) or math.isinf(w) for w in weights),
            "Saliency weights contain NaN or Inf — backward pass likely failed",
        )
        total = sum(weights)
        self.assertAlmostEqual(total, 1.0, places=2)
        # Uniform fallback produces identical values; real gradients vary
        self.assertGreater(max(weights) - min(weights), 1e-6, "Saliency weights are uniform — backward pass may have failed")
        self.assertIsInstance(vitals["explanation"], str)
        self.assertGreater(len(vitals["explanation"]), 0)

    def test_resp_failure_temporal_saliency_populated_and_not_uniform(self) -> None:
        payload = self.evaluate_patient("p000001")
        resp = payload["resp_failure_agent"]
        t_keys = [k for k in resp["feature_contributions"] if k.startswith("t_")]
        self.assertGreater(len(t_keys), 0)
        weights: List[float] = [resp["feature_contributions"][k] for k in t_keys]
        self.assertFalse(
            any(math.isnan(w) or math.isinf(w) for w in weights),
            "Resp saliency weights contain NaN or Inf",
        )
        total = sum(weights)
        self.assertAlmostEqual(total, 1.0, places=2)
        self.assertGreater(max(weights) - min(weights), 1e-6, "Resp saliency is uniform — backward pass may have failed")

    def test_patients_search_bounds(self) -> None:
        """Negative offset and zero limit must be rejected."""
        self.assertEqual(self.client.get("/patients?offset=-1").status_code, 422)
        self.assertEqual(self.client.get("/patients?limit=0").status_code, 422)

    def test_partial_suppression_penalizes_scores_in_api_response(self) -> None:
        """Partial suppression (soft range violation) must produce suppression-adjusted scores in the response payload."""
        # Temp=50°C is above the 43°C physiological upper bound → soft_range_violation → partial suppression.
        # HR/SBP/O2Sat are in-range so full suppression does not fire first.
        window = [
            {"values": {"HR": 80.0, "SBP": 120.0, "DBP": 70.0, "O2Sat": 97.0,
                        "Resp": 16.0, "Temp": 50.0, "ICULOS": float(i + 1)}}
            for i in range(2)
        ]
        response = self.client.post("/evaluate", json={"patient_id": "partial_suppression_test", "observation_window": window})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        sq = body["signal_quality"]
        self.assertEqual(sq["suppression_mode"], "partial", "Signal quality should be partial — check window construction")
        va = body["vitals_agent"]
        la = body["lab_agent"]
        self.assertEqual(va["status"], "available")
        self.assertEqual(la["status"], "available")
        # Scores must carry suppression-adjusted detail strings
        self.assertIn("suppression-adjusted", va["detail"])
        self.assertIn("suppression-adjusted", la["detail"])
        # threshold_ratio must be consistent with score/threshold (server-computed)
        if va["score"] is not None and va["decision_threshold"]:
            self.assertAlmostEqual(va["threshold_ratio"], va["score"] / va["decision_threshold"], places=4)

    def test_evaluate_ensemble_agent_in_response(self) -> None:
        """ensemble_agent must appear in /evaluate response with status and score in [0,1]."""
        payload = self.evaluate_patient("p000001")
        self.assertIn("ensemble_agent", payload, "ensemble_agent field missing from /evaluate response")
        ea = payload["ensemble_agent"]
        self.assertIn("status", ea)
        if ea["status"] == "available":
            self.assertIsNotNone(ea["score"])
            self.assertGreaterEqual(ea["score"], 0.0)
            self.assertLessEqual(ea["score"], 1.0)
            self.assertIn(ea["risk_band"], ("low", "moderate", "high"))

    def test_ensemble_medium_alert_triggered_by_threshold(self) -> None:
        """Ensemble score >= medium threshold must trigger a medium alert via the reasoner."""
        from agentic_icu.agents.reasoner import AlertPolicy, ClinicalReasoner
        from agentic_icu.domain.contracts import ModelAgentResult, SignalQualityResult

        policy = AlertPolicy(
            high_alert_ensemble_score_threshold=0.80,
            medium_alert_ensemble_score_threshold=0.55,
            high_alert_extreme_sequence_score_threshold=None,
            high_alert_supported_sequence_score_threshold=None,
            medium_alert_sequence_score_threshold=None,
            medium_alert_tabular_score_threshold=None,
        )
        reasoner = ClinicalReasoner(policy)
        sq = SignalQualityResult(signal_valid=True, suppression_recommendation=False, suppression_mode="none")
        unavailable = ModelAgentResult(status="unavailable", detail="n/a")

        decision, logs, *_ = reasoner.decide(sq, unavailable, unavailable, ensemble_score=0.70)
        self.assertTrue(decision.alert_triggered)
        self.assertIn("ensemble", decision.rationale.lower())

    def test_ensemble_high_alert_triggered_by_threshold(self) -> None:
        """Ensemble score >= high threshold must trigger a high alert."""
        from agentic_icu.agents.reasoner import AlertPolicy, ClinicalReasoner
        from agentic_icu.domain.contracts import ModelAgentResult, SignalQualityResult

        policy = AlertPolicy(
            high_alert_ensemble_score_threshold=0.80,
            medium_alert_ensemble_score_threshold=0.55,
            high_alert_extreme_sequence_score_threshold=None,
            high_alert_supported_sequence_score_threshold=None,
            medium_alert_sequence_score_threshold=None,
            medium_alert_tabular_score_threshold=None,
        )
        reasoner = ClinicalReasoner(policy)
        sq = SignalQualityResult(signal_valid=True, suppression_recommendation=False, suppression_mode="none")
        unavailable = ModelAgentResult(status="unavailable", detail="n/a")

        decision, logs, *_ = reasoner.decide(sq, unavailable, unavailable, ensemble_score=0.90)
        self.assertTrue(decision.alert_triggered)
        self.assertEqual(decision.priority, "high")
        self.assertIn("ensemble", decision.rationale.lower())

    def test_ensemble_partial_suppression_penalty(self) -> None:
        """Partial suppression must apply 0.7x penalty to ensemble score before alert check."""
        from agentic_icu.agents.reasoner import AlertPolicy, ClinicalReasoner
        from agentic_icu.domain.contracts import ModelAgentResult, SignalQualityResult

        policy = AlertPolicy(
            partial_suppression_factor=0.7,
            high_alert_ensemble_score_threshold=0.80,
            medium_alert_ensemble_score_threshold=0.55,
            high_alert_extreme_sequence_score_threshold=None,
            high_alert_supported_sequence_score_threshold=None,
            medium_alert_sequence_score_threshold=None,
            medium_alert_tabular_score_threshold=None,
        )
        reasoner = ClinicalReasoner(policy)
        sq = SignalQualityResult(
            signal_valid=True, suppression_recommendation=True, suppression_mode="partial",
            artifact_type="soft_range_violation", artifact_affected_features=["Temp"],
        )
        unavailable = ModelAgentResult(status="unavailable", detail="n/a")

        # 0.90 * 0.7 = 0.63 → still >= 0.55 (medium), but < 0.80 (high)
        decision, *_ = reasoner.decide(sq, unavailable, unavailable, ensemble_score=0.90)
        self.assertTrue(decision.alert_triggered)
        self.assertNotEqual(decision.priority, "high")  # penalty should drop it from high

        # 0.70 * 0.7 = 0.49 → < 0.55, should NOT trigger
        decision2, *_ = reasoner.decide(sq, unavailable, unavailable, ensemble_score=0.70)
        self.assertFalse(decision2.alert_triggered)

    def test_ensemble_metrics_in_model_metrics_endpoint(self) -> None:
        """Ensemble must appear in /model-metrics with auc, auprc, and formula."""
        response = self.client.get("/model-metrics")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("ensemble", body, "ensemble key missing from /model-metrics")
        ens = body["ensemble"]
        self.assertIn("metrics", ens)
        self.assertIn("auc", ens["metrics"])
        self.assertIn("average_precision", ens["metrics"])
        self.assertIn("formula", ens)
        self.assertGreater(ens["metrics"]["auc"], 0.0)

    def test_model_metrics_endpoint(self) -> None:
        """All four models must appear with valid AUC, AUPRC, and F1 values."""
        response = self.client.get("/model-metrics")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in ("sepsis_gru", "sepsis_xgb", "resp_gru", "resp_xgb"):
            self.assertIn(key, body, f"Missing model key: {key}")
            m = body[key]
            self.assertIn("metrics", m)
            self.assertIn("name", m)
            for metric in ("auc", "average_precision", "f1", "threshold"):
                self.assertIn(metric, m["metrics"], f"{key}: missing metric {metric}")
            self.assertGreater(m["metrics"]["auc"], 0.0)
            self.assertLessEqual(m["metrics"]["auc"], 1.0)

    def test_demo_patients_endpoint(self) -> None:
        """The demo patient pool must return a non-empty list with id, label, and tone fields."""
        response = self.client.get("/demo-patients")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("patients", body)
        self.assertIsInstance(body["patients"], list)
        self.assertGreater(len(body["patients"]), 0, "No demo patients available in data dir")
        for p in body["patients"]:
            self.assertIn("id", p)
            self.assertIn("label", p)
            self.assertIn("tone", p)

    def test_explain_respects_signal_quality_suppression(self) -> None:
        """On a fully suppressed window /explain must return unavailable explanations, not SHAP on corrupt data."""
        window = _load_patient_window("p000011", 24)
        response = self.client.post("/explain", json={"patient_id": "p000011", "observation_window": window})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["lab_explanation"]["status"], "unavailable")
        self.assertEqual(body["vitals_explanation"]["status"], "unavailable")
        self.assertIn("suppressed", body["lab_explanation"]["explanation"].lower())


# ── Integration: API error paths ──────────────────────────────────────────────

class ApiErrorPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        get_workflow.cache_clear()
        cls.client = TestClient(app)

    def test_empty_observation_window_returns_422(self) -> None:
        response = self.client.post("/evaluate", json={"patient_id": "p000001", "observation_window": []})
        self.assertEqual(response.status_code, 422)
        self.assertIn("error", response.json())

    def test_oversized_observation_window_returns_422(self) -> None:
        window = [{"values": {"HR": 80.0}} for _ in range(169)]
        response = self.client.post("/evaluate", json={"patient_id": "p000001", "observation_window": window})
        self.assertEqual(response.status_code, 422)
        self.assertIn("error", response.json())

    def test_blank_patient_id_returns_422(self) -> None:
        response = self.client.post("/evaluate", json={"patient_id": "   ", "observation_window": [{"values": {"HR": 80.0}}]})
        self.assertEqual(response.status_code, 422)
        self.assertIn("error", response.json())

    def test_missing_demo_patient_returns_404(self) -> None:
        response = self.client.get("/demo-patient/p_xyzzy_does_not_exist")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.json())

    def test_malformed_json_returns_422(self) -> None:
        response = self.client.post("/evaluate", content=b"not-valid-json", headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)

    def test_500_does_not_leak_internal_details(self) -> None:
        """The global 500 handler must not expose stack traces or internal paths."""
        from unittest.mock import patch
        # raise_server_exceptions=False prevents TestClient from re-raising; lets
        # the FastAPI exception handler return the 500 JSON response instead.
        safe_client = TestClient(app, raise_server_exceptions=False)
        with patch.object(get_workflow(), "evaluate", side_effect=RuntimeError("secret internal path /models/weights")):
            response = safe_client.post("/evaluate", json={"patient_id": "p000001", "observation_window": [{"values": {"HR": 80.0}}]})
        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertNotIn("secret internal path", str(body))
        self.assertIn("request_id", body)

    def test_explain_endpoint_returns_contributions(self) -> None:
        window = _load_patient_window("p000018", 24)
        response = self.client.post("/explain", json={"patient_id": "p000018", "observation_window": window})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["lab_explanation"]["status"], "available")
        self.assertEqual(body["vitals_explanation"]["status"], "available")
        self.assertGreater(len(body["vitals_explanation"]["feature_contributions"]), 0)
        self.assertGreater(len(body["lab_explanation"]["feature_contributions"]), 0)

    def test_invalid_patient_id_format_returns_422(self) -> None:
        """Patient IDs with path traversal or special chars must be rejected before hitting filesystem."""
        for bad_id in ("../etc/passwd", "p000001; rm -rf /", "p 001"):
            response = self.client.get(f"/demo-patient/{bad_id}")
            self.assertIn(response.status_code, (404, 422), f"Expected 404/422 for id {bad_id!r}")

    def test_security_headers_present(self) -> None:
        """Security headers must be set on all responses."""
        response = self.client.get("/health")
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
        self.assertIn("strict-origin", response.headers.get("Referrer-Policy", ""))
        csp = response.headers.get("Content-Security-Policy", "")
        self.assertIn("default-src 'self'", csp)
        self.assertIn("object-src 'none'", csp)
        self.assertIn("frame-ancestors 'none'", csp)

    def test_rate_limit_returns_429(self) -> None:
        """Exceeding rate limit on /evaluate must return 429 with Retry-After header."""
        import agentic_icu.api.main as main_mod
        orig = main_mod._RL_MAX_REQUESTS
        main_mod._RL_MAX_REQUESTS = 2
        main_mod._rl_counters.clear()
        client = TestClient(app, raise_server_exceptions=False)
        window = [{"values": {"HR": 80.0, "ICULOS": 1.0}}]
        body = {"patient_id": "rl_test", "observation_window": window}
        try:
            # First two should pass (or fail for model reasons, not 429)
            r1 = client.post("/evaluate", json=body)
            r2 = client.post("/evaluate", json=body)
            self.assertNotEqual(r1.status_code, 429)
            self.assertNotEqual(r2.status_code, 429)
            # Third must be 429
            r3 = client.post("/evaluate", json=body)
            self.assertEqual(r3.status_code, 429)
            self.assertIn("Retry-After", r3.headers)
        finally:
            main_mod._RL_MAX_REQUESTS = orig
            main_mod._rl_counters.clear()

    def test_explain_rate_limit_returns_429(self) -> None:
        """Rate limit must also apply to /explain, not just /evaluate."""
        import agentic_icu.api.main as main_mod
        orig = main_mod._RL_MAX_REQUESTS
        main_mod._RL_MAX_REQUESTS = 2
        main_mod._rl_counters.clear()
        client = TestClient(app, raise_server_exceptions=False)
        window = _load_patient_window("p000018", 4)
        body = {"patient_id": "p000018", "observation_window": window}
        try:
            r1 = client.post("/explain", json=body)
            r2 = client.post("/explain", json=body)
            self.assertNotEqual(r1.status_code, 429)
            self.assertNotEqual(r2.status_code, 429)
            r3 = client.post("/explain", json=body)
            self.assertEqual(r3.status_code, 429)
            self.assertIn("Retry-After", r3.headers)
        finally:
            main_mod._RL_MAX_REQUESTS = orig
            main_mod._rl_counters.clear()

    def test_rate_limit_prunes_empty_counters(self) -> None:
        """After all timestamps expire, the counter key must be pruned from the dict."""
        import agentic_icu.api.main as main_mod
        import time as time_mod
        main_mod._rl_counters.clear()
        window = [{"values": {"HR": 80.0, "ICULOS": 1.0}}]
        body = {"patient_id": "prune_test", "observation_window": window}
        client = TestClient(app, raise_server_exceptions=False)
        # Make one request — a key should be created
        client.post("/evaluate", json=body)
        # Manually expire all timestamps by pushing them past the window
        with main_mod._rl_lock:
            for q in main_mod._rl_counters.values():
                for _ in range(len(q)):
                    q.popleft()
                    q.appendleft(time_mod.monotonic() - main_mod._RL_WINDOW_S - 1)
        # Next request should prune the stale key then re-create it
        client.post("/evaluate", json=body)
        # After pruning+re-creation, at most 1 entry per key
        with main_mod._rl_lock:
            for q in main_mod._rl_counters.values():
                self.assertLessEqual(len(q), 1)
        main_mod._rl_counters.clear()


# ── Auth middleware ───────────────────────────────────────────────────────────

class ApiKeyMiddlewareTests(unittest.TestCase):
    def setUp(self) -> None:
        import agentic_icu.api.main as main_mod
        self._orig = main_mod._API_KEY
        main_mod._API_KEY = "test-secret-key"
        get_workflow.cache_clear()
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        import agentic_icu.api.main as main_mod
        main_mod._API_KEY = self._orig

    def test_missing_key_returns_401(self) -> None:
        response = self.client.get("/patients")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"], "unauthorized")

    def test_wrong_key_returns_401(self) -> None:
        response = self.client.get("/patients", headers={"X-API-Key": "wrong-key"})
        self.assertEqual(response.status_code, 401)

    def test_correct_key_is_accepted(self) -> None:
        response = self.client.get("/patients", headers={"X-API-Key": "test-secret-key"})
        self.assertEqual(response.status_code, 200)

    def test_health_is_exempt_from_auth(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_static_is_exempt_from_auth(self) -> None:
        response = self.client.get("/static/app.css")
        self.assertNotEqual(response.status_code, 401)

    def test_no_key_env_var_allows_all_requests(self) -> None:
        import agentic_icu.api.main as main_mod
        main_mod._API_KEY = ""
        response = self.client.get("/patients")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
