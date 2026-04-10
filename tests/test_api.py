from __future__ import annotations

import csv
import math
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_icu.api.dependencies import get_workflow
from agentic_icu.api.main import app


class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        get_workflow.cache_clear()
        cls.client = TestClient(app)

    def build_payload(self, patient_id: str, max_rows: int = 24) -> dict:
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
                    numeric_value = float(value)
                    if math.isfinite(numeric_value):
                        values[key] = numeric_value
                window.append({"values": values})

        self.assertGreaterEqual(len(window), 1)
        return {
            "patient_id": patient_id,
            "observation_window": window,
        }

    def evaluate_patient(self, patient_id: str) -> dict:
        response = self.client.post("/evaluate", json=self.build_payload(patient_id))
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_dashboard_root_serves_html(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Agentic-ICU", response.text)
        self.assertIn("Board Controls", response.text)
        self.assertIn("Threshold Ratio", response.text)
        self.assertIn("Runtime policy details", response.text)
        self.assertIn("Calibration Review", response.text)
        self.assertIn("Recommendation notes will appear here.", response.text)

    def test_runtime_config_exposes_policy_and_thresholds(self) -> None:
        response = self.client.get("/runtime-config")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertIn("alert_policy", payload)
        self.assertIn("model_thresholds", payload)
        self.assertIsNone(payload["alert_policy"]["high_alert_max_score_threshold"])
        self.assertIsNone(payload["alert_policy"]["high_alert_mean_score_threshold"])
        self.assertEqual(payload["alert_policy"]["high_alert_extreme_sequence_score_threshold"], 0.88)
        self.assertEqual(payload["alert_policy"]["high_alert_supported_sequence_score_threshold"], 0.8)
        self.assertEqual(payload["alert_policy"]["high_alert_tabular_support_score_threshold"], 0.25)
        self.assertIsNone(payload["alert_policy"]["medium_alert_max_score_threshold"])
        self.assertEqual(payload["alert_policy"]["medium_alert_sequence_score_threshold"], 0.55)
        self.assertEqual(payload["alert_policy"]["medium_alert_tabular_score_threshold"], 0.4)
        # v2 GRU (AUC=0.92, AP=0.76) optimises at a lower threshold — just verify it's a valid float
        seq_thresh = payload["model_thresholds"]["sequence_threshold"]
        self.assertGreater(seq_thresh, 0.0)
        self.assertLess(seq_thresh, 1.0)
        self.assertGreater(payload["model_thresholds"]["xgboost_threshold"], 0.5)

    def test_latest_alert_policy_report_returns_404_when_absent(self) -> None:
        # v2 retrain clears the reports dir; endpoint should 404 gracefully until a new report is generated
        response = self.client.get("/reports/alert-policy-latest")
        self.assertEqual(response.status_code, 404)

    def test_demo_patient_endpoint_returns_window(self) -> None:
        response = self.client.get("/demo-patient/p000018")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["patient_id"], "p000018")
        self.assertGreaterEqual(len(payload["observation_window"]), 1)
        self.assertIn("values", payload["observation_window"][0])

    def test_health_reports_runtime_ready(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["preprocessing_ready"])
        self.assertTrue(payload["xgboost_ready"])
        self.assertTrue(payload["sequence_ready"])

    def test_evaluate_returns_real_model_scores(self) -> None:
        payload = self.evaluate_patient("p000001")

        self.assertEqual(payload["patient_id"], "p000001")
        self.assertEqual(payload["vitals_agent"]["status"], "available")
        self.assertEqual(payload["lab_agent"]["status"], "available")
        self.assertGreaterEqual(payload["vitals_agent"]["score"], 0.0)
        self.assertLessEqual(payload["vitals_agent"]["score"], 1.0)
        self.assertGreaterEqual(payload["lab_agent"]["score"], 0.0)
        self.assertLessEqual(payload["lab_agent"]["score"], 1.0)
        self.assertIsNotNone(payload["vitals_agent"]["decision_threshold"])
        self.assertIsNotNone(payload["lab_agent"]["decision_threshold"])
        self.assertGreaterEqual(len(payload["reasoning_log"]), 5)

    def test_stable_patient_remains_below_alert_thresholds(self) -> None:
        payload = self.evaluate_patient("p000001")

        self.assertFalse(payload["clinical_decision"]["alert_triggered"])
        self.assertEqual(payload["clinical_decision"]["alert_type"], "Stable")
        self.assertEqual(payload["clinical_decision"]["priority"], "low")
        self.assertLess(payload["vitals_agent"]["score"], 0.55)
        self.assertLess(payload["lab_agent"]["score"], 0.4)

    def test_medium_risk_patient_triggers_deterioration_watch(self) -> None:
        payload = self.evaluate_patient("p000026")

        self.assertTrue(payload["clinical_decision"]["alert_triggered"])
        self.assertEqual(payload["clinical_decision"]["alert_type"], "Deterioration Watch")
        self.assertEqual(payload["clinical_decision"]["priority"], "medium")
        self.assertGreaterEqual(payload["vitals_agent"]["score"], 0.55)
        self.assertLess(payload["vitals_agent"]["score"], 0.8)

    def test_signal_quality_suppression_overrides_high_vitals_score(self) -> None:
        # p000011 has vitals score ~0.99 but signal quality agent flags artifact — alert must be suppressed
        payload = self.evaluate_patient("p000011")

        self.assertFalse(payload["clinical_decision"]["alert_triggered"])
        self.assertEqual(payload["clinical_decision"]["alert_type"], "Suppressed Artifact")
        self.assertEqual(payload["clinical_decision"]["priority"], "low")
        self.assertGreaterEqual(payload["vitals_agent"]["score"], 0.88)

    def test_high_risk_patient_triggers_sepsis_early_warning(self) -> None:
        payload = self.evaluate_patient("p000028")

        self.assertTrue(payload["clinical_decision"]["alert_triggered"])
        self.assertEqual(payload["clinical_decision"]["alert_type"], "Sepsis Early Warning")
        self.assertEqual(payload["clinical_decision"]["priority"], "high")
        self.assertGreaterEqual(payload["vitals_agent"]["score"], 0.88)

    def test_shap_contributions_populated_on_evaluate(self) -> None:
        """Lab agent must return SHAP feature contributions and a non-empty explanation."""
        payload = self.evaluate_patient("p000018")

        lab = payload["lab_agent"]
        # feature_contributions has top-3 SHAP values
        self.assertGreater(len(lab["feature_contributions"]), 0)
        # explanation is a non-empty human-readable string
        self.assertIsInstance(lab["explanation"], str)
        self.assertGreater(len(lab["explanation"]), 0)
        # rationale now includes the SHAP-driven explanation
        rationale = payload["clinical_decision"]["rationale"]
        self.assertIn("lab signal", rationale.lower())

    def test_temporal_saliency_populated_on_evaluate(self) -> None:
        """Vitals agent must return per-timestep saliency weights and a non-empty explanation."""
        payload = self.evaluate_patient("p000018")

        vitals = payload["vitals_agent"]
        # feature_contributions has t_01 … t_N keys
        t_keys = [k for k in vitals["feature_contributions"] if k.startswith("t_")]
        self.assertGreater(len(t_keys), 0)
        # weights should sum to approximately 1.0
        total = sum(vitals["feature_contributions"][k] for k in t_keys)
        self.assertAlmostEqual(total, 1.0, places=2)
        # explanation is populated
        self.assertIsInstance(vitals["explanation"], str)
        self.assertGreater(len(vitals["explanation"]), 0)


    def test_resp_failure_agent_present_in_response(self) -> None:
        payload = self.evaluate_patient("p000001")

        resp = payload["resp_failure_agent"]
        self.assertIn("status", resp)
        self.assertIn("score", resp)
        self.assertIn("risk_band", resp)
        self.assertIn("decision_threshold", resp)
        self.assertIn("detail", resp)
        self.assertEqual(resp["status"], "available")
        self.assertIsNotNone(resp["score"])
        self.assertGreaterEqual(resp["score"], 0.0)
        self.assertLessEqual(resp["score"], 1.0)
        self.assertIsNotNone(resp["decision_threshold"])
        self.assertGreater(resp["decision_threshold"], 0.0)
        self.assertIn("Resp GRU", resp["detail"])

    def test_resp_elevated_patient_triggers_resp_failure_alert(self) -> None:
        # p000312 has resp GRU score ~0.89 (above resp_high_alert_threshold=0.8)
        payload = self.evaluate_patient("p000312")

        resp = payload["resp_failure_agent"]
        self.assertGreaterEqual(resp["score"], 0.8)
        self.assertTrue(payload["clinical_decision"]["alert_triggered"])
        self.assertEqual(payload["clinical_decision"]["alert_type"], "Respiratory Failure Risk")
        self.assertEqual(payload["clinical_decision"]["priority"], "high")

    def test_resp_failure_temporal_saliency_populated(self) -> None:
        payload = self.evaluate_patient("p000001")

        resp = payload["resp_failure_agent"]
        t_keys = [k for k in resp["feature_contributions"] if k.startswith("t_")]
        self.assertGreater(len(t_keys), 0)
        total = sum(resp["feature_contributions"][k] for k in t_keys)
        self.assertAlmostEqual(total, 1.0, places=2)
        self.assertIsInstance(resp["explanation"], str)
        self.assertGreater(len(resp["explanation"]), 0)


class ApiErrorPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        get_workflow.cache_clear()
        cls.client = TestClient(app)

    def test_empty_observation_window_returns_422(self) -> None:
        response = self.client.post("/evaluate", json={
            "patient_id": "p000001",
            "observation_window": [],
        })
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertIn("error", body)

    def test_oversized_observation_window_returns_422(self) -> None:
        # 169 rows exceeds MAX_WINDOW_ROWS = 168
        window = [{"values": {"HR": 80.0}} for _ in range(169)]
        response = self.client.post("/evaluate", json={
            "patient_id": "p000001",
            "observation_window": window,
        })
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertIn("error", body)

    def test_blank_patient_id_returns_422(self) -> None:
        response = self.client.post("/evaluate", json={
            "patient_id": "   ",
            "observation_window": [{"values": {"HR": 80.0}}],
        })
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertIn("error", body)

    def test_missing_demo_patient_returns_404(self) -> None:
        response = self.client.get("/demo-patient/p_does_not_exist")
        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertIn("error", body)

    def test_malformed_json_returns_422(self) -> None:
        response = self.client.post(
            "/evaluate",
            content=b"not-valid-json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)

    def test_explain_endpoint_returns_contributions(self) -> None:
        """POST /explain returns lab and vitals explanations without a full decision."""
        ROOT = Path(__file__).resolve().parents[1]
        patient_path = ROOT / "data" / "raw" / "p000018.psv"
        window = []
        with patient_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            for index, row in enumerate(reader):
                if index >= 24:
                    break
                values = {}
                for key, value in row.items():
                    if key == "SepsisLabel" or value in (None, ""):
                        continue
                    numeric_value = float(value)
                    if math.isfinite(numeric_value):
                        values[key] = numeric_value
                window.append({"values": values})

        response = self.client.post("/explain", json={
            "patient_id": "p000018",
            "observation_window": window,
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("lab_explanation", body)
        self.assertIn("vitals_explanation", body)
        self.assertEqual(body["lab_explanation"]["status"], "available")
        self.assertEqual(body["vitals_explanation"]["status"], "available")
        self.assertGreater(len(body["vitals_explanation"]["feature_contributions"]), 0)


if __name__ == "__main__":
    unittest.main()


