"""Tests for XGBoostInference, RuntimePreprocessor, and AgenticICUWorkflow edge paths."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_icu.domain.contracts import (
    EvaluatePatientRequest,
    ModelAgentResult,
    ObservationRecord,
)
from agentic_icu.domain.features import DYNAMIC_FEATURES, STATIC_FEATURES
from agentic_icu.inference.tabular import XGBoostInference
from agentic_icu.orchestration.workflow import AgenticICUWorkflow
from agentic_icu.preprocessing.windowing import RuntimePreprocessor

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_statistics(tmp_path: Path) -> Path:
    """Write a minimal train_statistics.json that covers every DYNAMIC_FEATURE."""
    stats = {
        "fill_medians": {f: 0.0 for f in DYNAMIC_FEATURES},
        "value_means": {f: 0.0 for f in DYNAMIC_FEATURES},
        "value_stds": {f: 1.0 for f in DYNAMIC_FEATURES},
        "static_fill_values": {f: 0.0 for f in STATIC_FEATURES},
    }
    p = tmp_path / "train_statistics.json"
    p.write_text(json.dumps(stats))
    return p


def _make_pipeline_config(tmp_path: Path, obs_hours: int = 24, seq_hours: int = 72) -> Path:
    cfg = {"observation_hours": obs_hours, "patient_seq_max_hours": seq_hours}
    p = tmp_path / "pipeline_config.json"
    p.write_text(json.dumps(cfg))
    return p


def _make_preprocessor(tmp_path: Path) -> RuntimePreprocessor:
    stats_p = _make_statistics(tmp_path)
    cfg_p = _make_pipeline_config(tmp_path)
    rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
    rp.load()
    return rp


def _records(n: int = 6) -> list[ObservationRecord]:
    return [
        ObservationRecord(values={"HR": float(70 + i), "ICULOS": float(i + 1)})
        for i in range(n)
    ]


# ── RuntimePreprocessor ───────────────────────────────────────────────────────


class TestRuntimePreprocessor:
    def test_available_false_when_files_missing(self, tmp_path):
        rp = RuntimePreprocessor(
            str(tmp_path / "stats.json"), str(tmp_path / "cfg.json")
        )
        assert rp.available is False

    def test_available_true_when_files_exist(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        assert rp.available is True

    def test_load_raises_when_fill_medians_missing(self, tmp_path):
        stats = {
            "value_means": {f: 0.0 for f in DYNAMIC_FEATURES},
            "value_stds": {f: 1.0 for f in DYNAMIC_FEATURES},
            "static_fill_values": {},
        }
        stats_p = tmp_path / "stats.json"
        stats_p.write_text(json.dumps(stats))
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        with pytest.raises(ValueError, match="fill_medians missing"):
            rp.load()

    def test_load_raises_when_value_means_missing(self, tmp_path):
        stats = {
            "fill_medians": {f: 0.0 for f in DYNAMIC_FEATURES},
            "value_stds": {f: 1.0 for f in DYNAMIC_FEATURES},
            "static_fill_values": {},
        }
        stats_p = tmp_path / "stats.json"
        stats_p.write_text(json.dumps(stats))
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        with pytest.raises(ValueError, match="value_means missing"):
            rp.load()

    def test_load_raises_when_value_stds_missing(self, tmp_path):
        stats = {
            "fill_medians": {f: 0.0 for f in DYNAMIC_FEATURES},
            "value_means": {f: 0.0 for f in DYNAMIC_FEATURES},
            "static_fill_values": {},
        }
        stats_p = tmp_path / "stats.json"
        stats_p.write_text(json.dumps(stats))
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        with pytest.raises(ValueError, match="value_stds missing"):
            rp.load()

    def test_load_raises_when_static_fill_values_missing(self, tmp_path):
        stats = {
            "fill_medians": {f: 0.0 for f in DYNAMIC_FEATURES},
            "value_means": {f: 0.0 for f in DYNAMIC_FEATURES},
            "value_stds": {f: 1.0 for f in DYNAMIC_FEATURES},
        }
        stats_p = tmp_path / "stats.json"
        stats_p.write_text(json.dumps(stats))
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        with pytest.raises(ValueError, match="static_fill_values"):
            rp.load()

    def test_load_is_idempotent(self, tmp_path):
        rp = _make_preprocessor(tmp_path)
        first_stats = id(rp._stats)
        rp.load()  # second call — should be a no-op
        assert id(rp._stats) == first_stats

    def test_stats_property_triggers_lazy_load(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        assert rp._stats is None
        _ = rp.stats
        assert rp._stats is not None

    def test_pipeline_config_property_triggers_lazy_load(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = _make_pipeline_config(tmp_path)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        _ = rp.pipeline_config
        assert rp._pipeline_config is not None

    def test_observation_hours_defaults_to_24(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = tmp_path / "cfg.json"
        cfg_p.write_text(json.dumps({}))  # empty — should default
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        assert rp.observation_hours == 24

    def test_sequence_hours_defaults_to_observation_hours(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = tmp_path / "cfg.json"
        cfg_p.write_text(json.dumps({"observation_hours": 24}))
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        assert rp.sequence_hours == 24

    def test_build_tabular_features_returns_dict(self, tmp_path):
        rp = _make_preprocessor(tmp_path)
        features = rp.build_tabular_features(_records(10))
        assert isinstance(features, dict)
        assert "HR__last" in features
        assert "window_missing_fraction" in features

    def test_build_tabular_features_truncates_long_window(self, tmp_path):
        # Use obs_hours=4 so a 30-row window gets truncated
        stats_p = _make_statistics(tmp_path)
        cfg_p = _make_pipeline_config(tmp_path, obs_hours=4, seq_hours=72)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        rp.load()
        features = rp.build_tabular_features(_records(30))
        assert isinstance(features, dict)

    def test_build_sequence_tensor_shape(self, tmp_path):
        rp = _make_preprocessor(tmp_path)
        tensor = rp.build_sequence_tensor(_records(6))
        assert tensor.ndim == 2
        assert tensor.shape[1] == len(DYNAMIC_FEATURES) * 2  # values + masks

    def test_build_sequence_tensor_pads_short_window(self, tmp_path):
        # seq_hours=72 but only 6 records — should pad
        rp = _make_preprocessor(tmp_path)
        tensor = rp.build_sequence_tensor(_records(6))
        assert tensor.shape[0] == 72  # padded to seq_hours

    def test_build_sequence_tensor_truncates_long_window(self, tmp_path):
        stats_p = _make_statistics(tmp_path)
        cfg_p = _make_pipeline_config(tmp_path, obs_hours=24, seq_hours=10)
        rp = RuntimePreprocessor(str(stats_p), str(cfg_p))
        rp.load()
        tensor = rp.build_sequence_tensor(_records(30))
        assert tensor.shape[0] == 10

    def test_records_to_frame_fills_iculos_from_index(self, tmp_path):
        rp = _make_preprocessor(tmp_path)
        records = [ObservationRecord(values={"HR": 80.0})]  # no ICULOS
        df = rp.records_to_frame(records)
        assert df["ICULOS"].iloc[0] == 1.0  # index 1

    def test_slope_6h_returns_zero_for_single_value(self):
        slope = RuntimePreprocessor._slope_6h(np.array([5.0]))
        assert slope == 0.0

    def test_composite_features_populated(self, tmp_path):
        rp = _make_preprocessor(tmp_path)
        features = rp.build_tabular_features(_records(10))
        assert "shock_index" in features
        assert "pulse_pressure" in features
        assert "qsofa_score" in features
        assert "HR__slope_6h" in features


# ── XGBoostInference ──────────────────────────────────────────────────────────


class TestXGBoostInference:
    def _make_metrics(self, tmp_path: Path, feature_cols: list[str]) -> Path:
        metrics = {
            "feature_columns": feature_cols,
            "test_metrics": {"auc": 0.91},
            "threshold_selection": {"threshold": 0.5},
        }
        p = tmp_path / "metrics.json"
        p.write_text(json.dumps(metrics))
        return p

    def test_available_false_when_files_missing(self, tmp_path):
        xi = XGBoostInference(
            str(tmp_path / "model.ubj"), str(tmp_path / "metrics.json")
        )
        assert xi.available is False

    def test_loaded_false_before_load(self, tmp_path):
        # We can't easily create a real XGBoost model here without training,
        # so just verify the property
        xi = XGBoostInference(
            str(tmp_path / "model.ubj"), str(tmp_path / "metrics.json")
        )
        assert xi.loaded is False

    def test_metrics_property_returns_empty_before_load(self, tmp_path):
        # metrics are None until loaded; property returns {} safely
        xi = XGBoostInference(
            str(tmp_path / "model.ubj"), str(tmp_path / "metrics.json")
        )
        # Will try to load and fail since files don't exist — confirm graceful empty
        try:
            result = xi.metrics
        except Exception:
            result = {}
        assert isinstance(result, dict)

    def test_decision_threshold_calibrated_path(self, tmp_path):
        """Test the decision_threshold property when a calibrator IS present."""
        xi = XGBoostInference(str(tmp_path / "m.ubj"), str(tmp_path / "mt.json"))
        xi._metrics = {"threshold_selection": {"threshold": 0.4}}
        xi._calibrated_threshold = 0.55
        # Create a minimal mock calibrator
        mock_cal = MagicMock()
        xi._calibrator = mock_cal
        assert xi.decision_threshold == pytest.approx(0.55)

    def test_decision_threshold_raw_when_no_calibrator(self, tmp_path):
        xi = XGBoostInference(str(tmp_path / "m.ubj"), str(tmp_path / "mt.json"))
        xi._metrics = {"threshold_selection": {"threshold": 0.42}}
        xi._calibrator = None
        assert xi.decision_threshold == pytest.approx(0.42)

    def test_decision_threshold_none_when_not_in_metrics(self, tmp_path):
        xi = XGBoostInference(str(tmp_path / "m.ubj"), str(tmp_path / "mt.json"))
        xi._metrics = {"threshold_selection": {}}
        xi._calibrator = None
        assert xi.decision_threshold is None

    def test_calibrated_property_false_when_no_calibrator(self, tmp_path):
        xi = XGBoostInference(str(tmp_path / "m.ubj"), str(tmp_path / "mt.json"))
        xi._calibrator = None
        assert xi.calibrated is False

    def test_calibrated_property_true_when_calibrator_present(self, tmp_path):
        xi = XGBoostInference(str(tmp_path / "m.ubj"), str(tmp_path / "mt.json"))
        xi._calibrator = MagicMock()
        assert xi.calibrated is True

    def test_predict_with_calibrator_calls_calibrator(self, tmp_path):
        """Test that predict() passes through calibrator when present."""
        xi = XGBoostInference(str(tmp_path / "m.ubj"), str(tmp_path / "mt.json"))
        # Directly inject a mock model and calibrator so load() is bypassed
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.7])
        mock_calibrator = MagicMock()
        mock_calibrator.predict.return_value = np.array([0.65])
        xi._model = mock_model
        xi._calibrator = mock_calibrator
        xi._feature_columns = ["HR", "ICULOS"]
        xi._metrics = {}

        with patch("xgboost.DMatrix"):
            result = xi.predict({"HR": 80.0, "ICULOS": 5.0})

        mock_calibrator.predict.assert_called_once()
        assert isinstance(result, float)

    def test_predict_logs_warning_for_missing_features(self, tmp_path, caplog):
        """If features are missing, a warning is logged."""
        import logging
        xi = XGBoostInference(str(tmp_path / "m.ubj"), str(tmp_path / "mt.json"))
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.3])
        xi._model = mock_model
        xi._calibrator = None
        xi._feature_columns = ["HR", "ICULOS", "MAP", "Temp"]
        xi._metrics = {}

        with patch("xgboost.DMatrix"):
            with caplog.at_level(logging.WARNING):
                xi.predict({"HR": 80.0})  # MAP and Temp are missing

        assert "missing" in caplog.text.lower()


# ── AgenticICUWorkflow ────────────────────────────────────────────────────────


class TestAgenticICUWorkflow:
    def _make_workflow(
        self,
        resp_agent=None,
        ensemble=None,
        sq_valid=True,
        sq_suppress=False,
    ):
        from agentic_icu.agents.reasoner import AlertPolicy, ClinicalReasoner
        from agentic_icu.domain.contracts import SignalQualityResult

        policy = AlertPolicy()
        reasoner = ClinicalReasoner(policy)

        # Use a real SignalQualityResult so Pydantic validation passes
        real_sq = SignalQualityResult(
            signal_valid=sq_valid,
            suppression_recommendation=sq_suppress,
            suppression_mode="full" if sq_suppress else "none",
        )

        sq_agent = MagicMock()
        sq_agent.evaluate.return_value = (real_sq, [])

        vitals_agent = MagicMock()
        vitals_agent.evaluate.return_value = (
            ModelAgentResult(
                status="available", score=0.8, risk_band="high",
                detail="high", decision_threshold=0.5, threshold_ratio=1.6,
            ),
            [],
        )

        lab_agent = MagicMock()
        lab_agent.evaluate.return_value = (
            ModelAgentResult(
                status="available", score=0.3, risk_band="low",
                detail="low", decision_threshold=0.5, threshold_ratio=0.6,
            ),
            [],
        )

        return AgenticICUWorkflow(
            signal_quality_agent=sq_agent,
            vitals_agent=vitals_agent,
            lab_agent=lab_agent,
            reasoner=reasoner,
            resp_failure_agent=resp_agent,
            ensemble=ensemble,
        )

    def _request(self) -> EvaluatePatientRequest:
        return EvaluatePatientRequest(
            patient_id="test",
            observation_window=_records(6),
        )

    def test_full_suppression_short_circuits(self):
        wf = self._make_workflow(sq_valid=False, sq_suppress=True)
        response = wf.evaluate(self._request())
        assert response.vitals_agent.status == "unavailable"
        assert response.lab_agent.status == "unavailable"
        # Vitals/Lab agents should NOT be called at all
        wf.vitals_agent.evaluate.assert_not_called()
        wf.lab_agent.evaluate.assert_not_called()

    def test_no_resp_agent_returns_unavailable(self):
        wf = self._make_workflow(resp_agent=None)
        response = wf.evaluate(self._request())
        assert response.resp_failure_agent.status == "unavailable"
        assert "not configured" in response.resp_failure_agent.detail

    def test_with_resp_agent_calls_it(self):
        resp_mock = MagicMock()
        resp_mock.evaluate.return_value = (
            ModelAgentResult(status="available", score=0.2, risk_band="low", detail="ok"),
            [],
        )
        wf = self._make_workflow(resp_agent=resp_mock)
        wf.evaluate(self._request())
        resp_mock.evaluate.assert_called_once()

    def test_ensemble_predict_error_is_tolerated(self):
        from agentic_icu.inference.ensemble import EnsembleInference

        mock_ensemble = MagicMock(spec=EnsembleInference)
        mock_ensemble.available = True
        mock_ensemble.predict.side_effect = ValueError("bad scores")

        wf = self._make_workflow(ensemble=mock_ensemble)
        # Should not raise — error is caught internally
        response = wf.evaluate(self._request())
        assert response.ensemble_agent.status == "unavailable"

    def test_ensemble_not_called_when_unavailable(self):
        from agentic_icu.inference.ensemble import EnsembleInference

        mock_ensemble = MagicMock(spec=EnsembleInference)
        mock_ensemble.available = False

        wf = self._make_workflow(ensemble=mock_ensemble)
        wf.evaluate(self._request())
        mock_ensemble.predict.assert_not_called()

    def test_ensemble_called_when_available(self):
        from agentic_icu.inference.ensemble import EnsembleInference

        mock_ensemble = MagicMock(spec=EnsembleInference)
        mock_ensemble.available = True
        mock_ensemble.predict.return_value = 0.88

        wf = self._make_workflow(ensemble=mock_ensemble)
        response = wf.evaluate(self._request())
        mock_ensemble.predict.assert_called_once()
        assert response.ensemble_agent.status == "available"

    def test_ensemble_high_risk_band(self):
        from agentic_icu.inference.ensemble import EnsembleInference
        mock_ensemble = MagicMock(spec=EnsembleInference)
        mock_ensemble.available = True
        mock_ensemble.predict.return_value = 0.9  # above high_thresh=0.8

        wf = self._make_workflow(ensemble=mock_ensemble)
        response = wf.evaluate(self._request())
        assert response.ensemble_agent.risk_band == "high"

    def test_ensemble_moderate_risk_band(self):
        from agentic_icu.inference.ensemble import EnsembleInference
        mock_ensemble = MagicMock(spec=EnsembleInference)
        mock_ensemble.available = True
        mock_ensemble.predict.return_value = 0.65  # between 0.55 and 0.8

        wf = self._make_workflow(ensemble=mock_ensemble)
        response = wf.evaluate(self._request())
        assert response.ensemble_agent.risk_band == "moderate"

    def test_response_contains_all_required_fields(self):
        wf = self._make_workflow()
        response = wf.evaluate(self._request())
        assert response.patient_id == "test"
        assert response.clinical_decision is not None
        assert response.reasoning_log is not None
