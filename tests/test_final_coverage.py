"""Final targeted tests to push remaining gap modules to 93%+ coverage.

Covers:
- SequenceInference: properties, predict/saliency with injected model
- TabularExplainer: _ensure_loaded idempotency, format_explanation edge cases
- ClinicalReasoner high/medium alert triggers for max_score / mean_score paths
- api/main.py: latest_alert_policy_report_path (glob branch), health resp_failure
               path coverage, patients endpoint, WebSocket error path
- config.py: missing env var path
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_icu.agents.reasoner import AlertPolicy, ClinicalReasoner
from agentic_icu.domain.contracts import ModelAgentResult, SignalQualityResult

# ── SequenceInference properties without loading real model ───────────────────

class TestSequenceInferenceProperties:
    """Test SequenceInference without requiring real .pt weights."""

    def test_available_false_when_files_missing(self, tmp_path):
        from agentic_icu.inference.sequence import SequenceInference
        si = SequenceInference(str(tmp_path / "m.pt"), str(tmp_path / "metrics.json"))
        assert si.available is False

    def test_loaded_false_before_load(self, tmp_path):
        from agentic_icu.inference.sequence import SequenceInference
        si = SequenceInference(str(tmp_path / "m.pt"), str(tmp_path / "metrics.json"))
        assert si.loaded is False

    def test_calibrated_false_when_no_calibrator(self, tmp_path):
        from agentic_icu.inference.sequence import SequenceInference
        si = SequenceInference(str(tmp_path / "m.pt"), str(tmp_path / "metrics.json"))
        si._calibrator = None
        assert si.calibrated is False

    def test_calibrated_true_when_calibrator_present(self, tmp_path):
        from agentic_icu.inference.sequence import SequenceInference
        si = SequenceInference(str(tmp_path / "m.pt"), str(tmp_path / "metrics.json"))
        si._calibrator = MagicMock()
        assert si.calibrated is True

    def test_decision_threshold_raw_no_calibrator(self, tmp_path):
        from agentic_icu.inference.sequence import SequenceInference
        si = SequenceInference(str(tmp_path / "m.pt"), str(tmp_path / "metrics.json"))
        si._metrics = {"threshold_selection": {"threshold": 0.6}}
        si._calibrator = None
        assert si.decision_threshold == pytest.approx(0.6)

    def test_decision_threshold_calibrated(self, tmp_path):
        from agentic_icu.inference.sequence import SequenceInference
        si = SequenceInference(str(tmp_path / "m.pt"), str(tmp_path / "metrics.json"))
        si._metrics = {"threshold_selection": {"threshold": 0.6}}
        si._calibrator = MagicMock()
        si._calibrated_threshold = 0.72
        assert si.decision_threshold == pytest.approx(0.72)

    def test_decision_threshold_none_when_missing(self, tmp_path):
        from agentic_icu.inference.sequence import SequenceInference
        si = SequenceInference(str(tmp_path / "m.pt"), str(tmp_path / "metrics.json"))
        si._metrics = {"threshold_selection": {}}
        assert si.decision_threshold is None

    def test_predict_with_injected_model(self, tmp_path):
        """Test predict() by injecting a mock GRU model — forced to CPU."""
        import torch

        from agentic_icu.inference.sequence import SequenceGRU, SequenceInference
        si = SequenceInference(str(tmp_path / "m.pt"), str(tmp_path / "metrics.json"))
        si._metrics = {}

        n_features = 4
        mock_gru = SequenceGRU(input_size=n_features * 2, hidden_size=8, num_layers=1)
        mock_gru.eval()
        si._model = mock_gru
        si.device = torch.device("cpu")  # force CPU to match weights

        tensor = np.random.rand(8, n_features * 2).astype(np.float32)
        result = si.predict(tensor)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_with_calibrator(self, tmp_path):
        """Test predict() when calibrator is applied."""
        import torch

        from agentic_icu.inference.sequence import SequenceGRU, SequenceInference

        si = SequenceInference(str(tmp_path / "m.pt"), str(tmp_path / "metrics.json"))
        si._metrics = {}
        n_features = 4
        si._model = SequenceGRU(input_size=n_features * 2, hidden_size=8, num_layers=1)
        si._model.eval()
        si.device = torch.device("cpu")  # force CPU

        mock_cal = MagicMock()
        mock_cal.predict.return_value = np.array([0.77])
        si._calibrator = mock_cal

        tensor = np.random.rand(8, n_features * 2).astype(np.float32)
        result = si.predict(tensor)
        mock_cal.predict.assert_called_once()
        assert result == pytest.approx(0.77)

    def test_temporal_saliency_returns_list(self, tmp_path):
        """Test saliency returns a float list of length T."""
        import torch

        from agentic_icu.inference.sequence import SequenceGRU, SequenceInference
        si = SequenceInference(str(tmp_path / "m.pt"), str(tmp_path / "metrics.json"))
        si._metrics = {}
        n_features = 4
        si._model = SequenceGRU(input_size=n_features * 2, hidden_size=8, num_layers=1)
        si._model.eval()
        si.device = torch.device("cpu")  # force CPU

        tensor = np.random.rand(8, n_features * 2).astype(np.float32)
        weights = si.temporal_saliency(tensor)
        assert len(weights) == 8
        assert all(isinstance(w, float) for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-4  # sums to 1


# ── TabularExplainer edge cases ───────────────────────────────────────────────

class TestTabularExplainerEdges:
    def test_format_explanation_empty_list(self):
        from agentic_icu.inference.explainer import TabularExplainer
        mock_inference = MagicMock()
        te = TabularExplainer(mock_inference)
        assert te.format_explanation([]) == ""

    def test_format_explanation_decreases_risk_arrow(self):
        from agentic_icu.inference.explainer import TabularExplainer
        mock_inference = MagicMock()
        te = TabularExplainer(mock_inference)
        result = te.format_explanation([
            {"label": "Lactate", "shap_value": -0.1, "direction": "decreases_risk"},
        ])
        assert "↓" in result
        assert "Lactate" in result

    def test_ensure_loaded_idempotent(self):
        """_ensure_loaded should not re-init the explainer on repeat calls."""
        from agentic_icu.inference.explainer import TabularExplainer
        mock_inference = MagicMock()
        te = TabularExplainer(mock_inference)
        sentinel = MagicMock()
        te._explainer = sentinel  # pre-inject
        te._ensure_loaded()  # should be a no-op
        assert te._explainer is sentinel  # unchanged


# ── ClinicalReasoner — uncovered alert paths ──────────────────────────────────

class TestReasonerUncoveredPaths:
    def _sq_clear(self):
        return SignalQualityResult(
            signal_valid=True, suppression_recommendation=False, suppression_mode="none"
        )

    def _agent(self, score: float, threshold: float = 0.5) -> ModelAgentResult:
        return ModelAgentResult(
            status="available",
            score=score,
            risk_band="high" if score >= threshold else "low",
            detail="test",
            decision_threshold=threshold,
            threshold_ratio=round(score / threshold, 3),
        )

    def test_high_alert_via_max_score_threshold(self):
        """Trigger high alert through the max_score threshold path."""
        # Disable trigger paths, enable max_score only
        policy = AlertPolicy(
            high_alert_max_score_threshold=0.7,
            high_alert_extreme_sequence_score_threshold=None,
            high_alert_supported_sequence_score_threshold=None,
            high_alert_ensemble_score_threshold=None,
            high_alert_mean_score_threshold=None,
        )
        reasoner = ClinicalReasoner(policy)
        decision, *_ = reasoner.decide(
            self._sq_clear(), self._agent(0.9), self._agent(0.9)
        )
        assert decision.alert_triggered

    def test_high_alert_via_mean_score_threshold(self):
        """Trigger high alert through the mean_score threshold path."""
        policy = AlertPolicy(
            high_alert_max_score_threshold=None,
            high_alert_mean_score_threshold=0.3,
            high_alert_extreme_sequence_score_threshold=None,
            high_alert_supported_sequence_score_threshold=None,
            high_alert_ensemble_score_threshold=None,
        )
        reasoner = ClinicalReasoner(policy)
        decision, *_ = reasoner.decide(
            self._sq_clear(), self._agent(0.8), self._agent(0.8)
        )
        assert decision.alert_triggered

    def test_medium_alert_via_max_score_threshold(self):
        """Trigger medium alert via medium_alert_max_score_threshold."""
        policy = AlertPolicy(
            # Disable all high paths
            high_alert_extreme_sequence_score_threshold=None,
            high_alert_supported_sequence_score_threshold=None,
            high_alert_ensemble_score_threshold=None,
            high_alert_max_score_threshold=None,
            high_alert_mean_score_threshold=None,
            # Enable medium via max_score
            medium_alert_max_score_threshold=0.3,
            medium_alert_sequence_score_threshold=None,
            medium_alert_tabular_score_threshold=None,
            medium_alert_ensemble_score_threshold=None,
        )
        reasoner = ClinicalReasoner(policy)
        decision, *_ = reasoner.decide(
            self._sq_clear(), self._agent(0.4), self._agent(0.4)
        )
        assert decision.alert_triggered
        assert decision.priority == policy.medium_alert_priority


# ── api/main.py: report path glob branch ─────────────────────────────────────

class TestLatestReportPathGlob:
    def test_returns_most_recent_timestamped_report(self, tmp_path):
        """When timestamped reports exist, the most recent one is returned."""
        from agentic_icu.api.main import latest_alert_policy_report_path

        r1 = tmp_path / "alert_policy_comparison_20240101.json"
        r2 = tmp_path / "alert_policy_comparison_20240201.json"
        r1.write_text("{}")
        r2.write_text("{}")
        # Make r2 newer
        import time as _time
        _time.sleep(0.02)
        r2.touch()

        with patch("agentic_icu.api.main.settings.reports_dir", str(tmp_path)):
            result = latest_alert_policy_report_path()
        assert result == r2

    def test_fallback_to_latest_json_when_no_timestamped(self, tmp_path):
        from agentic_icu.api.main import latest_alert_policy_report_path
        fallback = tmp_path / "alert-policy-latest.json"
        fallback.write_text("{}")

        with patch("agentic_icu.api.main.settings.reports_dir", str(tmp_path)):
            result = latest_alert_policy_report_path()
        assert result == fallback

    def test_raises_when_no_report_exists(self, tmp_path):
        from agentic_icu.api.main import latest_alert_policy_report_path
        with patch("agentic_icu.api.main.settings.reports_dir", str(tmp_path)):
            with pytest.raises(FileNotFoundError):
                latest_alert_policy_report_path()


# ── api/main.py: health endpoint — resp_failure with load failure ─────────────

class TestHealthEndpointPaths:
    def test_health_with_resp_agent_load_exception(self):
        """Test health endpoint gracefully handles resp_failure load error."""
        from fastapi.testclient import TestClient

        from agentic_icu.api.main import app

        mock_wf = MagicMock()
        mock_wf.vitals_agent.preprocessor.available = True
        mock_wf.vitals_agent.predictor.available = True
        mock_wf.lab_agent.predictor.available = True
        mock_wf.resp_failure_agent = MagicMock()
        mock_wf.resp_failure_agent.gru_predictor.available = False
        mock_wf.vitals_agent.predictor.load.side_effect = RuntimeError("load failed")
        mock_wf.lab_agent.predictor.load.side_effect = RuntimeError("load failed")
        mock_wf.resp_failure_agent.gru_predictor.load.side_effect = RuntimeError("load failed")

        with patch("agentic_icu.api.main.get_workflow", return_value=mock_wf):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health")
        # Should still return 200 — load failure is tolerated
        assert response.status_code == 200


# ── api/main.py: WebSocket broadcast from /evaluate ──────────────────────────

class TestWebSocketBroadcast:
    def test_evaluate_broadcasts_on_high_priority_alert(self):
        """When /evaluate returns a high priority decision, broadcast is called."""
        from fastapi.testclient import TestClient

        from agentic_icu.api.main import app, manager
        from agentic_icu.domain.contracts import (
            ClinicalDecision,
            EvaluatePatientResponse,
            SignalQualityResult,
        )

        # Build a fake response with a high-priority alert
        mock_decision = ClinicalDecision(
            alert_triggered=True,
            alert_type="Sepsis Early Warning",
            priority="high",
            rationale="Test high alert",
        )
        mock_sq = SignalQualityResult(
            signal_valid=True,
            suppression_recommendation=False,
            suppression_mode="none",
        )
        unavail = ModelAgentResult(status="unavailable", detail="x")
        fake_response = EvaluatePatientResponse(
            patient_id="p000001",
            signal_quality=mock_sq,
            vitals_agent=unavail,
            lab_agent=unavail,
            resp_failure_agent=unavail,
            clinical_decision=mock_decision,
            reasoning_log=[],
        )

        mock_wf = MagicMock()
        mock_wf.evaluate.return_value = fake_response

        # Spy on broadcast
        broadcast_calls = []
        original_broadcast = manager.broadcast

        async def spy_broadcast(msg: str):
            broadcast_calls.append(msg)

        manager.broadcast = spy_broadcast
        try:
            with patch("agentic_icu.api.main.get_workflow", return_value=mock_wf):
                client = TestClient(app, raise_server_exceptions=False)
                body = {
                    "patient_id": "p000001",
                    "observation_window": [{"values": {"HR": 80.0, "ICULOS": 1.0}}],
                }
                response = client.post("/evaluate", json=body)
        finally:
            manager.broadcast = original_broadcast

        assert response.status_code == 200
        assert len(broadcast_calls) == 1
        import json as _json
        payload = _json.loads(broadcast_calls[0])
        assert payload["type"] == "CLINICAL_ALERT"
        assert payload["priority"] == "high"


# ── SequenceGRU forward pass ──────────────────────────────────────────────────

class TestSequenceGRU:
    def test_forward_unidirectional(self):
        import torch

        from agentic_icu.inference.sequence import SequenceGRU
        model = SequenceGRU(input_size=10, hidden_size=16, num_layers=1)
        model.eval()
        x = torch.randn(2, 6, 10)
        out = model(x)
        assert out.shape == (2,)

    def test_forward_bidirectional(self):
        import torch

        from agentic_icu.inference.sequence import SequenceGRU
        model = SequenceGRU(input_size=10, hidden_size=16, num_layers=2, bidirectional=True)
        model.eval()
        x = torch.randn(1, 8, 10)
        out = model(x)
        assert out.shape == (1,)
