"""Unit tests for VitalsAgent, LabTabularAgent and RespFailureAgent.

These tests use lightweight mocks so no real ML models are needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_icu.agents.resp_failure import RespFailureAgent
from agentic_icu.agents.tabular import LabTabularAgent
from agentic_icu.agents.vitals import VitalsAgent
from agentic_icu.domain.contracts import ObservationRecord

# ── Shared helpers ────────────────────────────────────────────────────────────

def _records(n: int = 6) -> list[ObservationRecord]:
    return [ObservationRecord(values={"HR": float(70 + i), "ICULOS": float(i + 1)}) for i in range(n)]


def _make_preprocessor(available: bool = True) -> MagicMock:
    m = MagicMock()
    m.available = available
    m.build_sequence_tensor.return_value = MagicMock()
    m.build_tabular_features.return_value = MagicMock()
    return m


def _make_seq_predictor(available: bool = True, score: float = 0.75, threshold: float = 0.5) -> MagicMock:
    m = MagicMock()
    m.available = available
    m.predict.return_value = score
    m.decision_threshold = threshold
    m.temporal_saliency.return_value = [0.1, 0.9, 0.5, 0.3, 0.2, 0.8]
    return m


def _make_xgb_predictor(available: bool = True, score: float = 0.3, threshold: float = 0.5) -> MagicMock:
    m = MagicMock()
    m.available = available
    m.predict.return_value = score
    m.decision_threshold = threshold
    return m


# ── VitalsAgent ───────────────────────────────────────────────────────────────

class TestVitalsAgent:
    def test_unavailable_when_preprocessor_not_ready(self):
        agent = VitalsAgent(_make_preprocessor(available=False), _make_seq_predictor())
        result, logs = agent.evaluate(_records())
        assert result.status == "unavailable"
        assert len(logs) == 1
        assert "Vitals Agent" in logs[0].agent

    def test_unavailable_when_predictor_not_ready(self):
        agent = VitalsAgent(_make_preprocessor(), _make_seq_predictor(available=False))
        result, logs = agent.evaluate(_records())
        assert result.status == "unavailable"

    def test_high_risk_band_above_threshold(self):
        # score=0.9, threshold=0.5 → ratio=1.8 → high
        agent = VitalsAgent(_make_preprocessor(), _make_seq_predictor(score=0.9, threshold=0.5))
        result, _ = agent.evaluate(_records())
        assert result.status == "available"
        assert result.risk_band == "high"
        assert result.score == pytest.approx(0.9)

    def test_moderate_risk_band(self):
        # score=0.38, threshold=0.5 → ratio=0.76 → moderate (>= 0.7)
        agent = VitalsAgent(_make_preprocessor(), _make_seq_predictor(score=0.38, threshold=0.5))
        result, _ = agent.evaluate(_records())
        assert result.risk_band == "moderate"

    def test_low_risk_band_below_threshold(self):
        # score=0.1, threshold=0.5 → ratio=0.2 → low
        agent = VitalsAgent(_make_preprocessor(), _make_seq_predictor(score=0.1, threshold=0.5))
        result, _ = agent.evaluate(_records())
        assert result.risk_band == "low"

    def test_saliency_populates_feature_contributions(self):
        agent = VitalsAgent(_make_preprocessor(), _make_seq_predictor())
        result, _ = agent.evaluate(_records())
        assert len(result.feature_contributions) > 0
        assert result.explanation != ""

    def test_saliency_failure_is_tolerated(self):
        predictor = _make_seq_predictor()
        predictor.temporal_saliency.side_effect = RuntimeError("CUDA OOM")
        agent = VitalsAgent(_make_preprocessor(), predictor)
        result, _ = agent.evaluate(_records())
        # Should still return a valid result, just without contributions
        assert result.status == "available"
        assert result.feature_contributions == {}

    def test_detail_string_contains_threshold_info(self):
        agent = VitalsAgent(_make_preprocessor(), _make_seq_predictor(score=0.9, threshold=0.5))
        result, _ = agent.evaluate(_records())
        assert "0.500" in result.detail or "0.5" in result.detail

    def test_no_decision_threshold_produces_simple_detail(self):
        predictor = _make_seq_predictor(score=0.6)
        predictor.decision_threshold = None
        agent = VitalsAgent(_make_preprocessor(), predictor)
        result, _ = agent.evaluate(_records())
        assert result.status == "available"
        assert result.threshold_ratio is None


# ── LabTabularAgent ───────────────────────────────────────────────────────────

class TestLabTabularAgent:
    def test_unavailable_when_preprocessor_not_ready(self):
        agent = LabTabularAgent(_make_preprocessor(available=False), _make_xgb_predictor())
        result, logs = agent.evaluate(_records())
        assert result.status == "unavailable"
        assert "Lab Agent" in logs[0].agent

    def test_unavailable_when_predictor_not_ready(self):
        agent = LabTabularAgent(_make_preprocessor(), _make_xgb_predictor(available=False))
        result, logs = agent.evaluate(_records())
        assert result.status == "unavailable"

    def test_high_risk_band(self):
        agent = LabTabularAgent(_make_preprocessor(), _make_xgb_predictor(score=0.9, threshold=0.5))
        result, _ = agent.evaluate(_records())
        assert result.risk_band == "high"

    def test_moderate_risk_band(self):
        agent = LabTabularAgent(_make_preprocessor(), _make_xgb_predictor(score=0.38, threshold=0.5))
        result, _ = agent.evaluate(_records())
        assert result.risk_band == "moderate"

    def test_low_risk_band(self):
        agent = LabTabularAgent(_make_preprocessor(), _make_xgb_predictor(score=0.1, threshold=0.5))
        result, _ = agent.evaluate(_records())
        assert result.risk_band == "low"

    def test_no_explainer_returns_empty_contributions(self):
        agent = LabTabularAgent(_make_preprocessor(), _make_xgb_predictor(), explainer=None)
        result, _ = agent.evaluate(_records())
        assert result.feature_contributions == {}
        assert result.explanation == ""

    def test_with_explainer_populates_contributions(self):
        explainer = MagicMock()
        explainer.top_contributions.return_value = (
            [{"feature": "HR", "shap_value": 0.15, "label": "HR", "direction": "increases_risk"}],
            {"HR": 0.15},
        )
        explainer.format_explanation.return_value = "High HR increases sepsis risk."
        agent = LabTabularAgent(_make_preprocessor(), _make_xgb_predictor(), explainer=explainer)
        result, _ = agent.evaluate(_records())
        assert "HR" in result.feature_contributions
        assert "HR" in result.detail or "increases" in result.explanation

    def test_shap_failure_is_tolerated(self):
        explainer = MagicMock()
        explainer.top_contributions.side_effect = RuntimeError("SHAP failed")
        agent = LabTabularAgent(_make_preprocessor(), _make_xgb_predictor(), explainer=explainer)
        result, _ = agent.evaluate(_records())
        assert result.status == "available"
        assert result.feature_contributions == {}

    def test_no_threshold_produces_simple_detail(self):
        predictor = _make_xgb_predictor(score=0.4)
        predictor.decision_threshold = None
        agent = LabTabularAgent(_make_preprocessor(), predictor)
        result, _ = agent.evaluate(_records())
        assert result.threshold_ratio is None


# ── RespFailureAgent ──────────────────────────────────────────────────────────

class TestRespFailureAgent:
    def test_unavailable_when_preprocessor_not_ready(self):
        agent = RespFailureAgent(_make_preprocessor(available=False), _make_seq_predictor())
        result, logs = agent.evaluate(_records())
        assert result.status == "unavailable"
        assert "Resp Failure Agent" in logs[0].agent

    def test_unavailable_when_predictor_not_ready(self):
        agent = RespFailureAgent(_make_preprocessor(), _make_seq_predictor(available=False))
        result, logs = agent.evaluate(_records())
        assert result.status == "unavailable"

    def test_high_risk_band(self):
        agent = RespFailureAgent(_make_preprocessor(), _make_seq_predictor(score=0.9, threshold=0.5))
        result, _ = agent.evaluate(_records())
        assert result.risk_band == "high"

    def test_moderate_risk_band(self):
        # ratio=0.76 → moderate
        agent = RespFailureAgent(_make_preprocessor(), _make_seq_predictor(score=0.38, threshold=0.5))
        result, _ = agent.evaluate(_records())
        assert result.risk_band == "moderate"

    def test_low_risk_band(self):
        agent = RespFailureAgent(_make_preprocessor(), _make_seq_predictor(score=0.1, threshold=0.5))
        result, _ = agent.evaluate(_records())
        assert result.risk_band == "low"

    def test_saliency_populates_contributions(self):
        agent = RespFailureAgent(_make_preprocessor(), _make_seq_predictor())
        result, _ = agent.evaluate(_records())
        assert len(result.feature_contributions) > 0
        assert "Resp model focused on" in result.explanation

    def test_saliency_failure_is_tolerated(self):
        predictor = _make_seq_predictor()
        predictor.temporal_saliency.side_effect = RuntimeError("GPU error")
        agent = RespFailureAgent(_make_preprocessor(), predictor)
        result, _ = agent.evaluate(_records())
        assert result.status == "available"
        assert result.feature_contributions == {}

    def test_no_threshold_produces_simple_detail(self):
        predictor = _make_seq_predictor(score=0.5)
        predictor.decision_threshold = None
        agent = RespFailureAgent(_make_preprocessor(), predictor)
        result, _ = agent.evaluate(_records())
        assert result.threshold_ratio is None

    def test_xgb_predictor_optional(self):
        # Should work fine with no xgb_predictor passed
        agent = RespFailureAgent(_make_preprocessor(), _make_seq_predictor(), xgb_predictor=None)
        result, _ = agent.evaluate(_records())
        assert result.status == "available"

    def test_top_step_consecutive_hours_description(self):
        predictor = _make_seq_predictor()
        # Weights that make hours 1,2,3 the top 3 (consecutive)
        predictor.temporal_saliency.return_value = [0.9, 0.8, 0.7, 0.1, 0.05, 0.02]
        agent = RespFailureAgent(_make_preprocessor(), predictor)
        result, _ = agent.evaluate(_records())
        assert "1-3" in result.explanation or "hours" in result.explanation

    def test_single_top_step_description(self):
        predictor = _make_seq_predictor()
        # Only one dominant step
        predictor.temporal_saliency.return_value = [0.99, 0.0, 0.0, 0.0, 0.0, 0.0]
        agent = RespFailureAgent(_make_preprocessor(), predictor)
        result, _ = agent.evaluate(_records())
        assert "hour" in result.explanation
