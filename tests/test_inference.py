"""Unit tests for inference layer: EnsembleInference."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_icu.inference.ensemble import EnsembleInference


def _make_real_lr_model(tmp_path: Path) -> Path:
    """Create a real pickled LogisticRegression model for testing."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    # Fit on trivial data so predict_proba works
    X = np.array([[0.1, 0.1], [0.9, 0.9], [0.3, 0.5], [0.7, 0.2]])
    y = np.array([0, 1, 0, 1])
    lr.fit(X, y)
    model_file = tmp_path / "ensemble.pkl"
    with model_file.open("wb") as fh:
        pickle.dump(lr, fh)
    return model_file


# ── EnsembleInference ─────────────────────────────────────────────────────────

class TestEnsembleInference:

    def test_available_false_when_file_missing(self, tmp_path):
        ei = EnsembleInference(str(tmp_path / "model.pkl"))
        assert ei.available is False

    def test_available_true_when_file_exists(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        assert ei.available is True

    def test_loaded_false_before_load(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        assert ei.loaded is False

    def test_load_raises_file_not_found(self, tmp_path):
        ei = EnsembleInference(str(tmp_path / "missing.pkl"))
        with pytest.raises(FileNotFoundError, match="Ensemble model not found"):
            ei.load()

    def test_load_succeeds_with_real_model(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        ei.load()
        assert ei.loaded is True

    def test_load_is_idempotent(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        ei.load()
        ei.load()  # second call should be a no-op
        assert ei.loaded is True

    def test_predict_raises_on_out_of_range_score(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        ei.load()
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            ei.predict(1.5, 0.5)

    def test_predict_raises_on_nan(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        ei.load()
        with pytest.raises(ValueError, match="finite number"):
            ei.predict(float("nan"), 0.5)

    def test_predict_raises_on_inf(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        ei.load()
        with pytest.raises(ValueError, match="finite number"):
            ei.predict(float("inf"), 0.5)

    def test_predict_returns_float_in_unit_range(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        ei.load()
        result = ei.predict(0.8, 0.6)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_triggers_lazy_load(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        assert ei.loaded is False
        result = ei.predict(0.5, 0.5)  # Should auto-load
        assert ei.loaded is True
        assert isinstance(result, float)

    def test_metrics_loads_json_file(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"val_auc": 0.95, "val_auprc": 0.91}))
        ei = EnsembleInference(str(model_file), str(metrics_file))
        ei.load()
        assert ei.metrics["val_auc"] == pytest.approx(0.95)
        assert ei.metrics["val_auprc"] == pytest.approx(0.91)

    def test_metrics_returns_empty_when_no_metrics_file(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        ei.load()
        assert ei.metrics == {}

    def test_metrics_triggers_lazy_load(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file))
        _ = ei.metrics  # Should trigger lazy load
        assert ei.loaded is True

    def test_metrics_returns_empty_when_metrics_file_missing(self, tmp_path):
        model_file = _make_real_lr_model(tmp_path)
        ei = EnsembleInference(str(model_file), str(tmp_path / "nonexistent.json"))
        ei.load()
        assert ei.metrics == {}
