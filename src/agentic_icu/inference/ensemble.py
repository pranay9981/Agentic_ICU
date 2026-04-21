from __future__ import annotations

import json
import math
import pickle
import threading
import warnings
from pathlib import Path
from typing import Optional


class EnsembleInference:
    """Logistic-regression meta-learner that fuses calibrated GRU + XGBoost scores."""

    def __init__(self, model_path: str, metrics_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self._model = None
        self._metrics: Optional[dict] = None
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    @property
    def loaded(self) -> bool:
        """True if the model has been loaded into memory (not just present on disk)."""
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self.model_path.exists():
                raise FileNotFoundError(f"Ensemble model not found: {self.model_path}")
            with self.model_path.open("rb") as fh:
                # Suppress only sklearn version mismatch warnings — expected when
                # artifacts were trained on a different sklearn minor version.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=UserWarning, module="sklearn"
                    )
                    model = pickle.load(fh)  # nosec B301
            metrics: dict = {}
            if self.metrics_path and self.metrics_path.exists():
                with self.metrics_path.open("r", encoding="utf-8") as fh:
                    metrics = json.load(fh)
            self._model = model
            self._metrics = metrics

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            self.load()
        return self._metrics or {}

    def predict(self, gru_score: float, xgb_score: float) -> float:
        for name, score in (("gru_score", gru_score), ("xgb_score", xgb_score)):
            if (
                not isinstance(score, (int, float))
                or math.isnan(score)
                or math.isinf(score)
            ):
                raise ValueError(f"{name} must be a finite number, got {score!r}")
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {score}")
        if self._model is None:
            self.load()
        if self._model is None:
            raise RuntimeError("EnsembleInference.predict called before model was loaded.")
        if not hasattr(self._model, 'predict_proba'):
            raise TypeError(f"Ensemble model has no predict_proba() method: {type(self._model)}")
        return float(self._model.predict_proba([[gru_score, xgb_score]])[0, 1])
