from __future__ import annotations

import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Dict, Optional

import xgboost as xgb

logger = logging.getLogger(__name__)


class XGBoostInference:
    def __init__(self, model_path: str, metrics_path: str, calibrator_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.calibrator_path = Path(calibrator_path) if calibrator_path else None
        self._model: Optional[xgb.Booster] = None
        self._feature_columns: Optional[list[str]] = None
        self._metrics: Optional[dict] = None
        self._calibrator = None
        self._calibrated_threshold: Optional[float] = None
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.model_path.exists() and self.metrics_path.exists()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:  # re-check after acquiring
                return
            model = xgb.Booster()
            model.load_model(str(self.model_path))
            with self.metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            self._model = model
            self._metrics = metrics
            self._feature_columns = metrics["feature_columns"]
            if self.calibrator_path and self.calibrator_path.exists():
                with self.calibrator_path.open("rb") as fh:
                    self._calibrator = pickle.load(fh)
                # Re-anchor the threshold to calibrated probability space.
                raw_threshold = metrics.get("threshold_selection", {}).get("threshold")
                if raw_threshold is not None:
                    try:
                        self._calibrated_threshold = float(self._calibrator.predict([raw_threshold])[0])
                    except Exception:
                        self._calibrated_threshold = None

    @property
    def feature_columns(self) -> list[str]:
        if self._feature_columns is None:
            self.load()
        return self._feature_columns or []

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            self.load()
        return self._metrics or {}

    @property
    def decision_threshold(self) -> float | None:
        threshold_payload = self.metrics.get("threshold_selection", {})
        raw_threshold = threshold_payload.get("threshold")
        if raw_threshold is None:
            return None
        if self._calibrator is not None and self._calibrated_threshold is not None:
            return self._calibrated_threshold
        return float(raw_threshold)

    @property
    def calibrated(self) -> bool:
        return self._calibrator is not None

    @property
    def model(self) -> xgb.Booster:
        if self._model is None:
            self.load()
        return self._model

    def predict(self, features: Dict[str, float]) -> float:
        if self._model is None:
            self.load()
        missing = [col for col in self.feature_columns if col not in features]
        if missing:
            logger.warning(
                "XGBoostInference.predict: %d feature(s) missing from input — defaulting to 0.0. "
                "First 5: %s. Caller should use RuntimePreprocessor.build_tabular_features().",
                len(missing), missing[:5],
            )
        aligned = [[features.get(col, 0.0) for col in self.feature_columns]]
        matrix = xgb.DMatrix(aligned, feature_names=self.feature_columns)
        raw_score = float(self._model.predict(matrix)[0])
        if self._calibrator is not None:
            return float(self._calibrator.predict([raw_score])[0])
        return raw_score
