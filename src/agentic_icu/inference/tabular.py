from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import xgboost as xgb


class XGBoostInference:
    def __init__(self, model_path: str, metrics_path: str, calibrator_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.calibrator_path = Path(calibrator_path) if calibrator_path else None
        self._model: Optional[xgb.Booster] = None
        self._feature_columns: Optional[list[str]] = None
        self._metrics: Optional[dict] = None
        self._calibrator = None

    @property
    def available(self) -> bool:
        return self.model_path.exists() and self.metrics_path.exists()

    def load(self) -> None:
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
        threshold = threshold_payload.get("threshold")
        return float(threshold) if threshold is not None else None

    @property
    def calibrated(self) -> bool:
        return self._calibrator is not None

    def predict(self, features: Dict[str, float]) -> float:
        if self._model is None:
            self.load()
        aligned = [[features.get(column, 0.0) for column in self.feature_columns]]
        matrix = xgb.DMatrix(aligned, feature_names=self.feature_columns)
        raw_score = float(self._model.predict(matrix)[0])
        if self._calibrator is not None:
            return float(self._calibrator.predict([raw_score])[0])
        return raw_score
