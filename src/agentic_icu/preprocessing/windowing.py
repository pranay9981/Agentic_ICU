from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from agentic_icu.domain.contracts import ObservationRecord
from agentic_icu.domain.features import DYNAMIC_FEATURES, STATIC_FEATURES


class RuntimePreprocessor:
    def __init__(
        self,
        train_statistics_path: str,
        pipeline_config_path: str,
    ) -> None:
        self.train_statistics_path = Path(train_statistics_path)
        self.pipeline_config_path = Path(pipeline_config_path)
        self._stats: Optional[Dict[str, Dict[str, float]]] = None
        self._pipeline_config: Optional[Dict[str, Any]] = None

    @property
    def available(self) -> bool:
        return self.train_statistics_path.exists() and self.pipeline_config_path.exists()

    def load(self) -> None:
        with self.train_statistics_path.open("r", encoding="utf-8") as handle:
            self._stats = json.load(handle)
        with self.pipeline_config_path.open("r", encoding="utf-8") as handle:
            self._pipeline_config = json.load(handle)

    @property
    def stats(self) -> Dict[str, Dict[str, float]]:
        if self._stats is None:
            self.load()
        return self._stats or {}

    @property
    def pipeline_config(self) -> Dict[str, Any]:
        if self._pipeline_config is None:
            self.load()
        return self._pipeline_config or {}

    @property
    def observation_hours(self) -> int:
        return int(self.pipeline_config.get("observation_hours", 24))

    def records_to_frame(self, records: Sequence[ObservationRecord]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for index, record in enumerate(records, start=1):
            row = {feature: np.nan for feature in DYNAMIC_FEATURES + STATIC_FEATURES}
            row.update(record.values)
            row["ICULOS"] = float(row.get("ICULOS", index))
            rows.append(row)
        return pd.DataFrame(rows)

    def _prepare(self, records: Sequence[ObservationRecord]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.records_to_frame(records)
        raw_dynamic = df[DYNAMIC_FEATURES].copy()
        dynamic_ffill = raw_dynamic.ffill().fillna(self.stats["fill_medians"])
        static_df = df[STATIC_FEATURES].copy().ffill().bfill().fillna(self.stats["static_fill_values"])
        return df, raw_dynamic, dynamic_ffill.join(static_df)

    def build_tabular_features(self, records: Sequence[ObservationRecord]) -> Dict[str, float]:
        df, raw_dynamic, prepared = self._prepare(records)
        dynamic_filled = prepared[DYNAMIC_FEATURES]
        static_df = prepared[STATIC_FEATURES]

        if len(df) > self.observation_hours:
            raw_dynamic = raw_dynamic.iloc[-self.observation_hours :]
            dynamic_filled = dynamic_filled.iloc[-self.observation_hours :]
            static_df = static_df.iloc[-self.observation_hours :]

        features: Dict[str, float] = {}
        static_row = static_df.iloc[-1]
        for feature in STATIC_FEATURES:
            features[feature] = float(static_row[feature])

        total_missing = 0
        for feature in DYNAMIC_FEATURES:
            values = dynamic_filled[feature].to_numpy(dtype=np.float32)
            observed_mask = raw_dynamic[feature].notna().to_numpy(dtype=np.float32)
            total_missing += int((1.0 - observed_mask).sum())

            observed_positions = np.where(observed_mask > 0)[0]
            hours_since_seen = float(len(values) - 1 - observed_positions[-1]) if len(observed_positions) else float(len(values))

            features[f"{feature}__last"] = float(values[-1])
            features[f"{feature}__mean"] = float(values.mean())
            features[f"{feature}__std"] = float(values.std())
            features[f"{feature}__min"] = float(values.min())
            features[f"{feature}__max"] = float(values.max())
            features[f"{feature}__delta"] = float(values[-1] - values[0])
            features[f"{feature}__obs_frac"] = float(observed_mask.mean())
            features[f"{feature}__hours_since_seen"] = hours_since_seen

        features["window_missing_fraction"] = float(total_missing / (len(dynamic_filled) * len(DYNAMIC_FEATURES)))
        return features

    def build_sequence_tensor(self, records: Sequence[ObservationRecord]) -> np.ndarray:
        _, raw_dynamic, prepared = self._prepare(records)
        dynamic_filled = prepared[DYNAMIC_FEATURES]

        if len(dynamic_filled) > self.observation_hours:
            raw_dynamic = raw_dynamic.iloc[-self.observation_hours :]
            dynamic_filled = dynamic_filled.iloc[-self.observation_hours :]

        if len(dynamic_filled) < self.observation_hours:
            pad_rows = self.observation_hours - len(dynamic_filled)
            first_value = dynamic_filled.iloc[[0]].copy()
            first_raw = raw_dynamic.iloc[[0]].copy()
            dynamic_filled = pd.concat([first_value] * pad_rows + [dynamic_filled], ignore_index=True)
            raw_dynamic = pd.concat([first_raw] * pad_rows + [raw_dynamic], ignore_index=True)

        means = np.array([self.stats["value_means"][feature] for feature in DYNAMIC_FEATURES], dtype=np.float32)
        stds = np.array([self.stats["value_stds"][feature] for feature in DYNAMIC_FEATURES], dtype=np.float32)
        values = dynamic_filled.to_numpy(dtype=np.float32)
        values = (values - means) / stds
        masks = raw_dynamic.notna().to_numpy(dtype=np.float32)
        return np.concatenate([values, masks], axis=1)

