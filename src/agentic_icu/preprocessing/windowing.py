from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from agentic_icu.domain.contracts import ObservationRecord
from agentic_icu.domain.features import DYNAMIC_FEATURES, STATIC_FEATURES

_SLOPE_WINDOW = 6  # hours used for 6h trend features


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
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.train_statistics_path.exists() and self.pipeline_config_path.exists()

    def load(self) -> None:
        if self._stats is not None:
            return
        with self._load_lock:
            if self._stats is not None:  # re-check after acquiring
                return
            with self.train_statistics_path.open("r", encoding="utf-8") as handle:
                stats = json.load(handle)
            with self.pipeline_config_path.open("r", encoding="utf-8") as handle:
                pipeline_config = json.load(handle)

            # Fail loudly if statistics don't cover every dynamic feature.
            # A missing key causes silent NaN corruption in the feature matrix.
            missing_medians = [f for f in DYNAMIC_FEATURES if f not in stats.get("fill_medians", {})]
            missing_means   = [f for f in DYNAMIC_FEATURES if f not in stats.get("value_means", {})]
            missing_stds    = [f for f in DYNAMIC_FEATURES if f not in stats.get("value_stds", {})]
            problems = []
            if missing_medians:
                problems.append(f"fill_medians missing: {missing_medians}")
            if missing_means:
                problems.append(f"value_means missing: {missing_means}")
            if missing_stds:
                problems.append(f"value_stds missing: {missing_stds}")
            if problems:
                raise ValueError(
                    f"train_statistics.json does not cover all DYNAMIC_FEATURES — {'; '.join(problems)}. "
                    "Re-run training or update the statistics file."
                )
            if "static_fill_values" not in stats:
                raise ValueError(
                    "train_statistics.json is missing 'static_fill_values'. "
                    "Re-run training or update the statistics file."
                )
            self._stats = stats
            self._pipeline_config = pipeline_config

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
        """Window length for tabular (XGBoost) feature extraction."""
        return int(self.pipeline_config.get("observation_hours", 24))

    @property
    def sequence_hours(self) -> int:
        """Sequence length the GRU models were trained on (patient_seq_max_hours)."""
        return int(self.pipeline_config.get("patient_seq_max_hours", self.observation_hours))

    def records_to_frame(self, records: Sequence[ObservationRecord]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for index, record in enumerate(records, start=1):
            row = {feature: np.nan for feature in DYNAMIC_FEATURES + STATIC_FEATURES}
            row.update(record.values)
            # dict.get() returns NaN (not the fallback) if the key already exists from
            # the initialiser above — read from record.values directly instead.
            row["ICULOS"] = float(record.values["ICULOS"]) if "ICULOS" in record.values else float(index)
            rows.append(row)
        return pd.DataFrame(rows)

    def _prepare(self, records: Sequence[ObservationRecord]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.records_to_frame(records)
        raw_dynamic = df[DYNAMIC_FEATURES].copy()
        dynamic_ffill = raw_dynamic.ffill().fillna(self.stats["fill_medians"])
        static_df = df[STATIC_FEATURES].copy().ffill().bfill().fillna(self.stats["static_fill_values"])
        return df, raw_dynamic, dynamic_ffill.join(static_df)

    @staticmethod
    def _slope_6h(values: np.ndarray) -> float:
        """Linear slope over the last _SLOPE_WINDOW values (rise per hour)."""
        tail = values[-_SLOPE_WINDOW:]
        if len(tail) < 2:
            return 0.0
        x = np.arange(len(tail), dtype=np.float32)
        m = float(np.polyfit(x, tail, 1)[0])
        return m

    def _add_composite_features(self, features: Dict[str, float], dynamic_filled: pd.DataFrame) -> None:
        """Compute the 13 clinical composite features added in v2 training (Phase 1.3).

        These must match exactly what kaggle_train_deterioration.py appends to
        the tabular record for each patient — otherwise training/serving skew
        persists for the XGBoost model.
        """
        hr  = features.get("HR__last",    0.0)
        sbp = features.get("SBP__last",   0.0)
        dbp = features.get("DBP__last",   0.0)
        spo2 = features.get("O2Sat__last", 0.0)
        fio2 = features.get("FiO2__last",  0.0)
        resp = features.get("Resp__last",  0.0)
        iculos = features.get("ICULOS",    0.0)
        hr_mean = features.get("HR__mean", 0.0)

        features["shock_index"]      = hr / sbp if sbp > 0 else 0.0
        features["pulse_pressure"]   = sbp - dbp
        features["spo2_fio2_ratio"]  = spo2 / fio2 if fio2 > 0 else 0.0
        features["map_computed"]     = (sbp + 2.0 * dbp) / 3.0
        features["qsofa_resp_flag"]  = 1.0 if resp >= 22.0 else 0.0
        features["qsofa_sbp_flag"]   = 1.0 if sbp <= 100.0 else 0.0
        features["qsofa_score"]      = features["qsofa_resp_flag"] + features["qsofa_sbp_flag"]
        features["iculos_x_hr_mean"] = iculos * hr_mean

        for col in ("HR", "SBP", "O2Sat", "Temp", "Resp"):
            arr = dynamic_filled[col].to_numpy(dtype=np.float32)
            features[f"{col}__slope_6h"] = self._slope_6h(arr)

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

        # Composite clinical features (must match training Phase 1.3 additions)
        self._add_composite_features(features, dynamic_filled)

        return features

    def build_sequence_tensor(self, records: Sequence[ObservationRecord]) -> np.ndarray:
        """Build the input tensor for the GRU sequence models.

        Uses sequence_hours (patient_seq_max_hours=72) — the length the GRU was
        trained on — NOT observation_hours (24), which is the tabular window.
        """
        _, raw_dynamic, prepared = self._prepare(records)
        dynamic_filled = prepared[DYNAMIC_FEATURES]
        seq_len = self.sequence_hours

        if len(dynamic_filled) > seq_len:
            raw_dynamic = raw_dynamic.iloc[-seq_len:]
            dynamic_filled = dynamic_filled.iloc[-seq_len:]

        if len(dynamic_filled) < seq_len:
            pad_rows = seq_len - len(dynamic_filled)
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
