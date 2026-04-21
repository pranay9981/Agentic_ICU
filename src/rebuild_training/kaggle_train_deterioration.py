#!/usr/bin/env python
"""
Leakage-safe training pipeline for ICU deterioration prediction on Kaggle.

This script is intentionally independent from the legacy training code in this
repository. It is designed around medical-model hygiene:

1. Patient-level splits only.
2. Causal feature construction only from data available at prediction time.
3. 4-6 hour prediction horizon labels with ambiguous near-onset windows removed.
4. Preprocessing statistics fit on training patients only.
5. Separate validation and test cohorts.

Primary output:
- XGBoost tabular model trained from scratch on engineered causal window features.

Optional outputs:
- Leakage-safe sequence arrays for a future neural model.
- A simple GRU-based sequence model trained on those arrays.

Important:
- This pipeline trains against the provided `SepsisLabel` target by default.
- Pass --train_resp_failure to also train respiratory failure models using an
  O2Sat/FiO2 SF-ratio proxy label (SF < 135, sustained 2h, 8h lookahead).
  Proxy validated by threshold sweep: ~5.83% positive row rate, ~5,817 positive
  patients. Outputs: xgboost_resp_*, sequence_resp_gru_* artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("agentic_icu_rebuild")


STATIC_FEATURES = [
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
]

LABEL_COLUMN = "SepsisLabel"

ALL_CLINICAL_FEATURES = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
]

DYNAMIC_FEATURES = [col for col in ALL_CLINICAL_FEATURES if col not in STATIC_FEATURES]

# ── Respiratory failure proxy label constants ─────────────────────────────────
# Proxy: O2Sat/FiO2 (SF ratio) < threshold, sustained for SUSTAIN_HOURS
# consecutive hours, labelled at any anchor within LOOKAHEAD_HOURS of onset.
# Threshold SF < 135 chosen by feasibility sweep (5.83% positive row rate,
# ~5,817 positive patients) — maps to moderate-severe ARDS territory.
RESP_SF_THRESHOLD = 135.0
RESP_SUSTAIN_HOURS = 2
RESP_LOOKAHEAD_HOURS = 8


@dataclass
class PipelineConfig:
    data_dir: str
    output_dir: str
    observation_hours: int = 24
    horizon_min_hours: int = 4
    horizon_max_hours: int = 8
    train_resp_failure: bool = False  # Phase 6: second pass with SF-ratio proxy label
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 42
    max_patients: Optional[int] = None
    export_sequence_arrays: bool = False
    train_sequence_model: bool = False
    sequence_only: bool = False
    sequence_hidden_size: int = 256
    sequence_layers: int = 2
    sequence_dropout: float = 0.3
    sequence_batch_size: int = 128
    sequence_epochs: int = 30
    sequence_learning_rate: float = 5e-4
    sequence_bidirectional: bool = True
    sequence_focal_alpha: float = 0.75
    sequence_focal_gamma: float = 2.0
    sequence_early_stopping_patience: int = 10
    patient_seq_max_hours: int = 72  # max ICU hours used per patient for GRU training
    xgb_num_boost_round: int = 3000
    xgb_early_stopping_rounds: int = 200
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.7
    xgb_min_child_weight: int = 10
    xgb_max_delta_step: int = 5
    use_smote: bool = False
    smote_sampling_strategy: float = 0.3
    train_ensemble: bool = False


@dataclass
class PatientMeta:
    patient_id: str
    path: str
    row_count: int
    sepsis_onset_index: Optional[int]
    ever_sepsis: int
    resp_failure_onset_index: Optional[int] = None
    ever_resp_failure: int = 0


@dataclass
class TrainStatistics:
    fill_medians: Dict[str, float]
    value_means: Dict[str, float]
    value_stds: Dict[str, float]
    static_fill_values: Dict[str, float]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe ICU deterioration models from scratch."
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing patient PSV files."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory where artifacts will be written."
    )
    parser.add_argument(
        "--observation_hours",
        type=int,
        default=24,
        help="Causal history length used for each prediction window.",
    )
    parser.add_argument(
        "--horizon_min_hours",
        type=int,
        default=4,
        help="Minimum lead time for positive labels.",
    )
    parser.add_argument(
        "--horizon_max_hours",
        type=int,
        default=6,
        help="Maximum lead time for positive labels.",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.15, help="Validation patient fraction."
    )
    parser.add_argument(
        "--test_fraction", type=float, default=0.15, help="Test patient fraction."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Global seed for reproducibility."
    )
    parser.add_argument(
        "--max_patients", type=int, default=None, help="Optional cap for smoke tests."
    )
    parser.add_argument(
        "--export_sequence_arrays",
        action="store_true",
        help="Export causal sequence arrays for a later neural model.",
    )
    parser.add_argument(
        "--train_sequence_model",
        action="store_true",
        help="Train the GRU sequence model after exporting arrays.",
    )
    parser.add_argument(
        "--sequence_only",
        action="store_true",
        help="Skip dataset rebuilding and train the GRU only from previously exported sequence arrays.",
    )
    parser.add_argument(
        "--train_resp_failure",
        action="store_true",
        default=False,
        help="Phase 6: also train respiratory failure models using the SF-ratio proxy label.",
    )
    parser.add_argument("--sequence_hidden_size", type=int, default=256)
    parser.add_argument("--sequence_layers", type=int, default=2)
    parser.add_argument("--sequence_dropout", type=float, default=0.3)
    parser.add_argument("--sequence_batch_size", type=int, default=128)
    parser.add_argument("--sequence_epochs", type=int, default=30)
    parser.add_argument("--sequence_learning_rate", type=float, default=5e-4)
    parser.add_argument("--sequence_bidirectional", action="store_true", default=True)
    parser.add_argument("--sequence_focal_alpha", type=float, default=0.75)
    parser.add_argument("--sequence_focal_gamma", type=float, default=2.0)
    parser.add_argument("--sequence_early_stopping_patience", type=int, default=10)
    parser.add_argument(
        "--patient_seq_max_hours",
        type=int,
        default=72,
        help="Max ICU hours per patient sequence for GRU training.",
    )
    parser.add_argument("--xgb_num_boost_round", type=int, default=3000)
    parser.add_argument("--xgb_early_stopping_rounds", type=int, default=200)
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.03)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.7)
    parser.add_argument("--xgb_min_child_weight", type=int, default=10)
    parser.add_argument("--xgb_max_delta_step", type=int, default=5)
    parser.add_argument(
        "--use_smote",
        action="store_true",
        default=False,
        help="Apply SMOTE oversampling to the XGBoost training set (requires imbalanced-learn).",
    )
    parser.add_argument(
        "--smote_sampling_strategy",
        type=float,
        default=0.3,
        help="SMOTE target minority/majority ratio (0.3 = 30%% positives).",
    )
    parser.add_argument(
        "--train_ensemble",
        action="store_true",
        default=False,
        help="Train a logistic-regression meta-learner on GRU + XGBoost calibrated val probabilities.",
    )
    args = parser.parse_args()
    return PipelineConfig(**vars(args))


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_patient_files(data_dir: str, max_patients: Optional[int]) -> List[Path]:
    paths = sorted(Path(data_dir).glob("*.psv"))
    if not paths:
        paths = sorted(Path(data_dir).glob("**/*.psv"))
    if max_patients is not None:
        paths = paths[:max_patients]
    if not paths:
        raise FileNotFoundError(f"No PSV files found under {data_dir}")
    logger.info("Discovered %s patient files.", len(paths))
    return paths


def read_patient_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    expected = set(DYNAMIC_FEATURES + STATIC_FEATURES + [LABEL_COLUMN])
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")
    df = df.sort_values("ICULOS").reset_index(drop=True)
    return df


def build_manifest(paths: Sequence[Path]) -> pd.DataFrame:
    rows: List[PatientMeta] = []
    for idx, path in enumerate(paths, start=1):
        if idx % 1000 == 0:
            logger.info("Manifest scan: %s / %s patients", idx, len(paths))
        df = pd.read_csv(path, sep="|")
        labels = df[LABEL_COLUMN].fillna(0).astype(int)
        sepsis_onset = int(labels.idxmax()) if labels.max() > 0 else None
        resp_onset = build_resp_failure_onset_index(df)
        rows.append(
            PatientMeta(
                patient_id=path.stem,
                path=str(path),
                row_count=len(df),
                sepsis_onset_index=sepsis_onset,
                ever_sepsis=int(labels.max() > 0),
                resp_failure_onset_index=resp_onset,
                ever_resp_failure=int(resp_onset is not None),
            )
        )
    return pd.DataFrame([asdict(row) for row in rows])


def stratified_patient_split(
    manifest: pd.DataFrame, cfg: PipelineConfig
) -> pd.DataFrame:
    patient_ids = manifest["patient_id"]
    labels = manifest["ever_sepsis"]
    train_val_ids, test_ids = train_test_split(
        patient_ids,
        test_size=cfg.test_fraction,
        random_state=cfg.random_seed,
        stratify=labels,
    )

    train_val = manifest[manifest["patient_id"].isin(train_val_ids)].copy()
    val_share_within_train_val = cfg.val_fraction / (1.0 - cfg.test_fraction)
    train_ids, val_ids = train_test_split(
        train_val["patient_id"],
        test_size=val_share_within_train_val,
        random_state=cfg.random_seed,
        stratify=train_val["ever_sepsis"],
    )

    split_map = {
        **{patient_id: "train" for patient_id in train_ids},
        **{patient_id: "val" for patient_id in val_ids},
        **{patient_id: "test" for patient_id in test_ids},
    }
    out = manifest.copy()
    out["split"] = out["patient_id"].map(split_map)
    return out


def causal_prepare_patient_frame(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dynamic = df[DYNAMIC_FEATURES].copy()
    dynamic_ffill = raw_dynamic.ffill()
    static = df[STATIC_FEATURES].copy().ffill().bfill()
    return raw_dynamic, dynamic_ffill, static


def fit_train_statistics(train_paths: Sequence[Path]) -> TrainStatistics:
    dynamic_blocks: List[pd.DataFrame] = []
    static_blocks: List[pd.DataFrame] = []

    for idx, path in enumerate(train_paths, start=1):
        if idx % 1000 == 0:
            logger.info(
                "Fitting statistics: %s / %s training patients", idx, len(train_paths)
            )
        df = read_patient_frame(path)
        _, dynamic_ffill, static = causal_prepare_patient_frame(df)
        dynamic_blocks.append(dynamic_ffill)
        static_blocks.append(static)

    dynamic_df = pd.concat(dynamic_blocks, ignore_index=True)
    static_df = pd.concat(static_blocks, ignore_index=True)

    fill_medians = dynamic_df.median(numeric_only=True).to_dict()
    dynamic_filled = dynamic_df.fillna(fill_medians)
    value_means = dynamic_filled.mean(numeric_only=True).to_dict()
    value_stds = (
        dynamic_filled.std(numeric_only=True).replace(0, 1.0).fillna(1.0).to_dict()
    )
    static_fill_values = static_df.median(numeric_only=True).to_dict()

    return TrainStatistics(
        fill_medians={str(k): float(v) for k, v in fill_medians.items()},
        value_means={str(k): float(v) for k, v in value_means.items()},
        value_stds={str(k): float(v) for k, v in value_stds.items()},
        static_fill_values={str(k): float(v) for k, v in static_fill_values.items()},
    )


def apply_causal_imputation(
    dynamic_ffill: pd.DataFrame, stats: TrainStatistics
) -> pd.DataFrame:
    return dynamic_ffill.fillna(stats.fill_medians)


def apply_static_fill(static_df: pd.DataFrame, stats: TrainStatistics) -> pd.DataFrame:
    return static_df.fillna(stats.static_fill_values)


def classify_anchor(
    onset_index: Optional[int], anchor_index: int, cfg: PipelineConfig
) -> Optional[int]:
    if onset_index is None:
        return 0
    if anchor_index >= onset_index:
        return None

    hours_to_onset = onset_index - anchor_index
    if cfg.horizon_min_hours <= hours_to_onset <= cfg.horizon_max_hours:
        return 1
    if hours_to_onset < cfg.horizon_min_hours:
        return None
    return 0


def build_resp_failure_onset_index(df: pd.DataFrame) -> Optional[int]:
    """Compute the first row index where the SF-ratio proxy label = 1.

    Proxy definition (validated by threshold sweep, 2026-04-08):
      - SF ratio = O2Sat / FiO2
      - Criteria: SF < RESP_SF_THRESHOLD (135) sustained >= RESP_SUSTAIN_HOURS (2)
        consecutive hours
      - Label at anchor t = 1 if criteria sustained within t+1 .. t+RESP_LOOKAHEAD_HOURS

    Why 8h lookahead: respiratory failure develops gradually over hours — there are
    measurable precursor trends (rising FiO2 requirement, drifting SpO2) before the
    threshold is crossed. 8h gives clinical teams time to escalate oxygen, order
    imaging, and prepare for intubation while remaining achievable for the model.
    """
    o2sat = df["O2Sat"].to_numpy(dtype=np.float64)
    fio2 = df["FiO2"].to_numpy(dtype=np.float64)
    n = len(df)

    # Forward-fill O2Sat and FiO2 so isolated missing rows don't break the run
    last_o2 = np.nan
    last_fi = np.nan
    sf_vals: List[Optional[float]] = []
    for i in range(n):
        if np.isfinite(o2sat[i]):
            last_o2 = o2sat[i]
        if np.isfinite(fio2[i]):
            last_fi = fio2[i]
        if np.isfinite(last_o2) and np.isfinite(last_fi) and last_fi > 0:
            sf_vals.append(last_o2 / last_fi)
        else:
            sf_vals.append(None)

    # Sustained flag: True at row i if criteria holds for RESP_SUSTAIN_HOURS ending at i
    sustained = [False] * n
    run = 0
    for i in range(n):
        sf = sf_vals[i]
        if sf is not None and sf < RESP_SF_THRESHOLD:
            run += 1
        else:
            run = 0
        sustained[i] = run >= RESP_SUSTAIN_HOURS

    # Lookahead label: row t labelled 1 if any of t+1..t+RESP_LOOKAHEAD_HOURS is sustained
    labels = [False] * n
    for i in range(n):
        for j in range(i + 1, min(i + RESP_LOOKAHEAD_HOURS + 1, n)):
            if sustained[j]:
                labels[i] = True
                break

    first_positive = next((i for i, v in enumerate(labels) if v), None)
    return first_positive


def _to_onset_int(value) -> Optional[int]:
    """Convert a manifest onset_index value to int or None.

    pandas stores Python None as NaN (float64) in mixed-type columns.
    itertuples() therefore returns nan, not None, for negative patients.
    This helper normalises both representations to the canonical form used
    throughout the pipeline (int for a real onset, None for no onset).
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def build_tabular_records_for_patient(
    path: Path,
    onset_index: Optional[int],
    stats: TrainStatistics,
    cfg: PipelineConfig,
) -> List[Dict[str, Any]]:
    df = read_patient_frame(path)
    raw_dynamic, dynamic_ffill, static = causal_prepare_patient_frame(df)
    dynamic_filled = apply_causal_imputation(dynamic_ffill, stats)
    static = apply_static_fill(static, stats)

    records: List[Dict[str, Any]] = []
    min_anchor = cfg.observation_hours - 1

    for anchor_idx in range(min_anchor, len(df)):
        label = classify_anchor(onset_index, anchor_idx, cfg)
        if label is None:
            continue

        start_idx = anchor_idx - cfg.observation_hours + 1
        raw_window = raw_dynamic.iloc[start_idx : anchor_idx + 1]
        filled_window = dynamic_filled.iloc[start_idx : anchor_idx + 1]
        static_row = static.iloc[anchor_idx]

        row: Dict[str, Any] = {
            "patient_id": path.stem,
            "anchor_iculos": float(df.iloc[anchor_idx]["ICULOS"]),
            "target": int(label),
        }

        for feature in STATIC_FEATURES:
            row[feature] = float(static_row[feature])

        total_missing = 0
        for feature in DYNAMIC_FEATURES:
            values = filled_window[feature].to_numpy(dtype=np.float32)
            observed_mask = raw_window[feature].notna().to_numpy(dtype=np.float32)
            total_missing += int((1.0 - observed_mask).sum())

            observed_positions = np.where(observed_mask > 0)[0]
            hours_since_seen = (
                float(len(values) - 1 - observed_positions[-1])
                if len(observed_positions)
                else float(len(values))
            )

            row[f"{feature}__last"] = float(values[-1])
            row[f"{feature}__mean"] = float(values.mean())
            row[f"{feature}__std"] = float(values.std())
            row[f"{feature}__min"] = float(values.min())
            row[f"{feature}__max"] = float(values.max())
            row[f"{feature}__delta"] = float(values[-1] - values[0])
            row[f"{feature}__obs_frac"] = float(observed_mask.mean())
            row[f"{feature}__hours_since_seen"] = hours_since_seen

        row["window_missing_fraction"] = float(
            total_missing / (cfg.observation_hours * len(DYNAMIC_FEATURES))
        )

        # --- Composite clinical features (Phase 1.3) ---
        hr_last = row.get("HR__last", 80.0)
        sbp_last = row.get("SBP__last", 120.0)
        dbp_last = row.get("DBP__last", 80.0)
        resp_last = row.get("Resp__last", 16.0)
        o2sat_last = row.get("O2Sat__last", 98.0)
        fio2_last = row.get("FiO2__last", 0.21)

        # Shock index: HR / SBP (elevated >1.0 indicates hemodynamic stress)
        row["shock_index"] = float(hr_last / max(sbp_last, 1.0))

        # Pulse pressure: SBP - DBP (low <25 or high >100 are both concerning)
        row["pulse_pressure"] = float(sbp_last - dbp_last)

        # SpO2/FiO2 ratio (P/F ratio proxy; <300 suggests respiratory compromise)
        row["spo2_fio2_ratio"] = float(o2sat_last / max(fio2_last, 0.01))

        # Computed MAP: (SBP + 2*DBP) / 3 (more precise than measured MAP alone)
        row["map_computed"] = float((sbp_last + 2.0 * dbp_last) / 3.0)

        # qSOFA flags (each is a binary clinical alarm)
        row["qsofa_resp_flag"] = float(resp_last >= 22.0)
        row["qsofa_sbp_flag"] = float(sbp_last <= 100.0)
        row["qsofa_score"] = row["qsofa_resp_flag"] + row["qsofa_sbp_flag"]

        # Slope features: linear trend over last 6 observations for key vitals
        # A positive slope on HR or negative slope on SBP in the short window
        # captures acute deterioration not visible in 24h summary stats.
        slope_window = min(6, cfg.observation_hours)
        x_slope = np.arange(slope_window, dtype=np.float32)
        for slope_feat in ("HR", "SBP", "O2Sat", "Temp", "Resp"):
            feat_vals = filled_window[slope_feat].to_numpy(dtype=np.float32)[
                -slope_window:
            ]
            if len(feat_vals) >= 2 and feat_vals.std() > 0:
                slope_val = float(
                    np.polyfit(x_slope[-len(feat_vals) :], feat_vals, 1)[0]
                )
            else:
                slope_val = 0.0
            row[f"{slope_feat}__slope_6h"] = slope_val

        # ICULOS interaction: early ICU with abnormal HR is different risk than late ICU
        iculos_val = row.get("ICULOS", 0.0)
        hr_mean = row.get("HR__mean", 80.0)
        row["iculos_x_hr_mean"] = float(iculos_val * hr_mean)

        records.append(row)

    return records


def build_tabular_dataset(
    manifest: pd.DataFrame,
    split_name: str,
    stats: TrainStatistics,
    cfg: PipelineConfig,
    onset_column: str = "sepsis_onset_index",
) -> pd.DataFrame:
    split_manifest = manifest[manifest["split"] == split_name]
    all_records: List[Dict[str, Any]] = []
    for idx, row in enumerate(split_manifest.itertuples(index=False), start=1):
        if idx % 500 == 0:
            logger.info(
                "Building %s tabular set: %s / %s patients",
                split_name,
                idx,
                len(split_manifest),
            )
        onset_index = _to_onset_int(getattr(row, onset_column, None))
        records = build_tabular_records_for_patient(
            path=Path(str(getattr(row, "path"))),
            onset_index=onset_index,
            stats=stats,
            cfg=cfg,
        )
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            f"No training windows produced for split={split_name}, onset_column={onset_column}"
        )
    return pd.DataFrame(all_records)


def get_xgb_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if col not in {"patient_id", "anchor_iculos", "target"}
    ]


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    labels = np.unique(y_true)
    if labels.size < 2:
        return 1.0 if labels.size == 1 and labels[0] == 1 else 0.0
    return float(average_precision_score(y_true, y_prob))


def choose_threshold_from_validation(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    if np.unique(y_true).size < 2:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f2": 0.0}

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f2": 0.0}

    precision = precision[:-1]
    recall = recall[:-1]
    beta_sq = 4.0
    scores = (
        (1 + beta_sq)
        * precision
        * recall
        / np.maximum(beta_sq * precision + recall, 1e-8)
    )
    best_idx = int(np.argmax(scores))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f2": float(scores[best_idx]),
    }


def score_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": safe_auc(y_true, y_prob),
        "average_precision": safe_average_precision(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "positive_rate": float(np.mean(y_true)),
        "prediction_rate": float(np.mean(y_pred)),
        "classification_report": classification_report(
            y_true, y_pred, digits=4, output_dict=True, zero_division=0
        ),
    }


def train_xgboost_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
    output_dir: Path,
    model_prefix: str = "xgboost",
) -> Dict[str, Any]:
    feature_cols = get_xgb_feature_columns(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    if y_train.nunique() < 2:
        raise RuntimeError(
            "Training windows contain only one class. Increase the cohort size or run on the full dataset."
        )
    X_val = val_df[feature_cols]
    y_val = val_df["target"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    pos_count = int(y_train.sum())
    neg_count = int((1 - y_train).sum())
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    # SMOTE oversampling (optional): synthesise minority-class windows before
    # building the DMatrix.  SMOTE operates in feature space, not patient space,
    # so it must run AFTER window-level feature extraction.
    if cfg.use_smote:
        try:
            from imblearn.over_sampling import SMOTE  # type: ignore

            smote = SMOTE(
                sampling_strategy=cfg.smote_sampling_strategy,
                random_state=cfg.random_seed,
            )
            X_train_sm, y_train_sm = smote.fit_resample(
                X_train.to_numpy(), y_train.to_numpy()
            )
            logger.info(
                "SMOTE applied: %d → %d training rows (pos rate %.3f → %.3f)",
                len(y_train),
                len(y_train_sm),
                float(y_train.mean()),
                float(y_train_sm.mean()),
            )
            dtrain = xgb.DMatrix(
                X_train_sm, label=y_train_sm, feature_names=feature_cols
            )
            # Recompute scale_pos_weight on the SMOTE-augmented set
            pos_count = int(y_train_sm.sum())
            neg_count = int((1 - y_train_sm).sum())
            scale_pos_weight = float(neg_count / max(pos_count, 1))
        except ImportError:
            logger.warning(
                "--use_smote requires imbalanced-learn: pip install imbalanced-learn. Skipping SMOTE."
            )

    # CRITICAL: put aucpr LAST — XGBoost early_stopping_rounds monitors the
    # last metric in the list.  When auc was last (original bug), training stopped
    # at iteration ~92 (AUC plateaus fast on imbalanced data) before AUPRC had
    # time to improve.  Now early stopping is gated on AUPRC directly.
    eval_metrics = ["logloss", "auc", "aucpr"] if y_val.nunique() > 1 else ["logloss"]

    params = {
        "objective": "binary:logistic",
        "eval_metric": eval_metrics,
        "max_depth": cfg.xgb_max_depth,
        "learning_rate": cfg.xgb_learning_rate,
        "subsample": cfg.xgb_subsample,
        "colsample_bytree": cfg.xgb_colsample_bytree,
        "min_child_weight": cfg.xgb_min_child_weight,
        "max_delta_step": cfg.xgb_max_delta_step,
        "lambda": 1.5,
        "alpha": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
    }
    if torch.cuda.is_available():
        params["device"] = "cuda"

    logger.info(
        "Training XGBoost | features=%d scale_pos_weight=%.2f max_depth=%d min_child_weight=%d max_delta_step=%d",
        len(feature_cols),
        scale_pos_weight,
        cfg.xgb_max_depth,
        cfg.xgb_min_child_weight,
        cfg.xgb_max_delta_step,
    )
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=cfg.xgb_num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=cfg.xgb_early_stopping_rounds,
        verbose_eval=100,
    )

    model_path = output_dir / f"{model_prefix}_deterioration_model.json"
    booster.save_model(model_path)

    val_prob = booster.predict(dval)
    threshold_info = choose_threshold_from_validation(y_val.to_numpy(), val_prob)
    threshold = threshold_info["threshold"]
    test_prob = booster.predict(dtest)

    metrics = {
        "feature_count": len(feature_cols),
        "best_iteration": int(booster.best_iteration),
        "best_score": float(booster.best_score),
        "threshold_selection": threshold_info,
        "validation_metrics": score_predictions(y_val.to_numpy(), val_prob, threshold),
        "test_metrics": score_predictions(y_test.to_numpy(), test_prob, threshold),
        "feature_columns": feature_cols,
    }

    importance = booster.get_score(importance_type="gain")
    metrics["feature_importance_gain"] = dict(
        sorted(importance.items(), key=lambda item: item[1], reverse=True)[:50]
    )

    # Isotonic calibration: fit on validation set so the output is a calibrated probability
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_prob, y_val.to_numpy())
    calibrator_path = output_dir / f"{model_prefix}_calibrator.pkl"
    with calibrator_path.open("wb") as fh:
        pickle.dump(calibrator, fh)

    cal_val_prob = calibrator.predict(val_prob)
    cal_test_prob = calibrator.predict(test_prob)
    metrics["calibration"] = {
        "method": "isotonic_regression",
        "val_metrics_calibrated": score_predictions(
            y_val.to_numpy(), cal_val_prob, threshold
        ),
        "test_metrics_calibrated": score_predictions(
            y_test.to_numpy(), cal_test_prob, threshold
        ),
    }

    with (output_dir / f"{model_prefix}_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def train_ensemble_model(
    cfg: PipelineConfig,
    output_dir: Path,
    xgb_model_prefix: str = "xgboost",
    gru_model_prefix: str = "sequence_gru",
) -> Dict[str, Any]:
    """Train a logistic-regression meta-learner that fuses GRU + XGBoost val probabilities.

    Requires that both models have been trained and their calibrators saved to output_dir.
    The meta-learner is trained on validation-set probabilities (not training-set) to avoid
    leaking the base models' training signal into the stacked layer.
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import roc_auc_score, average_precision_score  # type: ignore

    # ── Load XGBoost on val set ───────────────────────────────────────────────
    xgb_model_path = output_dir / f"{xgb_model_prefix}_deterioration_model.json"
    xgb_metrics_path = output_dir / f"{xgb_model_prefix}_metrics.json"
    xgb_cal_path = output_dir / f"{xgb_model_prefix}_calibrator.pkl"
    val_parquet = output_dir / "tabular_val.parquet"

    if not all(p.exists() for p in [xgb_model_path, xgb_metrics_path, val_parquet]):
        raise FileNotFoundError(
            "Ensemble training requires xgb model, metrics, and tabular_val.parquet in output_dir. "
            "Run full pipeline first."
        )

    with xgb_metrics_path.open("r", encoding="utf-8") as fh:
        xgb_meta = json.load(fh)
    feature_cols = xgb_meta["feature_columns"]

    xgb_booster = xgb.Booster()
    xgb_booster.load_model(str(xgb_model_path))
    xgb_calibrator = None
    if xgb_cal_path.exists():
        with xgb_cal_path.open("rb") as fh:
            xgb_calibrator = pickle.load(fh)  # nosec B301

    val_df = pd.read_parquet(val_parquet)
    y_val = val_df["target"].astype(int).to_numpy()
    dval = xgb.DMatrix(val_df[feature_cols], label=y_val, feature_names=feature_cols)
    xgb_raw_val = xgb_booster.predict(dval)
    xgb_val_prob = (
        xgb_calibrator.predict(xgb_raw_val)
        if xgb_calibrator is not None
        else xgb_raw_val
    )

    # ── Load GRU on val set ───────────────────────────────────────────────────
    gru_val_y_path = (
        output_dir / f"{gru_model_prefix.replace('sequence_', 'sequence_')}_val_y.npy"
    )
    gru_val_y_path = output_dir / "sequence_val_y.npy"
    gru_val_X_path = output_dir / "sequence_val_X.npy"
    gru_metrics_path = output_dir / f"{gru_model_prefix}_metrics.json"
    gru_model_path = output_dir / f"{gru_model_prefix}_model.pt"
    gru_cal_path = output_dir / f"{gru_model_prefix}_calibrator.pkl"

    if not all(
        p.exists()
        for p in [gru_val_X_path, gru_val_y_path, gru_metrics_path, gru_model_path]
    ):
        raise FileNotFoundError(
            "Ensemble training requires sequence_val_X.npy, sequence_val_y.npy, "
            f"{gru_model_prefix}_metrics.json, and {gru_model_prefix}_model.pt in output_dir."
        )

    with gru_metrics_path.open("r", encoding="utf-8") as fh:
        gru_meta = json.load(fh)

    arch = gru_meta.get("architecture", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gru_net = SequenceGRU(
        input_size=int(gru_meta["input_size"]),
        hidden_size=int(arch.get("hidden_size", 256)),
        num_layers=int(arch.get("num_layers", 2)),
        dropout=float(arch.get("dropout", 0.3)),
        bidirectional=bool(arch.get("bidirectional", True)),
    ).to(device)
    gru_net.load_state_dict(
        torch.load(gru_model_path, map_location=device, weights_only=True)
    )
    gru_net.eval()

    gru_calibrator = None
    if gru_cal_path.exists():
        with gru_cal_path.open("rb") as fh:
            gru_calibrator = pickle.load(fh)  # nosec B301

    X_val_seq = np.load(gru_val_X_path, mmap_mode="r")
    y_val_seq = np.load(gru_val_y_path, mmap_mode="r")

    # GRU val set is patient-level; XGBoost val set is window-level.
    # Use GRU patient-level y for ensemble evaluation — ensemble is patient-level.
    gru_val_loader = build_sequence_loader(
        np.asarray(X_val_seq), np.asarray(y_val_seq), batch_size=256, shuffle=False
    )
    _, gru_raw_val, gru_val_true = evaluate_sequence_model(
        gru_net, gru_val_loader, device
    )
    gru_val_prob = (
        gru_calibrator.predict(gru_raw_val)
        if gru_calibrator is not None
        else gru_raw_val
    )

    # The two val sets have different granularities (patient vs window).
    # For the ensemble we use the GRU's patient-level labels and probabilities,
    # and aggregate XGBoost window probabilities to patient level (mean of last
    # 6 windows — mirrors inference-time behaviour).
    val_df["xgb_prob"] = xgb_val_prob
    xgb_patient_prob = (
        val_df.sort_values("anchor_iculos")
        .groupby("patient_id")["xgb_prob"]
        .apply(lambda s: float(s.tail(6).mean()))
    )

    # Align on patient IDs from the GRU val set
    gru_patient_ids = np.load(output_dir / "sequence_val_patient_ids.npy")
    xgb_aligned = np.array(
        [
            float(xgb_patient_prob.get(pid, xgb_val_prob.mean()))
            for pid in gru_patient_ids
        ]
    )

    X_meta = np.column_stack([gru_val_prob, xgb_aligned])
    y_meta = gru_val_true.astype(int)

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg.random_seed)
    meta.fit(X_meta, y_meta)

    meta_prob = meta.predict_proba(X_meta)[:, 1]
    ensemble_auc = (
        float(roc_auc_score(y_meta, meta_prob)) if np.unique(y_meta).size > 1 else None
    )
    ensemble_auprc = (
        float(average_precision_score(y_meta, meta_prob))
        if np.unique(y_meta).size > 1
        else None
    )

    logger.info(
        "Ensemble meta-learner | val_auc=%.4f val_auprc=%.4f | coefs=[gru=%.3f xgb=%.3f] intercept=%.3f",
        ensemble_auc or 0,
        ensemble_auprc or 0,
        float(meta.coef_[0][0]),
        float(meta.coef_[0][1]),
        float(meta.intercept_[0]),
    )

    ensemble_path = output_dir / "ensemble_meta.pkl"
    with ensemble_path.open("wb") as fh:
        pickle.dump(meta, fh)

    ensemble_metrics = {
        "val_auc": ensemble_auc,
        "val_auprc": ensemble_auprc,
        "coef_gru": float(meta.coef_[0][0]),
        "coef_xgb": float(meta.coef_[0][1]),
        "intercept": float(meta.intercept_[0]),
        "gru_val_auprc": (
            float(average_precision_score(y_meta, gru_val_prob))
            if np.unique(y_meta).size > 1
            else None
        ),
        "xgb_val_auprc": (
            float(average_precision_score(y_meta, xgb_aligned))
            if np.unique(y_meta).size > 1
            else None
        ),
    }
    with (output_dir / "ensemble_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(ensemble_metrics, fh, indent=2)

    return ensemble_metrics


def zscore_normalize(
    values: np.ndarray, stats: TrainStatistics, features: Sequence[str]
) -> np.ndarray:
    means = np.array(
        [stats.value_means[feature] for feature in features], dtype=np.float32
    )
    stds = np.array(
        [stats.value_stds[feature] for feature in features], dtype=np.float32
    )
    return (values - means) / stds


def build_sequence_arrays_for_patient(
    path: Path,
    onset_index: Optional[int],
    stats: TrainStatistics,
    cfg: PipelineConfig,
) -> Tuple[List[np.ndarray], List[int], List[str], List[int]]:
    df = read_patient_frame(path)
    raw_dynamic, dynamic_ffill, _ = causal_prepare_patient_frame(df)
    dynamic_filled = apply_causal_imputation(dynamic_ffill, stats)

    windows: List[np.ndarray] = []
    labels: List[int] = []
    patient_ids: List[str] = []
    anchors: List[int] = []

    min_anchor = cfg.observation_hours - 1
    for anchor_idx in range(min_anchor, len(df)):
        label = classify_anchor(onset_index, anchor_idx, cfg)
        if label is None:
            continue

        start_idx = anchor_idx - cfg.observation_hours + 1
        raw_window = raw_dynamic.iloc[start_idx : anchor_idx + 1]
        filled_window = dynamic_filled.iloc[start_idx : anchor_idx + 1]

        value_array = filled_window[DYNAMIC_FEATURES].to_numpy(dtype=np.float32)
        value_array = zscore_normalize(value_array, stats, DYNAMIC_FEATURES)
        mask_array = raw_window[DYNAMIC_FEATURES].notna().to_numpy(dtype=np.float32)
        sequence = np.concatenate([value_array, mask_array], axis=1)

        windows.append(sequence)
        labels.append(int(label))
        patient_ids.append(path.stem)
        anchors.append(int(df.iloc[anchor_idx]["ICULOS"]))

    return windows, labels, patient_ids, anchors


def export_sequence_arrays(
    manifest: pd.DataFrame,
    split_name: str,
    stats: TrainStatistics,
    cfg: PipelineConfig,
    output_dir: Path,
    onset_column: str = "sepsis_onset_index",
    array_prefix: str = "sequence",
) -> Dict[str, Any]:
    split_manifest = manifest[manifest["split"] == split_name]
    all_windows: List[np.ndarray] = []
    all_labels: List[int] = []
    all_patient_ids: List[str] = []
    all_anchors: List[int] = []

    for idx, row in enumerate(split_manifest.itertuples(index=False), start=1):
        if idx % 500 == 0:
            logger.info(
                "Building %s sequence set: %s / %s patients",
                split_name,
                idx,
                len(split_manifest),
            )
        onset_index = _to_onset_int(getattr(row, onset_column, None))
        windows, labels, patient_ids_batch, anchors_batch = build_sequence_arrays_for_patient(
            path=Path(str(getattr(row, "path"))),
            onset_index=onset_index,
            stats=stats,
            cfg=cfg,
        )
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_patient_ids.extend(patient_ids_batch)
        all_anchors.extend(anchors_batch)

    if not all_windows:
        raise RuntimeError(f"No sequence windows produced for split={split_name}")

    X_arr = np.asarray(all_windows, dtype=np.float32)
    y_arr = np.asarray(all_labels, dtype=np.float32)
    patient_ids_arr = np.asarray(all_patient_ids)
    anchors_arr = np.asarray(all_anchors, dtype=np.int32)

    np.save(output_dir / f"{array_prefix}_{split_name}_X.npy", X_arr)
    np.save(output_dir / f"{array_prefix}_{split_name}_y.npy", y_arr)
    np.save(output_dir / f"{array_prefix}_{split_name}_patient_ids.npy", patient_ids_arr)
    np.save(output_dir / f"{array_prefix}_{split_name}_anchor_iculos.npy", anchors_arr)

    return {
        "split": split_name,
        "shape": list(X_arr.shape),
        "positive_rate": float(y_arr.mean()),
    }


def build_patient_level_sequence(
    path: Path,
    onset_index: Optional[int],
    stats: TrainStatistics,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, int]:
    """Build one leakage-safe sequence for a single patient (patient-level GRU).

    Window-level labelling (one label per 24h sliding window) produces ~1%
    positive rate, starving the GRU of signal. Patient-level labelling gives
    ~7% positive rate — the same as the patient cohort — making training
    tractable.

    Leakage rule: for positive patients, only data up to
    onset_index - horizon_min_hours is included. The GRU never sees data from
    the deterioration window itself.

    The resulting sequence is truncated to the last cfg.patient_seq_max_hours
    rows (most recent data) then left-padded with zeros to exactly
    cfg.patient_seq_max_hours rows, so the GRU's final hidden state always
    encodes the most recent physiology.

    Returns:
        X: float32 array of shape [patient_seq_max_hours, n_seq_features]
        y: int  (1 = positive patient, 0 = negative)
    """
    df = read_patient_frame(path)
    raw_dynamic, dynamic_ffill, _ = causal_prepare_patient_frame(df)
    dynamic_filled = apply_causal_imputation(dynamic_ffill, stats)

    # Normalise onset_index: pandas may pass nan instead of None for negative patients
    onset_index = _to_onset_int(onset_index)
    if onset_index is not None:
        cutoff = int(max(0, onset_index - cfg.horizon_min_hours))
        label = 1
    else:
        cutoff = len(df)
        label = 0

    raw_slice = raw_dynamic.iloc[:cutoff]
    filled_slice = dynamic_filled.iloc[:cutoff]

    max_hours = cfg.patient_seq_max_hours
    if len(filled_slice) > max_hours:
        raw_slice = raw_slice.iloc[-max_hours:]
        filled_slice = filled_slice.iloc[-max_hours:]

    value_array = filled_slice[DYNAMIC_FEATURES].to_numpy(dtype=np.float32)
    value_array = zscore_normalize(value_array, stats, DYNAMIC_FEATURES)
    mask_array = raw_slice[DYNAMIC_FEATURES].notna().to_numpy(dtype=np.float32)
    seq = np.concatenate([value_array, mask_array], axis=1)  # [T, n_features]

    # Left-pad: zeros mean "no data yet"; mask channel of 0 = missing
    n_features = seq.shape[1]
    if len(seq) < max_hours:
        pad = np.zeros((max_hours - len(seq), n_features), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)

    return seq, label


def export_patient_sequences(
    manifest: pd.DataFrame,
    split_name: str,
    stats: TrainStatistics,
    cfg: PipelineConfig,
    output_dir: Path,
    onset_column: str = "sepsis_onset_index",
    array_prefix: str = "sequence",
) -> Dict[str, Any]:
    """Export patient-level sequence arrays: one sequence per patient.

    Produces X of shape [n_patients, patient_seq_max_hours, n_features] and
    y of shape [n_patients] using patient-level labels (~7% positive rate vs
    ~1% for window-level). This is the training data for the patient-level GRU.
    """
    split_manifest = manifest[manifest["split"] == split_name]
    all_sequences: List[np.ndarray] = []
    all_labels: List[int] = []
    all_patient_ids: List[str] = []

    for idx, row in enumerate(split_manifest.itertuples(index=False), start=1):
        if idx % 1000 == 0:
            logger.info(
                "Building %s patient sequences: %s / %s",
                split_name,
                idx,
                len(split_manifest),
            )
        onset_index = _to_onset_int(getattr(row, onset_column, None))
        seq, label = build_patient_level_sequence(
            path=Path(str(getattr(row, "path"))),
            onset_index=onset_index,
            stats=stats,
            cfg=cfg,
        )
        all_sequences.append(seq)
        all_labels.append(label)
        all_patient_ids.append(str(getattr(row, "patient_id")))

    if not all_sequences:
        raise RuntimeError(f"No patient sequences produced for split={split_name}")

    X = np.asarray(all_sequences, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.float32)
    patient_ids_arr = np.asarray(all_patient_ids)

    np.save(output_dir / f"{array_prefix}_{split_name}_X.npy", X)
    np.save(output_dir / f"{array_prefix}_{split_name}_y.npy", y)
    np.save(
        output_dir / f"{array_prefix}_{split_name}_patient_ids.npy", patient_ids_arr
    )

    return {
        "split": split_name,
        "shape": list(X.shape),
        "positive_rate": float(y.mean()),
    }


class FocalLoss(nn.Module):
    """Focal loss for binary classification with severe class imbalance.

    Down-weights the loss contribution from easy negatives (gamma > 0),
    forcing the model to focus on hard positives. alpha controls the
    base weight for the positive class.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class SequenceGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_output_size),
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_output_size // 2, 1),
        )

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        final_state = output[:, -1, :]
        return self.classifier(final_state).squeeze(-1)


def build_sequence_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    # sampler is mutually exclusive with shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
    )


def evaluate_sequence_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: List[float] = []
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            per_row_loss = criterion(logits, yb)
            losses.append(float(per_row_loss.mean().item()))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.cpu().numpy())

    probs_out = np.concatenate(probabilities)
    # Replace any NaN/Inf from numerical issues with 0.5 (uninformative) so downstream metrics don't crash
    probs_out = np.nan_to_num(probs_out, nan=0.5, posinf=1.0, neginf=0.0)
    return float(np.mean(losses)), probs_out, np.concatenate(labels)


def resolve_sequence_training_device() -> torch.device:
    if not torch.cuda.is_available():
        logger.info("Sequence training device: cpu")
        return torch.device("cpu")

    cuda_device = torch.device("cuda")
    try:
        # Probe the exact GRU path that failed on Kaggle P100 instances with an
        # incompatible PyTorch CUDA build, and fall back cleanly if it breaks.
        probe = nn.GRU(input_size=1, hidden_size=1, batch_first=True).to(cuda_device)
        sample = torch.zeros((1, 1, 1), device=cuda_device)
        with torch.no_grad():
            probe(sample)
        torch.cuda.synchronize()
        logger.info("Sequence training device: cuda")
        return cuda_device
    except Exception as exc:
        logger.warning(
            "CUDA is available but incompatible for GRU training; falling back to CPU. Reason: %s",
            exc,
        )
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return torch.device("cpu")


def train_sequence_model(
    cfg: PipelineConfig,
    output_dir: Path,
    array_prefix: str = "sequence",
    model_prefix: str = "sequence_gru",
) -> Dict[str, Any]:
    X_train = np.load(output_dir / f"{array_prefix}_train_X.npy", mmap_mode="r")
    y_train = np.load(output_dir / f"{array_prefix}_train_y.npy", mmap_mode="r")
    X_val = np.load(output_dir / f"{array_prefix}_val_X.npy", mmap_mode="r")
    y_val = np.load(output_dir / f"{array_prefix}_val_y.npy", mmap_mode="r")
    X_test = np.load(output_dir / f"{array_prefix}_test_X.npy", mmap_mode="r")
    y_test = np.load(output_dir / f"{array_prefix}_test_y.npy", mmap_mode="r")

    if np.unique(np.asarray(y_train)).size < 2:
        raise RuntimeError(
            "Sequence training windows contain only one class. Increase the cohort size or run on the full dataset."
        )

    y_train_arr = np.asarray(y_train)
    pos_count = float(y_train_arr.sum())
    neg_count = float(len(y_train_arr) - pos_count)
    # Weight each sample so the effective positive rate per batch is ~50%.
    # This is the sequence equivalent of XGBoost's scale_pos_weight.
    sample_weights = np.where(y_train_arr == 1, neg_count / max(pos_count, 1.0), 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = build_sequence_loader(
        np.asarray(X_train),
        y_train_arr,
        cfg.sequence_batch_size,
        shuffle=False,
        sampler=sampler,
    )
    val_loader = build_sequence_loader(
        np.asarray(X_val), np.asarray(y_val), cfg.sequence_batch_size, shuffle=False
    )
    test_loader = build_sequence_loader(
        np.asarray(X_test), np.asarray(y_test), cfg.sequence_batch_size, shuffle=False
    )

    device = resolve_sequence_training_device()
    model = SequenceGRU(
        input_size=int(X_train.shape[-1]),
        hidden_size=cfg.sequence_hidden_size,
        num_layers=cfg.sequence_layers,
        dropout=cfg.sequence_dropout,
        bidirectional=cfg.sequence_bidirectional,
    ).to(device)
    model._init_weights()

    # BCEWithLogitsLoss with no pos_weight: WeightedRandomSampler already
    # balances the class distribution in each batch to ~50/50, so the loss
    # does not need an additional per-class weight. This mirrors XGBoost's
    # scale_pos_weight approach and avoids the loss-collapse seen with FocalLoss
    # on window-level data where positives are only ~1% of rows.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.sequence_learning_rate, weight_decay=1e-4
    )

    # Cosine annealing with 2-epoch linear warmup
    warmup_epochs = 2

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(
            max(cfg.sequence_epochs - warmup_epochs, 1)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_ap = -math.inf
    best_path = output_dir / f"{model_prefix}_model.pt"
    # Save initial weights as fallback so best_path always exists after the loop
    torch.save(model.state_dict(), best_path)
    history: List[Dict[str, float]] = []
    patience_counter = 0

    for epoch in range(cfg.sequence_epochs):
        model.train()
        epoch_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        val_loss, val_prob, val_true = evaluate_sequence_model(
            model, val_loader, device
        )

        # Guard against NaN probabilities (can occur with very small datasets or early divergence)
        if not np.isfinite(val_prob).all():
            logger.warning(
                "Epoch %s: val_prob contains NaN/Inf — skipping metric update",
                epoch + 1,
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(epoch_losses)),
                    "val_loss": float("nan"),
                    "val_average_precision": 0.0,
                    "val_auc": 0.0,
                    "lr": float(scheduler.get_last_lr()[0]),
                }
            )
            patience_counter += 1
            if patience_counter >= cfg.sequence_early_stopping_patience:
                logger.info("Early stopping at epoch %s (NaN divergence)", epoch + 1)
                break
            continue

        val_ap = safe_average_precision(val_true, val_prob)
        val_auc = safe_auc(val_true, val_prob) or 0.0

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": val_loss,
            "val_average_precision": val_ap,
            "val_auc": val_auc,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_summary)
        logger.info(
            "Sequence epoch %s/%s | train_loss=%.4f val_loss=%.4f val_ap=%.4f val_auc=%.4f lr=%.2e",
            epoch + 1,
            cfg.sequence_epochs,
            epoch_summary["train_loss"],
            epoch_summary["val_loss"],
            val_ap,
            val_auc,
            epoch_summary["lr"],
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.sequence_early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %s (patience=%s)",
                    epoch + 1,
                    cfg.sequence_early_stopping_patience,
                )
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _, val_prob, val_true = evaluate_sequence_model(model, val_loader, device)
    if not np.isfinite(val_prob).all():
        logger.warning(
            "Best model still produces non-finite val_prob — defaulting threshold to 0.5"
        )
        val_prob = np.nan_to_num(val_prob, nan=0.5, posinf=1.0, neginf=0.0)
    threshold_info = choose_threshold_from_validation(val_true, val_prob)
    threshold = threshold_info["threshold"]
    _, test_prob, test_true = evaluate_sequence_model(model, test_loader, device)

    metrics = {
        "threshold_selection": threshold_info,
        "validation_metrics": score_predictions(val_true, val_prob, threshold),
        "test_metrics": score_predictions(test_true, test_prob, threshold),
        "history": history,
        "input_size": int(X_train.shape[-1]),
        "observation_hours": int(X_train.shape[1]),
        "device": device.type,
        "architecture": {
            "hidden_size": cfg.sequence_hidden_size,
            "num_layers": cfg.sequence_layers,
            "dropout": cfg.sequence_dropout,
            "bidirectional": cfg.sequence_bidirectional,
            "loss": "bce_weighted_sampler",
            "sampler_pos_weight": float(neg_count / max(pos_count, 1.0)),
        },
    }

    # Isotonic calibration: fit on validation set (skip if val_prob has NaN from divergence)
    _, test_prob_raw, test_true_raw = evaluate_sequence_model(
        model, test_loader, device
    )
    if np.isfinite(val_prob).all() and np.unique(val_true).size > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_prob, val_true)
        calibrator_path = output_dir / f"{model_prefix}_calibrator.pkl"
        with calibrator_path.open("wb") as fh:
            pickle.dump(calibrator, fh)
        cal_val_prob = calibrator.predict(val_prob)
        cal_test_prob = calibrator.predict(test_prob_raw)
    else:
        logger.warning(
            "Skipping sequence calibration: val_prob is not finite or val set has single class."
        )
        calibrator = None
        cal_val_prob = val_prob
        cal_test_prob = test_prob_raw
    metrics["calibration"] = {
        "method": "isotonic_regression" if calibrator is not None else "none",
        "val_metrics_calibrated": (
            score_predictions(val_true, cal_val_prob, threshold)
            if calibrator is not None
            else None
        ),
        "test_metrics_calibrated": (
            score_predictions(test_true_raw, cal_test_prob, threshold)
            if calibrator is not None
            else None
        ),
    }

    with (output_dir / f"{model_prefix}_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def save_statistics(stats: TrainStatistics, output_dir: Path) -> None:
    with (output_dir / "train_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(stats), handle, indent=2)


def save_manifest(manifest: pd.DataFrame, output_dir: Path) -> None:
    manifest.to_csv(output_dir / "patient_manifest.csv", index=False)


def save_tabular_dataset(df: pd.DataFrame, split_name: str, output_dir: Path) -> None:
    df.to_parquet(output_dir / f"tabular_{split_name}.parquet", index=False)


def summarize_split(manifest: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    split_df = manifest[manifest["split"] == split_name]
    return {
        "patients": int(len(split_df)),
        "positive_patients": int(split_df["ever_sepsis"].sum()),
        "positive_patient_rate": float(split_df["ever_sepsis"].mean()),
    }


def run_pipeline(cfg: PipelineConfig) -> None:
    set_random_seed(cfg.random_seed)
    output_dir = ensure_output_dir(cfg.output_dir)

    with (output_dir / "pipeline_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    if cfg.sequence_only:
        if not cfg.train_sequence_model:
            raise ValueError("--sequence_only requires --train_sequence_model.")

        required_arrays = [
            output_dir / "sequence_train_X.npy",
            output_dir / "sequence_train_y.npy",
            output_dir / "sequence_val_X.npy",
            output_dir / "sequence_val_y.npy",
            output_dir / "sequence_test_X.npy",
            output_dir / "sequence_test_y.npy",
        ]
        missing = [str(path) for path in required_arrays if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Sequence-only training requires exported arrays in the output directory. Missing: "
                + ", ".join(missing)
            )

        logger.info("Sequence-only mode: reusing exported arrays from %s", output_dir)
        sequence_metrics = train_sequence_model(cfg, output_dir)
        logger.info(
            "Sequence GRU complete | test_auc=%.4f test_ap=%.4f",
            sequence_metrics["test_metrics"]["auc"],
            sequence_metrics["test_metrics"]["average_precision"],
        )
        return

    paths = discover_patient_files(cfg.data_dir, cfg.max_patients)
    manifest = build_manifest(paths)
    manifest = stratified_patient_split(manifest, cfg)
    save_manifest(manifest, output_dir)

    split_summary = {
        split: summarize_split(manifest, split) for split in ("train", "val", "test")
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, indent=2)
    logger.info("Split summary: %s", split_summary)

    train_paths = [
        Path(path) for path in manifest.loc[manifest["split"] == "train", "path"]
    ]
    stats = fit_train_statistics(train_paths)
    save_statistics(stats, output_dir)

    train_df = build_tabular_dataset(manifest, "train", stats, cfg)
    val_df = build_tabular_dataset(manifest, "val", stats, cfg)
    test_df = build_tabular_dataset(manifest, "test", stats, cfg)

    save_tabular_dataset(train_df, "train", output_dir)
    save_tabular_dataset(val_df, "val", output_dir)
    save_tabular_dataset(test_df, "test", output_dir)

    logger.info(
        "Tabular windows | train=%s val=%s test=%s",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    xgb_metrics = train_xgboost_model(train_df, val_df, test_df, cfg, output_dir)
    logger.info(
        "XGBoost complete | test_auc=%.4f test_ap=%.4f",
        xgb_metrics["test_metrics"]["auc"],
        xgb_metrics["test_metrics"]["average_precision"],
    )

    if cfg.export_sequence_arrays or cfg.train_sequence_model:
        sequence_summaries = {}
        for split_name in ("train", "val", "test"):
            sequence_summaries[split_name] = export_patient_sequences(
                manifest, split_name, stats, cfg, output_dir
            )
        with (output_dir / "sequence_export_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(sequence_summaries, handle, indent=2)
        logger.info("Sequence arrays exported: %s", sequence_summaries)

    if cfg.train_sequence_model:
        sequence_metrics = train_sequence_model(cfg, output_dir)
        logger.info(
            "Sequence GRU complete | test_auc=%.4f test_ap=%.4f",
            sequence_metrics["test_metrics"]["auc"],
            sequence_metrics["test_metrics"]["average_precision"],
        )

    # ── Phase 6: Respiratory failure second pass ──────────────────────────────
    if cfg.train_resp_failure:
        logger.info(
            "=== RESPIRATORY FAILURE MODEL (SF<%.0f, sustained %dh, lookahead %dh) ===",
            RESP_SF_THRESHOLD,
            RESP_SUSTAIN_HOURS,
            RESP_LOOKAHEAD_HOURS,
        )

        resp_pos = int(manifest["ever_resp_failure"].sum())
        logger.info(
            "Resp failure cohort: %s positive patients / %s total (%.1f%%)",
            resp_pos,
            len(manifest),
            100 * resp_pos / max(len(manifest), 1),
        )

        train_df_resp = build_tabular_dataset(
            manifest, "train", stats, cfg, onset_column="resp_failure_onset_index"
        )
        val_df_resp = build_tabular_dataset(
            manifest, "val", stats, cfg, onset_column="resp_failure_onset_index"
        )
        test_df_resp = build_tabular_dataset(
            manifest, "test", stats, cfg, onset_column="resp_failure_onset_index"
        )

        save_tabular_dataset(train_df_resp, "resp_train", output_dir)
        save_tabular_dataset(val_df_resp, "resp_val", output_dir)
        save_tabular_dataset(test_df_resp, "resp_test", output_dir)

        logger.info(
            "Resp tabular windows | train=%s val=%s test=%s",
            len(train_df_resp),
            len(val_df_resp),
            len(test_df_resp),
        )

        resp_xgb_metrics = train_xgboost_model(
            train_df_resp,
            val_df_resp,
            test_df_resp,
            cfg,
            output_dir,
            model_prefix="xgboost_resp",
        )
        logger.info(
            "Resp XGBoost complete | test_auc=%.4f test_ap=%.4f",
            resp_xgb_metrics["test_metrics"]["auc"],
            resp_xgb_metrics["test_metrics"]["average_precision"],
        )

        if cfg.export_sequence_arrays or cfg.train_sequence_model:
            resp_seq_summaries = {}
            for split_name in ("train", "val", "test"):
                resp_seq_summaries[split_name] = export_patient_sequences(
                    manifest,
                    split_name,
                    stats,
                    cfg,
                    output_dir,
                    onset_column="resp_failure_onset_index",
                    array_prefix="sequence_resp",
                )
            with (output_dir / "sequence_resp_export_summary.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(resp_seq_summaries, handle, indent=2)
            logger.info("Resp sequence arrays exported: %s", resp_seq_summaries)

        if cfg.train_sequence_model:
            resp_seq_metrics = train_sequence_model(
                cfg,
                output_dir,
                array_prefix="sequence_resp",
                model_prefix="sequence_resp_gru",
            )
            logger.info(
                "Resp GRU complete | test_auc=%.4f test_ap=%.4f",
                resp_seq_metrics["test_metrics"]["auc"],
                resp_seq_metrics["test_metrics"]["average_precision"],
            )

    # ── Ensemble meta-learner (optional) ─────────────────────────────────────
    if cfg.train_ensemble:
        if not cfg.train_sequence_model:
            logger.warning(
                "--train_ensemble requires --train_sequence_model to have run first."
            )
        else:
            logger.info("=== ENSEMBLE META-LEARNER ===")
            ensemble_metrics = train_ensemble_model(cfg, output_dir)
            logger.info(
                "Ensemble complete | val_auc=%.4f val_auprc=%.4f",
                ensemble_metrics.get("val_auc") or 0,
                ensemble_metrics.get("val_auprc") or 0,
            )


if __name__ == "__main__":
    run_pipeline(parse_args())
