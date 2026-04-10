from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass
class RuntimeSettings:
    host: str = os.getenv("AGENTIC_ICU_HOST", "0.0.0.0")
    port: int = int(os.getenv("AGENTIC_ICU_PORT", "8000"))
    xgboost_model_path: str = os.getenv(
        "AGENTIC_ICU_XGB_MODEL",
        str(BASE_DIR / "artifacts" / "xgboost_deterioration_model.json"),
    )
    xgboost_metrics_path: str = os.getenv(
        "AGENTIC_ICU_XGB_METRICS",
        str(BASE_DIR / "artifacts" / "xgboost_metrics.json"),
    )
    sequence_model_path: str = os.getenv(
        "AGENTIC_ICU_SEQUENCE_MODEL",
        str(BASE_DIR / "artifacts" / "sequence_gru_model.pt"),
    )
    sequence_metrics_path: str = os.getenv(
        "AGENTIC_ICU_SEQUENCE_METRICS",
        str(BASE_DIR / "artifacts" / "sequence_gru_metrics.json"),
    )
    xgboost_calibrator_path: str = os.getenv(
        "AGENTIC_ICU_XGB_CALIBRATOR",
        str(BASE_DIR / "artifacts" / "xgboost_calibrator.pkl"),
    )
    sequence_calibrator_path: str = os.getenv(
        "AGENTIC_ICU_SEQUENCE_CALIBRATOR",
        str(BASE_DIR / "artifacts" / "sequence_gru_calibrator.pkl"),
    )
    train_statistics_path: str = os.getenv(
        "AGENTIC_ICU_TRAIN_STATS",
        str(BASE_DIR / "artifacts" / "train_statistics.json"),
    )
    pipeline_config_path: str = os.getenv(
        "AGENTIC_ICU_PIPELINE_CONFIG",
        str(BASE_DIR / "artifacts" / "pipeline_config.json"),
    )
    raw_data_dir: str = os.getenv(
        "AGENTIC_ICU_RAW_DATA",
        str(BASE_DIR / "data" / "raw"),
    )
    reports_dir: str = os.getenv(
        "AGENTIC_ICU_REPORTS_DIR",
        str(BASE_DIR / "reports"),
    )
    alert_policy_path: str = os.getenv(
        "AGENTIC_ICU_ALERT_POLICY",
        str(BASE_DIR / "configs" / "runtime_alert_policy.json"),
    )
    alert_policy_profiles_path: str = os.getenv(
        "AGENTIC_ICU_ALERT_POLICY_PROFILES",
        str(BASE_DIR / "configs" / "runtime_alert_policy_profiles.json"),
    )
    resp_sequence_model_path: str = os.getenv(
        "AGENTIC_ICU_RESP_SEQUENCE_MODEL",
        str(BASE_DIR / "artifacts" / "sequence_resp_gru_model.pt"),
    )
    resp_sequence_metrics_path: str = os.getenv(
        "AGENTIC_ICU_RESP_SEQUENCE_METRICS",
        str(BASE_DIR / "artifacts" / "sequence_resp_gru_metrics.json"),
    )
    resp_sequence_calibrator_path: str = os.getenv(
        "AGENTIC_ICU_RESP_SEQUENCE_CALIBRATOR",
        str(BASE_DIR / "artifacts" / "sequence_resp_gru_calibrator.pkl"),
    )
    resp_xgboost_model_path: str = os.getenv(
        "AGENTIC_ICU_RESP_XGB_MODEL",
        str(BASE_DIR / "artifacts" / "xgboost_resp_deterioration_model.json"),
    )
    resp_xgboost_metrics_path: str = os.getenv(
        "AGENTIC_ICU_RESP_XGB_METRICS",
        str(BASE_DIR / "artifacts" / "xgboost_resp_metrics.json"),
    )
    resp_xgboost_calibrator_path: str = os.getenv(
        "AGENTIC_ICU_RESP_XGB_CALIBRATOR",
        str(BASE_DIR / "artifacts" / "xgboost_resp_calibrator.pkl"),
    )

    def load_alert_policy(self) -> Dict[str, Any]:
        path = Path(self.alert_policy_path)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_alert_policy_profiles(self) -> Dict[str, Dict[str, Any]]:
        path = Path(self.alert_policy_profiles_path)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


settings = RuntimeSettings()
