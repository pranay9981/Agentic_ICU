from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd
import shap

if TYPE_CHECKING:
    from agentic_icu.inference.tabular import XGBoostInference


# Human-readable labels for raw feature names used in XGBoost training.
# Pattern: <FEATURE>__<AGG> or composite name.
_BASE_LABELS: Dict[str, str] = {
    # Vitals
    "HR": "Heart rate",
    "O2Sat": "SpO2",
    "Temp": "Temperature",
    "SBP": "Systolic BP",
    "MAP": "Mean arterial pressure",
    "DBP": "Diastolic BP",
    "Resp": "Respiratory rate",
    "EtCO2": "End-tidal CO2",
    # Labs
    "BaseExcess": "Base excess",
    "HCO3": "Bicarbonate",
    "FiO2": "FiO2",
    "pH": "Blood pH",
    "PaCO2": "PaCO2",
    "SaO2": "SaO2",
    "AST": "AST",
    "BUN": "BUN",
    "Alkalinephos": "Alkaline phosphatase",
    "Calcium": "Calcium",
    "Chloride": "Chloride",
    "Creatinine": "Creatinine",
    "Bilirubin_direct": "Direct bilirubin",
    "Glucose": "Glucose",
    "Lactate": "Lactate",
    "Magnesium": "Magnesium",
    "Phosphate": "Phosphate",
    "Potassium": "Potassium",
    "Bilirubin_total": "Total bilirubin",
    "TroponinI": "Troponin I",
    "Hct": "Hematocrit",
    "Hgb": "Hemoglobin",
    "PTT": "PTT",
    "WBC": "WBC",
    "Fibrinogen": "Fibrinogen",
    "Platelets": "Platelet count",
    # Static
    "Age": "Patient age",
    "Gender": "Gender",
    "Unit1": "ICU unit 1",
    "Unit2": "ICU unit 2",
    "HospAdmTime": "Hours before ICU admission",
    "ICULOS": "ICU length of stay (h)",
}

_AGG_SUFFIXES: Dict[str, str] = {
    "__last": "(current)",
    "__mean": "(24h mean)",
    "__std": "(24h variability)",
    "__min": "(24h min)",
    "__max": "(24h max)",
    "__delta": "(24h change)",
    "__obs_frac": "(measurement freq.)",
    "__hours_since_seen": "(h since last measured)",
    "__slope_6h": "(6h trend)",
}

_COMPOSITE_LABELS: Dict[str, str] = {
    "shock_index": "Shock index (HR/SBP)",
    "pulse_pressure": "Pulse pressure (SBP−DBP)",
    "spo2_fio2_ratio": "SpO2/FiO2 ratio",
    "map_computed": "Computed MAP",
    "qsofa_resp_flag": "qSOFA: RR ≥ 22",
    "qsofa_sbp_flag": "qSOFA: SBP ≤ 100",
    "qsofa_score": "qSOFA score",
    "iculos_x_hr_mean": "ICU stay × avg HR",
    "window_missing_fraction": "Overall data missingness",
}


def feature_label(raw_name: str) -> str:
    """Return a human-readable label for a raw XGBoost feature name."""
    if raw_name in _COMPOSITE_LABELS:
        return _COMPOSITE_LABELS[raw_name]
    for suffix, suffix_label in _AGG_SUFFIXES.items():
        if raw_name.endswith(suffix):
            base = raw_name[: -len(suffix)]
            base_label = _BASE_LABELS.get(base, base.replace("_", " ").title())
            return f"{base_label} {suffix_label}"
    return _BASE_LABELS.get(raw_name, raw_name.replace("__", " ").replace("_", " ").title())


class TabularExplainer:
    """SHAP-based feature contribution explainer for the XGBoost tabular model.

    Lazily initialises the TreeExplainer on first use so startup latency is
    unchanged.  The explainer is cached after the first call.
    """

    def __init__(self, inference: "XGBoostInference") -> None:
        self._inference = inference
        self._explainer: Optional[shap.TreeExplainer] = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._explainer is not None:
            return
        with self._lock:
            if self._explainer is not None:  # re-check after acquiring lock
                return
            self._inference.load()  # idempotent
            self._explainer = shap.TreeExplainer(self._inference.model)

    def top_contributions(
        self,
        features: Dict[str, float],
        n: int = 3,
    ) -> tuple[List[dict], Dict[str, float]]:
        """Return the top-n SHAP-based feature contributions.

        Returns:
            top_list: list of dicts with keys feature, label, shap_value, direction
            all_contributions: dict mapping every feature to its SHAP value
        """
        self._ensure_loaded()
        cols = self._inference.feature_columns
        X = pd.DataFrame([[features.get(col, 0.0) for col in cols]], columns=cols)
        raw_shap = self._explainer.shap_values(X)
        # shap_values() on a binary Booster returns a 2-D array (1, n_features)
        shap_row = raw_shap[0] if raw_shap.ndim == 2 else raw_shap
        all_contributions: Dict[str, float] = {
            col: float(shap_row[i]) for i, col in enumerate(cols)
        }
        top_items = sorted(all_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
        top_list = [
            {
                "feature": feat,
                "label": feature_label(feat),
                "shap_value": val,
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            }
            for feat, val in top_items
        ]
        return top_list, all_contributions

    def format_explanation(self, top_list: List[dict]) -> str:
        """Compose a one-line plain-English explanation from top-n contributions."""
        if not top_list:
            return ""
        parts = []
        for item in top_list:
            arrow = "↑" if item["direction"] == "increases_risk" else "↓"
            parts.append(f"{item['label']} ({arrow})")
        return "Top lab signals: " + ", ".join(parts) + "."
