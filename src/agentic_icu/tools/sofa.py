from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SofaScore:
    respiratory: Optional[int]    # 0-4 (SpO2/FiO2 proxy)
    coagulation: Optional[int]    # 0-4 (Platelets)
    liver: Optional[int]          # 0-4 (Bilirubin_total)
    cardiovascular: Optional[int] # 0-1 (MAP only, no vasopressor data)
    cns: Optional[int]            # always None — GCS not in dataset
    renal: Optional[int]          # 0-4 (Creatinine)
    total: int                    # sum of available components
    components_available: int     # how many components contributed
    interpretation: str           # "low"/"moderate"/"high"/"critical"


def _is_unavailable(features: Dict[str, float], last_key: str, obs_frac_key: str) -> bool:
    """Return True if a feature was never measured (both __last and __obs_frac are 0)."""
    return features.get(last_key, 0.0) == 0.0 and features.get(obs_frac_key, 0.0) == 0.0


def _spo2_only_score(spo2: float) -> int:
    """SpO2-based SOFA respiratory proxy when FiO2 is unavailable.

    Based on Pandharipande et al. (2016) and standard sepsis literature:
    maps SpO2 alone to the 0-4 SOFA respiratory scale, assuming the
    observed desaturation reflects pulmonary compromise.
    """
    if spo2 > 95:
        return 0
    elif spo2 >= 90:
        return 1
    elif spo2 >= 85:
        return 2
    elif spo2 >= 75:
        return 3
    else:
        return 4


def _respiratory_score(features: Dict[str, float]) -> Optional[int]:
    """SOFA respiratory component using SpO2/FiO2 ratio with SpO2-only fallback."""
    # Try the pre-computed ratio first
    ratio = features.get("spo2_fio2_ratio__last", 0.0)
    if ratio == 0.0:
        spo2 = features.get("O2Sat__last", 0.0)
        fio2 = features.get("FiO2__last", 0.0)
        if fio2 > 0.0:
            ratio = spo2 / fio2
        elif spo2 > 0.0:
            # FiO2 not measured but SpO2 available — use SpO2-only proxy
            return _spo2_only_score(spo2)
        else:
            return None

    if ratio <= 0.0:
        return None

    if ratio > 400:
        return 0
    elif ratio > 300:
        return 1
    elif ratio > 200:
        return 2
    elif ratio > 100:
        return 3
    else:
        return 4


def compute_sofa(features: Dict[str, float]) -> SofaScore:
    """Compute partial SOFA from the last-observation feature dict produced by RuntimePreprocessor."""

    # --- Respiratory ---
    if _is_unavailable(features, "spo2_fio2_ratio__last", "spo2_fio2_ratio__obs_frac") and \
       _is_unavailable(features, "O2Sat__last", "O2Sat__obs_frac"):
        respiratory = None
    else:
        respiratory = _respiratory_score(features)

    # --- Coagulation (Platelets ×10³/μL) ---
    if _is_unavailable(features, "Platelets__last", "Platelets__obs_frac"):
        coagulation = None
    else:
        plt = features.get("Platelets__last", 0.0)
        if plt >= 150:
            coagulation = 0
        elif plt >= 100:
            coagulation = 1
        elif plt >= 50:
            coagulation = 2
        elif plt >= 20:
            coagulation = 3
        else:
            coagulation = 4

    # --- Liver (Bilirubin_total, mg/dL) ---
    if _is_unavailable(features, "Bilirubin_total__last", "Bilirubin_total__obs_frac"):
        liver = None
    else:
        bili = features.get("Bilirubin_total__last", 0.0)
        if bili < 1.2:
            liver = 0
        elif bili < 2.0:
            liver = 1
        elif bili < 6.0:
            liver = 2
        elif bili < 12.0:
            liver = 3
        else:
            liver = 4

    # --- Cardiovascular (MAP, mmHg) ---
    if _is_unavailable(features, "MAP__last", "MAP__obs_frac"):
        cardiovascular = None
    else:
        map_val = features.get("MAP__last", 0.0)
        cardiovascular = 0 if map_val >= 70 else 1

    # --- CNS: always None (no GCS in dataset) ---
    cns: Optional[int] = None

    # --- Renal (Creatinine, mg/dL) ---
    if _is_unavailable(features, "Creatinine__last", "Creatinine__obs_frac"):
        renal = None
    else:
        cr = features.get("Creatinine__last", 0.0)
        if cr < 1.2:
            renal = 0
        elif cr < 2.0:
            renal = 1
        elif cr < 3.5:
            renal = 2
        elif cr < 5.0:
            renal = 3
        else:
            renal = 4

    # --- Aggregate ---
    components = [respiratory, coagulation, liver, cardiovascular, cns, renal]
    available = [c for c in components if c is not None]
    total = sum(available)
    components_available = len(available)

    if total <= 1:
        interpretation = "low"
    elif total <= 5:
        interpretation = "moderate"
    elif total <= 9:
        interpretation = "high"
    else:
        interpretation = "critical"

    return SofaScore(
        respiratory=respiratory,
        coagulation=coagulation,
        liver=liver,
        cardiovascular=cardiovascular,
        cns=cns,
        renal=renal,
        total=total,
        components_available=components_available,
        interpretation=interpretation,
    )
