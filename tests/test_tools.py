from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_icu.tools.evaluate_alert_policy import (
    decide_case,
    load_patient_window,
    merge_policy,
    summarize_policy,
)


def test_load_patient_window(tmp_path):
    """Test loading patient data from a PSV file."""
    patient_file = tmp_path / "pTest.psv"
    # Note: load_patient_window expects PSV format (delimiter=|)
    content = "HR|Temp|SepsisLabel\n80.0|37.0|0\n85.0|37.5|1\n"
    patient_file.write_text(content)

    records, ever_sepsis = load_patient_window(patient_file, max_rows=10)
    assert len(records) == 2
    assert ever_sepsis == 1
    assert records[0].values["HR"] == 80.0

def test_merge_policy_logic():
    """Test merging dict params into a policy object."""
    from agentic_icu.agents.reasoner import AlertPolicy
    base_policy = AlertPolicy(high_alert_max_score_threshold=0.9)
    policy_dict = {"high_alert_max_score_threshold": 0.85, "stable_priority": "medium"}

    merged = merge_policy(base_policy, policy_dict)
    assert merged.high_alert_max_score_threshold == 0.85
    assert merged.stable_priority == "medium"

def test_decide_case():
    """Test the decision logic for a single case."""
    case = {
        "signal_quality": {"signal_valid": True, "suppression_recommendation": False, "suppression_mode": "none"},
        "vitals_agent": {"status": "available", "score": 0.9, "risk_band": "high", "detail": "test", "decision_threshold": 0.5, "threshold_ratio": 1.8},
        "lab_agent": {"status": "available", "score": 0.1, "risk_band": "low", "detail": "test", "decision_threshold": 0.5, "threshold_ratio": 0.2}
    }
    # Create a policy dict that maps to AlertPolicy fields
    policy = {
        "high_alert_max_score_threshold": 0.8
    }

    alert_triggered, alert_type, priority = decide_case(case, policy)
    assert alert_triggered is True
    assert priority == "high"

def test_summarize_policy():
    """Test metrics aggregation for a sequence of cases."""
    cases = [
        {
            "patient_id": "p1",
            "ever_sepsis": 1,
            "signal_quality": {"signal_valid": True, "suppression_recommendation": False, "suppression_mode": "none"},
            "vitals_agent": {"status": "available", "score": 0.9, "risk_band": "high", "detail": "test", "decision_threshold": 0.5, "threshold_ratio": 1.8},
            "lab_agent": {"status": "available", "score": 0.1, "risk_band": "low", "detail": "test", "decision_threshold": 0.5, "threshold_ratio": 0.2}
        },
        {
            "patient_id": "p2",
            "ever_sepsis": 0,
            "signal_quality": {"signal_valid": True, "suppression_recommendation": False, "suppression_mode": "none"},
            "vitals_agent": {"status": "available", "score": 0.1, "risk_band": "low", "detail": "test", "decision_threshold": 0.5, "threshold_ratio": 0.2},
            "lab_agent": {"status": "available", "score": 0.1, "risk_band": "low", "detail": "test", "decision_threshold": 0.5, "threshold_ratio": 0.2}
        }
    ]
    policy = {"high_alert_max_score_threshold": 0.8}

    summary = summarize_policy("TestPolicy", policy, cases)
    assert summary["patients_evaluated"] == 2
    assert summary["metrics"]["balanced_accuracy"] == 1.0
