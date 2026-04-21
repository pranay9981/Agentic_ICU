from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agentic_icu.agents.reasoner import AlertPolicy, ClinicalReasoner  # noqa: E402
from agentic_icu.api.dependencies import get_workflow  # noqa: E402
from agentic_icu.config import settings  # noqa: E402
from agentic_icu.domain.contracts import (  # noqa: E402
    EvaluatePatientRequest,
    ModelAgentResult,
    ObservationRecord,
    SignalQualityResult,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate alert policy behavior on local patient data."
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=100,
        help="Maximum number of local PSV files to scan.",
    )
    parser.add_argument(
        "--observation-rows",
        type=int,
        default=24,
        help="Rows per patient window to feed into runtime evaluation.",
    )
    parser.add_argument(
        "--profile",
        default="active",
        help="Policy profile to evaluate. Use 'active' for the runtime policy or a name from the policy profile catalog.",  # noqa: E501
    )
    parser.add_argument(
        "--compare-profiles",
        action="store_true",
        help="Evaluate all named policy profiles from the profile catalog in one run.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the full JSON evaluation output.",
    )
    parser.add_argument(
        "--output-markdown",
        help="Optional path to write a Markdown summary report.",
    )
    return parser.parse_args()


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return (numerator / denominator) if denominator else 0.0


def load_patient_window(
    path: Path, max_rows: int
) -> tuple[list[ObservationRecord], int]:
    records: list[ObservationRecord] = []
    ever_sepsis = 0

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for index, row in enumerate(reader):
            label = row.get("SepsisLabel")
            if label is not None and label != "":
                numeric_label = float(label)
                if math.isfinite(numeric_label) and numeric_label >= 1.0:
                    ever_sepsis = 1

            if index >= max_rows:
                continue

            values = {}
            for key, value in row.items():
                if key == "SepsisLabel" or value in (None, ""):
                    continue
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    values[key] = numeric_value

            if values:
                records.append(ObservationRecord(values=values))

    return records, ever_sepsis


def merge_policy(
    base_policy: AlertPolicy, overrides: Dict[str, Any]
) -> AlertPolicy:
    """Merge overrides into a policy dataclass."""
    data = base_policy.__dict__.copy()
    data.update(overrides)
    return AlertPolicy(**data)


def collect_cases(
    max_patients: int, observation_rows: int
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    get_workflow.cache_clear()
    workflow = get_workflow()
    raw_dir = Path(settings.raw_data_dir)
    cases: list[Dict[str, Any]] = []

    for path in sorted(raw_dir.glob("p*.psv"))[:max_patients]:
        records, ever_sepsis = load_patient_window(path, observation_rows)
        if not records:
            continue

        request = EvaluatePatientRequest(
            patient_id=path.stem, observation_window=records
        )
        response = workflow.evaluate(request)
        cases.append(
            {
                "patient_id": path.stem,
                "ever_sepsis": ever_sepsis,
                "signal_quality": response.signal_quality.model_dump(),
                "vitals_agent": response.vitals_agent.model_dump(),
                "lab_agent": response.lab_agent.model_dump(),
                "ensemble_score": response.ensemble_agent.score if response.ensemble_agent.status == "available" else None,
            }
        )

    return cases, workflow.reasoner.policy.__dict__


def decide_case(case: Dict[str, Any], policy: Any) -> tuple[bool, str, str]:
    if isinstance(policy, dict):
        p_obj = AlertPolicy.from_dict(policy)
    else:
        p_obj = policy
    reasoner = ClinicalReasoner(policy=p_obj)
    signal_quality = SignalQualityResult(**case["signal_quality"])
    vitals_result = ModelAgentResult(**case["vitals_agent"])
    lab_result = ModelAgentResult(**case["lab_agent"])
    decision, *_ = reasoner.decide(
        signal_quality, vitals_result, lab_result,
        ensemble_score=case.get("ensemble_score")
    )
    return decision.alert_triggered, decision.alert_type or "Unknown", decision.priority


def summarize_policy(
    policy_name: str, policy: Any, cases: list[Dict[str, Any]]
) -> Dict[str, Any]:
    if isinstance(policy, dict):
        p_obj = AlertPolicy.from_dict(policy)
    else:
        p_obj = policy

    alert_counter: Counter[str] = Counter()
    priority_counter: Counter[str] = Counter()
    total = len(cases)
    positives = sum(case["ever_sepsis"] for case in cases)
    negatives = total - positives
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    high_alerts = 0
    suppressed_artifacts = 0
    sample_alerts: list[Dict[str, Any]] = []

    for case in cases:
        triggered, alert_type, priority = decide_case(case, policy)
        alert_counter[alert_type] += 1
        priority_counter[priority] += 1

        if alert_type == p_obj.suppressed_artifact_alert_type:
            suppressed_artifacts += 1

        if triggered and priority == p_obj.high_alert_priority:
            high_alerts += 1

        if triggered:
            if case["ever_sepsis"]:
                true_positive += 1
            else:
                false_positive += 1

            if len(sample_alerts) < 10:
                sample_alerts.append(
                    {
                        "patient_id": case["patient_id"],
                        "sequence_score": case["vitals_agent"].get("score"),
                        "tabular_score": case["lab_agent"].get("score"),
                        "alert_type": alert_type,
                        "priority": priority,
                        "ever_sepsis": case["ever_sepsis"],
                    }
                )
        else:
            if case["ever_sepsis"]:
                false_negative += 1
            else:
                true_negative += 1

    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, positives)
    specificity = safe_ratio(true_negative, negatives)
    negative_predictive_value = safe_ratio(
        true_negative, true_negative + false_negative
    )
    false_positive_rate = safe_ratio(false_positive, negatives)
    false_negative_rate = safe_ratio(false_negative, positives)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "profile": policy_name,
        "patients_evaluated": total,
        "positive_patients": positives,
        "negative_patients": negatives,
        "positive_patient_rate": safe_ratio(positives, total),
        "any_alert_rate": safe_ratio(true_positive + false_positive, total),
        "high_alert_rate": safe_ratio(high_alerts, total),
        "suppressed_artifact_rate": safe_ratio(suppressed_artifacts, total),
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "negative_predictive_value": negative_predictive_value,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
        },
        "alert_type_counts": dict(alert_counter),
        "priority_counts": dict(priority_counter),
        "policy": policy,
        "sample_alerts": sample_alerts,
    }


def ensure_parent_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json_report(path_str: str, payload: Dict[str, Any]) -> None:
    path = ensure_parent_dir(path_str)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_profile_block(profile: Dict[str, Any]) -> str:
    metrics = profile["metrics"]
    confusion = profile["confusion_matrix"]
    return "\n".join(
        [
            f"## {profile['profile']}",
            "",
            f"- Patients evaluated: `{profile['patients_evaluated']}`",
            f"- Positive patients: `{profile['positive_patients']}`",
            f"- Any alert rate: `{profile['any_alert_rate']:.4f}`",
            f"- High alert rate: `{profile['high_alert_rate']:.4f}`",
            f"- True positive: `{confusion['true_positive']}`",
            f"- False positive: `{confusion['false_positive']}`",
            f"- True negative: `{confusion['true_negative']}`",
            f"- False negative: `{confusion['false_negative']}`",
            f"- Precision: `{metrics['precision']:.4f}`",
            f"- Recall: `{metrics['recall']:.4f}`",
            f"- Specificity: `{metrics['specificity']:.4f}`",
            f"- F1: `{metrics['f1']:.4f}`",
            f"- Balanced accuracy: `{metrics['balanced_accuracy']:.4f}`",
            "",
        ]
    )


def build_markdown_report(payload: Dict[str, Any]) -> str:
    if "profiles" in payload:
        lines = [
            "# Alert Policy Comparison Report",
            "",
            f"- Patients evaluated: `{payload['patients_evaluated']}`",
            f"- Observation rows: `{payload['observation_rows']}`",
            "",
        ]
        for profile in payload["profiles"]:
            lines.append(markdown_profile_block(profile))
        return "\n".join(lines).strip() + "\n"

    return (
        "\n".join(
            [
                "# Alert Policy Evaluation Report",
                "",
                markdown_profile_block(payload),
            ]
        ).strip()
        + "\n"
    )


def write_markdown_report(path_str: str, payload: Dict[str, Any]) -> None:
    path = ensure_parent_dir(path_str)
    path.write_text(build_markdown_report(payload), encoding="utf-8")


def maybe_write_reports(args: argparse.Namespace, payload: Dict[str, Any]) -> None:
    if args.output_json:
        write_json_report(args.output_json, payload)
    if args.output_markdown:
        write_markdown_report(args.output_markdown, payload)


def main() -> None:
    args = parse_args()
    cases, active_policy = collect_cases(args.max_patients, args.observation_rows)
    profiles = settings.load_alert_policy_profiles()

    if args.compare_profiles:
        active_policy_obj = AlertPolicy.from_dict(active_policy) if isinstance(active_policy, dict) else active_policy
        profile_summaries = [summarize_policy("active", active_policy_obj, cases)]
        for profile_name, overrides in profiles.items():
            merged_policy = merge_policy(active_policy_obj, overrides)
            profile_summaries.append(
                summarize_policy(profile_name, merged_policy, cases)
            )

        payload = {
            "patients_evaluated": len(cases),
            "observation_rows": args.observation_rows,
            "profiles": profile_summaries,
        }
        maybe_write_reports(args, payload)
        print(json.dumps(payload, indent=2))
        return

    if args.profile == "active":
        active_policy_obj = AlertPolicy.from_dict(active_policy) if isinstance(active_policy, dict) else active_policy
        payload = summarize_policy("active", active_policy_obj, cases)
        maybe_write_reports(args, payload)
        print(json.dumps(payload, indent=2))
        return

    if args.profile not in profiles:
        available_profiles = ", ".join(sorted(profiles))
        raise SystemExit(
            f"Unknown profile '{args.profile}'. Available profiles: active, {available_profiles}"
        )

    active_policy_obj = AlertPolicy.from_dict(active_policy) if isinstance(active_policy, dict) else active_policy
    selected_policy = merge_policy(active_policy_obj, profiles[args.profile])
    payload = summarize_policy(args.profile, selected_policy, cases)
    maybe_write_reports(args, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
