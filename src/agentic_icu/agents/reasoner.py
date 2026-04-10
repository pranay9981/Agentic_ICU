from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from agentic_icu.domain.contracts import (
    AgentLogEntry,
    ClinicalDecision,
    ModelAgentResult,
    SignalQualityResult,
)


@dataclass(frozen=True)
class AlertPolicy:
    high_alert_max_score_threshold: float | None = None
    high_alert_mean_score_threshold: float | None = None
    high_alert_extreme_sequence_score_threshold: float | None = 0.88
    high_alert_supported_sequence_score_threshold: float | None = 0.8
    high_alert_tabular_support_score_threshold: float | None = 0.25
    medium_alert_max_score_threshold: float | None = None
    medium_alert_sequence_score_threshold: float | None = 0.55
    medium_alert_tabular_score_threshold: float | None = 0.4
    suppressed_artifact_alert_type: str = "Suppressed Artifact"
    models_unavailable_alert_type: str = "Models Unavailable"
    high_alert_type: str = "Sepsis Early Warning"
    medium_alert_type: str = "Deterioration Watch"
    stable_alert_type: str = "Stable"
    suppressed_artifact_priority: str = "low"
    models_unavailable_priority: str = "low"
    high_alert_priority: str = "high"
    medium_alert_priority: str = "medium"
    stable_priority: str = "low"
    suppressed_artifact_rationale: str = "Signal Quality Agent suppressed the event before clinical escalation."
    models_unavailable_rationale: str = "No trained runtime artifacts are loaded yet, so no predictive alert is issued."
    high_alert_rationale: str = "At least one specialized agent reports high deterioration risk on validated signals."
    medium_alert_rationale: str = "Predictive risk is elevated and should be monitored closely."
    stable_rationale: str = "Available agent scores remain below the current alert thresholds."
    resp_high_alert_threshold: float | None = 0.8
    resp_medium_alert_threshold: float | None = 0.55
    resp_high_alert_type: str = "Respiratory Failure Risk"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AlertPolicy":
        return cls(**payload)


class ClinicalReasoner:
    def __init__(self, policy: AlertPolicy | None = None) -> None:
        self.policy = policy or AlertPolicy()

    def _high_alert_triggered(
        self,
        vitals_result: ModelAgentResult,
        lab_result: ModelAgentResult,
        max_score: float,
        mean_score: float,
    ) -> tuple[bool, str]:
        triggers: list[str] = []

        extreme_sequence = self.policy.high_alert_extreme_sequence_score_threshold
        if (
            extreme_sequence is not None
            and vitals_result.score is not None
            and vitals_result.score >= extreme_sequence
        ):
            triggers.append(f"sequence>={extreme_sequence:.2f}")

        supported_sequence = self.policy.high_alert_supported_sequence_score_threshold
        tabular_support = self.policy.high_alert_tabular_support_score_threshold
        if (
            supported_sequence is not None
            and tabular_support is not None
            and vitals_result.score is not None
            and lab_result.score is not None
            and vitals_result.score >= supported_sequence
            and lab_result.score >= tabular_support
        ):
            triggers.append(f"sequence>={supported_sequence:.2f} with tabular>={tabular_support:.2f}")

        if triggers:
            return True, ", ".join(triggers)

        if self.policy.high_alert_max_score_threshold is not None and max_score >= self.policy.high_alert_max_score_threshold:
            return True, f"max_score>={self.policy.high_alert_max_score_threshold:.2f}"

        if self.policy.high_alert_mean_score_threshold is not None and mean_score >= self.policy.high_alert_mean_score_threshold:
            return True, f"mean_score>={self.policy.high_alert_mean_score_threshold:.2f}"

        return False, ""

    def _medium_alert_triggered(
        self,
        vitals_result: ModelAgentResult,
        lab_result: ModelAgentResult,
        max_score: float,
    ) -> tuple[bool, str]:
        triggers: list[str] = []

        sequence_threshold = self.policy.medium_alert_sequence_score_threshold
        if (
            sequence_threshold is not None
            and vitals_result.score is not None
            and vitals_result.score >= sequence_threshold
        ):
            triggers.append(f"sequence>={sequence_threshold:.2f}")

        tabular_threshold = self.policy.medium_alert_tabular_score_threshold
        if (
            tabular_threshold is not None
            and lab_result.score is not None
            and lab_result.score >= tabular_threshold
        ):
            triggers.append(f"tabular>={tabular_threshold:.2f}")

        if triggers:
            return True, ", ".join(triggers)

        if self.policy.medium_alert_max_score_threshold is not None and max_score >= self.policy.medium_alert_max_score_threshold:
            return True, f"max_score>={self.policy.medium_alert_max_score_threshold:.2f}"

        return False, ""

    PARTIAL_SUPPRESSION_FACTOR = 0.7  # score multiplier for partial-suppression mode

    def decide(
        self,
        signal_quality: SignalQualityResult,
        vitals_result: ModelAgentResult,
        lab_result: ModelAgentResult,
        resp_result: ModelAgentResult | None = None,
    ) -> tuple[ClinicalDecision, list[AgentLogEntry]]:
        logs: list[AgentLogEntry] = []

        # Full suppression — hard block, no alert
        if not signal_quality.signal_valid and signal_quality.suppression_recommendation:
            decision = ClinicalDecision(
                alert_triggered=False,
                alert_type=self.policy.suppressed_artifact_alert_type,
                priority=self.policy.suppressed_artifact_priority,
                rationale=self.policy.suppressed_artifact_rationale,
            )
            logs.append(AgentLogEntry(agent="Clinical Reasoner", message=decision.rationale))
            return decision, logs

        # Partial suppression — signal valid but suspect; apply score penalty
        if signal_quality.suppression_mode == "partial" and signal_quality.suppression_recommendation:
            affected = signal_quality.artifact_affected_features
            f = self.PARTIAL_SUPPRESSION_FACTOR
            vitals_score_adj = vitals_result.score * f if vitals_result.score is not None else None
            lab_score_adj = lab_result.score * f if lab_result.score is not None else None
            # Build adjusted result copies (shallow override via model_copy for Pydantic v2)
            vitals_result = vitals_result.model_copy(update={"score": vitals_score_adj})
            lab_result = lab_result.model_copy(update={"score": lab_score_adj})
            logs.append(AgentLogEntry(
                agent="Clinical Reasoner",
                message=(
                    f"Partial suppression applied ({f:.0%} score penalty) — "
                    f"signal suspect ({signal_quality.artifact_type}) "
                    f"affecting {', '.join(affected) if affected else 'unspecified features'}."
                ),
            ))

        available_scores = [score for score in (vitals_result.score, lab_result.score) if score is not None]

        if not available_scores:
            decision = ClinicalDecision(
                alert_triggered=False,
                alert_type=self.policy.models_unavailable_alert_type,
                priority=self.policy.models_unavailable_priority,
                rationale=self.policy.models_unavailable_rationale,
            )
            logs.append(AgentLogEntry(agent="Clinical Reasoner", message=decision.rationale))
            return decision, logs

        max_score = max(available_scores)
        mean_score = sum(available_scores) / len(available_scores)
        high_triggered, high_basis = self._high_alert_triggered(vitals_result, lab_result, max_score, mean_score)
        medium_triggered, medium_basis = self._medium_alert_triggered(vitals_result, lab_result, max_score)

        rationale = self._compose_rationale(
            high_triggered=high_triggered,
            medium_triggered=medium_triggered,
            high_basis=high_basis,
            medium_basis=medium_basis,
            vitals_result=vitals_result,
            lab_result=lab_result,
        )

        # --- Resp failure override ---
        resp_score = resp_result.score if resp_result is not None else None
        resp_high = (
            self.policy.resp_high_alert_threshold is not None
            and resp_score is not None
            and resp_score >= self.policy.resp_high_alert_threshold
        )
        resp_medium = (
            not resp_high
            and self.policy.resp_medium_alert_threshold is not None
            and resp_score is not None
            and resp_score >= self.policy.resp_medium_alert_threshold
        )

        if resp_high and not high_triggered:
            # Resp is critical but sepsis is not — issue respiratory failure alert
            resp_rationale = (
                f"{rationale} Respiratory failure risk is elevated (resp score {resp_score:.3f})."
                if rationale
                else f"Respiratory failure risk is elevated (resp score {resp_score:.3f})."
            )
            decision = ClinicalDecision(
                alert_triggered=True,
                alert_type=self.policy.resp_high_alert_type,
                priority=self.policy.high_alert_priority,
                rationale=resp_rationale,
            )
        elif high_triggered:
            # Sepsis high + resp high: co-alert note in rationale
            extra = f" Respiratory compromise also detected (resp score {resp_score:.3f})." if resp_high else ""
            decision = ClinicalDecision(
                alert_triggered=True,
                alert_type=self.policy.high_alert_type,
                priority=self.policy.high_alert_priority,
                rationale=rationale + extra,
            )
        elif medium_triggered:
            extra = f" Respiratory failure risk also elevated (resp score {resp_score:.3f})." if resp_high or resp_medium else ""
            decision = ClinicalDecision(
                alert_triggered=True,
                alert_type=self.policy.medium_alert_type,
                priority=self.policy.medium_alert_priority,
                rationale=rationale + extra,
            )
        elif resp_medium:
            # Resp elevated but not critical, sepsis stable — bump to medium watch
            decision = ClinicalDecision(
                alert_triggered=True,
                alert_type=self.policy.medium_alert_type,
                priority=self.policy.medium_alert_priority,
                rationale=f"Respiratory failure risk is elevated (resp score {resp_score:.3f}). {rationale}".strip(),
            )
        else:
            decision = ClinicalDecision(
                alert_triggered=False,
                alert_type=self.policy.stable_alert_type,
                priority=self.policy.stable_priority,
                rationale=rationale,
            )

        score_message = (
            f"Fusion scores: sequence={vitals_result.score if vitals_result.score is not None else 'n/a'}, "
            f"tabular={lab_result.score if lab_result.score is not None else 'n/a'}, "
            f"resp={resp_score if resp_score is not None else 'n/a'}, "
            f"max={max_score:.3f}, mean={mean_score:.3f}."
        )
        logs.append(AgentLogEntry(agent="Clinical Reasoner", message=score_message))
        if high_basis:
            logs.append(AgentLogEntry(agent="Clinical Reasoner", message=f"High-alert basis: {high_basis}."))
        if medium_basis:
            logs.append(AgentLogEntry(agent="Clinical Reasoner", message=f"Medium-alert basis: {medium_basis}."))
        if resp_high:
            logs.append(AgentLogEntry(agent="Clinical Reasoner", message=f"Resp high-alert: resp score {resp_score:.3f} >= {self.policy.resp_high_alert_threshold}."))
        elif resp_medium:
            logs.append(AgentLogEntry(agent="Clinical Reasoner", message=f"Resp medium-alert: resp score {resp_score:.3f} >= {self.policy.resp_medium_alert_threshold}."))
        logs.append(AgentLogEntry(agent="Clinical Reasoner", message=decision.rationale))
        return decision, logs

    def _compose_rationale(
        self,
        high_triggered: bool,
        medium_triggered: bool,
        high_basis: str,
        medium_basis: str,
        vitals_result: ModelAgentResult,
        lab_result: ModelAgentResult,
    ) -> str:
        """Compose a plain-language rationale from agent explanations.

        If agents have produced SHAP / saliency explanations, the rationale
        describes *what* drove the alert.  Falls back to static policy strings
        when no explanation data is available.
        """
        parts: list[str] = []

        # --- Tabular (SHAP) explanation ---
        if lab_result.explanation:
            parts.append(lab_result.explanation)

        # --- Sequence (temporal saliency) explanation ---
        if vitals_result.explanation:
            parts.append(vitals_result.explanation)

        # --- Alert context ---
        if high_triggered:
            if not parts:
                return self.policy.high_alert_rationale
            context = f"High deterioration risk triggered ({high_basis})."
            parts.append(context)
        elif medium_triggered:
            if not parts:
                return self.policy.medium_alert_rationale
            context = f"Elevated risk — monitoring recommended ({medium_basis})."
            parts.append(context)
        else:
            if not parts:
                return self.policy.stable_rationale
            parts.append("All scores below alert thresholds.")

        return " ".join(parts)
