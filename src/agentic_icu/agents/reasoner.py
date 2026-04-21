from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Literal

from agentic_icu.domain.contracts import (
    AgentLogEntry,
    ClinicalDecision,
    ModelAgentResult,
    SignalQualityResult,
)

_PROBABILITY_FIELDS = frozenset(
    {
        "high_alert_extreme_sequence_score_threshold",
        "high_alert_supported_sequence_score_threshold",
        "high_alert_tabular_support_score_threshold",
        "high_alert_max_score_threshold",
        "high_alert_mean_score_threshold",
        "medium_alert_sequence_score_threshold",
        "medium_alert_tabular_score_threshold",
        "medium_alert_max_score_threshold",
        "resp_high_alert_threshold",
        "resp_medium_alert_threshold",
        "high_alert_ensemble_score_threshold",
        "medium_alert_ensemble_score_threshold",
    }
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
    suppressed_artifact_priority: Literal["low", "medium", "high"] = "low"
    models_unavailable_priority: Literal["low", "medium", "high"] = "low"
    high_alert_priority: Literal["low", "medium", "high"] = "high"
    medium_alert_priority: Literal["low", "medium", "high"] = "medium"
    stable_priority: Literal["low", "medium", "high"] = "low"
    suppressed_artifact_rationale: str = (
        "Signal Quality Agent suppressed the event before clinical escalation."
    )
    models_unavailable_rationale: str = (
        "No trained runtime artifacts are loaded yet, so no predictive alert is issued."
    )
    high_alert_rationale: str = (
        "At least one specialized agent reports high deterioration risk on validated signals."
    )
    medium_alert_rationale: str = (
        "Predictive risk is elevated and should be monitored closely."
    )
    stable_rationale: str = (
        "Available agent scores remain below the current alert thresholds."
    )
    resp_high_alert_threshold: float | None = 0.8
    resp_medium_alert_threshold: float | None = 0.55
    resp_high_alert_type: str = "Respiratory Failure Risk"
    partial_suppression_factor: float = 0.7
    high_alert_ensemble_score_threshold: float | None = 0.80
    medium_alert_ensemble_score_threshold: float | None = 0.55

    def __post_init__(self) -> None:
        for field_name in _PROBABILITY_FIELDS:
            val = getattr(self, field_name)
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"AlertPolicy.{field_name} must be in [0, 1], got {val!r}"
                )
        if not (0.0 <= self.partial_suppression_factor <= 1.0):
            raise ValueError(
                f"AlertPolicy.partial_suppression_factor must be in [0, 1], "
                f"got {self.partial_suppression_factor!r}"
            )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AlertPolicy":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in valid_keys})


class ClinicalReasoner:
    def __init__(self, policy: AlertPolicy | None = None) -> None:
        self.policy = policy or AlertPolicy()

    def _high_alert_triggered(
        self,
        vitals_result: ModelAgentResult,
        lab_result: ModelAgentResult,
        max_score: float,
        mean_score: float,
        ensemble_score: float | None = None,
    ) -> tuple[bool, str]:
        triggers: list[str] = []

        ensemble_threshold = self.policy.high_alert_ensemble_score_threshold
        if (
            ensemble_score is not None
            and ensemble_threshold is not None
            and ensemble_score >= ensemble_threshold
        ):
            triggers.append(f"ensemble>={ensemble_threshold:.2f}")

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
            triggers.append(
                f"sequence>={supported_sequence:.2f} with tabular>={tabular_support:.2f}"
            )

        if triggers:
            return True, ", ".join(triggers)

        if (
            self.policy.high_alert_max_score_threshold is not None
            and max_score >= self.policy.high_alert_max_score_threshold
        ):
            return True, f"max_score>={self.policy.high_alert_max_score_threshold:.2f}"

        if (
            self.policy.high_alert_mean_score_threshold is not None
            and mean_score >= self.policy.high_alert_mean_score_threshold
        ):
            return (
                True,
                f"mean_score>={self.policy.high_alert_mean_score_threshold:.2f}",
            )

        return False, ""

    def _medium_alert_triggered(
        self,
        vitals_result: ModelAgentResult,
        lab_result: ModelAgentResult,
        max_score: float,
        ensemble_score: float | None = None,
    ) -> tuple[bool, str]:
        triggers: list[str] = []

        ensemble_threshold = self.policy.medium_alert_ensemble_score_threshold
        if (
            ensemble_score is not None
            and ensemble_threshold is not None
            and ensemble_score >= ensemble_threshold
        ):
            triggers.append(f"ensemble>={ensemble_threshold:.2f}")

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

        if (
            self.policy.medium_alert_max_score_threshold is not None
            and max_score >= self.policy.medium_alert_max_score_threshold
        ):
            return (
                True,
                f"max_score>={self.policy.medium_alert_max_score_threshold:.2f}",
            )

        return False, ""

    @staticmethod
    def _recompute_agent_result(
        result: ModelAgentResult, new_score: float | None
    ) -> ModelAgentResult:
        """Return a copy of result with score, risk_band, threshold_ratio, and detail all consistent."""
        if new_score is None:
            return result.model_copy(
                update={"score": None, "risk_band": "low", "threshold_ratio": None}
            )

        dt = result.decision_threshold
        new_ratio = (new_score / dt) if (dt is not None and dt > 0.0) else None
        moderate_ratio = 0.7
        if new_ratio is not None and new_ratio >= 1.0:
            new_band = "high"
        elif new_ratio is not None and new_ratio >= moderate_ratio:
            new_band = "moderate"
        else:
            new_band = "low"

        if dt is not None and new_ratio is not None:
            new_detail = f"Score suppression-adjusted to {new_score:.3f} ({new_ratio:.2f}x threshold)."
        else:
            new_detail = f"Score suppression-adjusted to {new_score:.3f}."

        return result.model_copy(
            update={
                "score": new_score,
                "risk_band": new_band,
                "threshold_ratio": new_ratio,
                "detail": new_detail,
            }
        )

    def decide(
        self,
        signal_quality: SignalQualityResult,
        vitals_result: ModelAgentResult,
        lab_result: ModelAgentResult,
        resp_result: ModelAgentResult | None = None,
        ensemble_score: float | None = None,
    ) -> tuple[
        ClinicalDecision,
        list[AgentLogEntry],
        ModelAgentResult,
        ModelAgentResult,
        ModelAgentResult | None,
    ]:
        logs: list[AgentLogEntry] = []

        # Full suppression — hard block, no alert
        if (
            not signal_quality.signal_valid
            and signal_quality.suppression_recommendation
        ):
            decision = ClinicalDecision(
                alert_triggered=False,
                alert_type=self.policy.suppressed_artifact_alert_type,
                priority=self.policy.suppressed_artifact_priority,
                rationale=self.policy.suppressed_artifact_rationale,
            )
            logs.append(
                AgentLogEntry(agent="Clinical Reasoner", message=decision.rationale)
            )
            return decision, logs, vitals_result, lab_result, resp_result

        # Partial suppression — signal valid but suspect; apply score penalty to all
        # agent results so that score, risk_band, threshold_ratio, and detail remain
        # internally consistent in the response payload.
        if (
            signal_quality.suppression_mode == "partial"
            and signal_quality.suppression_recommendation
        ):
            affected = signal_quality.artifact_affected_features
            f = self.policy.partial_suppression_factor
            vitals_score_adj = (
                vitals_result.score * f if vitals_result.score is not None else None
            )
            lab_score_adj = (
                lab_result.score * f if lab_result.score is not None else None
            )
            vitals_result = self._recompute_agent_result(
                vitals_result, vitals_score_adj
            )
            lab_result = self._recompute_agent_result(lab_result, lab_score_adj)
            if resp_result is not None and resp_result.score is not None:
                resp_result = self._recompute_agent_result(
                    resp_result, resp_result.score * f
                )
            logs.append(
                AgentLogEntry(
                    agent="Clinical Reasoner",
                    message=(
                        f"Partial suppression applied ({f:.0%} score penalty) — "
                        f"signal suspect ({signal_quality.artifact_type}) "
                        f"affecting {', '.join(affected) if affected else 'unspecified features'}."
                    ),
                )
            )

        # Partial suppression also penalises ensemble score
        if (
            signal_quality.suppression_mode == "partial"
            and signal_quality.suppression_recommendation
        ):
            if ensemble_score is not None:
                ensemble_score = ensemble_score * self.policy.partial_suppression_factor

        available_scores = [
            score
            for score in (vitals_result.score, lab_result.score)
            if score is not None
        ]

        if not available_scores:
            # Sepsis models have no score — check ensemble, then resp, before giving up.
            if ensemble_score is not None:
                if (
                    self.policy.high_alert_ensemble_score_threshold is not None
                    and ensemble_score
                    >= self.policy.high_alert_ensemble_score_threshold
                ):
                    decision = ClinicalDecision(
                        alert_triggered=True,
                        alert_type=self.policy.high_alert_type,
                        priority=self.policy.high_alert_priority,
                        rationale=(
                            f"Ensemble risk elevated (score {ensemble_score:.3f} >= "
                            f"{self.policy.high_alert_ensemble_score_threshold:.2f}). "
                            "Sepsis models unavailable."
                        ),
                    )
                    logs.append(
                        AgentLogEntry(
                            agent="Clinical Reasoner", message=decision.rationale
                        )
                    )
                    return decision, logs, vitals_result, lab_result, resp_result
                if (
                    self.policy.medium_alert_ensemble_score_threshold is not None
                    and ensemble_score
                    >= self.policy.medium_alert_ensemble_score_threshold
                ):
                    decision = ClinicalDecision(
                        alert_triggered=True,
                        alert_type=self.policy.medium_alert_type,
                        priority=self.policy.medium_alert_priority,
                        rationale=(
                            f"Ensemble risk elevated (score {ensemble_score:.3f}). "
                            "Sepsis models unavailable."
                        ),
                    )
                    logs.append(
                        AgentLogEntry(
                            agent="Clinical Reasoner", message=decision.rationale
                        )
                    )
                    return decision, logs, vitals_result, lab_result, resp_result

            resp_score_lone = resp_result.score if resp_result is not None else None
            if resp_score_lone is not None:
                if (
                    self.policy.resp_high_alert_threshold is not None
                    and resp_score_lone >= self.policy.resp_high_alert_threshold
                ):
                    decision = ClinicalDecision(
                        alert_triggered=True,
                        alert_type=self.policy.resp_high_alert_type,
                        priority=self.policy.high_alert_priority,
                        rationale=(
                            f"Respiratory failure risk is elevated (resp score {resp_score_lone:.3f}). "
                            "Sepsis models unavailable."
                        ),
                    )
                    logs.append(
                        AgentLogEntry(
                            agent="Clinical Reasoner", message=decision.rationale
                        )
                    )
                    return decision, logs, vitals_result, lab_result, resp_result
                if (
                    self.policy.resp_medium_alert_threshold is not None
                    and resp_score_lone >= self.policy.resp_medium_alert_threshold
                ):
                    decision = ClinicalDecision(
                        alert_triggered=True,
                        alert_type=self.policy.medium_alert_type,
                        priority=self.policy.medium_alert_priority,
                        rationale=(
                            f"Respiratory failure risk is elevated (resp score {resp_score_lone:.3f}). "
                            "Sepsis models unavailable."
                        ),
                    )
                    logs.append(
                        AgentLogEntry(
                            agent="Clinical Reasoner", message=decision.rationale
                        )
                    )
                    return decision, logs, vitals_result, lab_result, resp_result
            decision = ClinicalDecision(
                alert_triggered=False,
                alert_type=self.policy.models_unavailable_alert_type,
                priority=self.policy.models_unavailable_priority,
                rationale=self.policy.models_unavailable_rationale,
            )
            logs.append(
                AgentLogEntry(agent="Clinical Reasoner", message=decision.rationale)
            )
            return decision, logs, vitals_result, lab_result, resp_result

        max_score = max(available_scores)
        mean_score = sum(available_scores) / len(available_scores)
        high_triggered, high_basis = self._high_alert_triggered(
            vitals_result, lab_result, max_score, mean_score, ensemble_score
        )
        medium_triggered, medium_basis = self._medium_alert_triggered(
            vitals_result, lab_result, max_score, ensemble_score
        )

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
            extra = (
                f" Respiratory compromise also detected (resp score {resp_score:.3f})."
                if resp_high
                else ""
            )
            decision = ClinicalDecision(
                alert_triggered=True,
                alert_type=self.policy.high_alert_type,
                priority=self.policy.high_alert_priority,
                rationale=rationale + extra,
            )
        elif medium_triggered:
            extra = (
                f" Respiratory failure risk also elevated (resp score {resp_score:.3f})."
                if resp_high or resp_medium
                else ""
            )
            decision = ClinicalDecision(
                alert_triggered=True,
                alert_type=self.policy.medium_alert_type,
                priority=self.policy.medium_alert_priority,
                rationale=rationale + extra,
            )
        elif resp_medium:
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
            f"ensemble={f'{ensemble_score:.3f}' if ensemble_score is not None else 'n/a'}, "
            f"resp={resp_score if resp_score is not None else 'n/a'}, "
            f"max={max_score:.3f}, mean={mean_score:.3f}."
        )
        logs.append(AgentLogEntry(agent="Clinical Reasoner", message=score_message))
        if high_basis:
            logs.append(
                AgentLogEntry(
                    agent="Clinical Reasoner",
                    message=f"High-alert basis: {high_basis}.",
                )
            )
        if medium_basis:
            logs.append(
                AgentLogEntry(
                    agent="Clinical Reasoner",
                    message=f"Medium-alert basis: {medium_basis}.",
                )
            )
        if resp_high:
            logs.append(
                AgentLogEntry(
                    agent="Clinical Reasoner",
                    message=f"Resp high-alert: resp score {resp_score:.3f} >= {self.policy.resp_high_alert_threshold}.",
                )
            )
        elif resp_medium:
            logs.append(
                AgentLogEntry(
                    agent="Clinical Reasoner",
                    message=f"Resp medium-alert: resp score {resp_score:.3f} >= {self.policy.resp_medium_alert_threshold}.",  # noqa: E501
                )
            )
        logs.append(
            AgentLogEntry(agent="Clinical Reasoner", message=decision.rationale)
        )
        return decision, logs, vitals_result, lab_result, resp_result

    def _compose_rationale(
        self,
        high_triggered: bool,
        medium_triggered: bool,
        high_basis: str,
        medium_basis: str,
        vitals_result: ModelAgentResult,
        lab_result: ModelAgentResult,
    ) -> str:
        parts: list[str] = []

        if lab_result.explanation:
            parts.append(lab_result.explanation)

        if vitals_result.explanation:
            parts.append(vitals_result.explanation)

        if high_triggered:
            if not parts:
                return self.policy.high_alert_rationale
            parts.append(f"High deterioration risk triggered ({high_basis}).")
        elif medium_triggered:
            if not parts:
                return self.policy.medium_alert_rationale
            parts.append(f"Elevated risk — monitoring recommended ({medium_basis}).")
        else:
            if not parts:
                return self.policy.stable_rationale
            parts.append("All scores below alert thresholds.")

        return " ".join(parts)
