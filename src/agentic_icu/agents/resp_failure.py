from __future__ import annotations

import logging
from typing import Optional

from agentic_icu.domain.contracts import AgentLogEntry, ModelAgentResult
from agentic_icu.inference.sequence import SequenceInference
from agentic_icu.inference.tabular import XGBoostInference
from agentic_icu.preprocessing.windowing import RuntimePreprocessor

logger = logging.getLogger(__name__)


class RespFailureAgent:
    """Respiratory failure risk agent.

    Primary score: resp GRU (SequenceInference — test_auc=0.96, test_ap=0.92).
    Secondary: resp XGBoost (context only — test_auc=0.69, test_ap=0.012).
    """

    def __init__(
        self,
        preprocessor: RuntimePreprocessor,
        gru_predictor: SequenceInference,
        xgb_predictor: Optional[XGBoostInference] = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.gru_predictor = gru_predictor
        self.xgb_predictor = xgb_predictor

    def evaluate(self, records) -> tuple[ModelAgentResult, list[AgentLogEntry]]:
        if not self.preprocessor.available or not self.gru_predictor.available:
            result = ModelAgentResult(
                status="unavailable",
                detail="Respiratory failure model artifacts are not available yet.",
            )
            return result, [
                AgentLogEntry(agent="Resp Failure Agent", message=result.detail)
            ]

        sequence_tensor = self.preprocessor.build_sequence_tensor(records)
        score = self.gru_predictor.predict(sequence_tensor)
        decision_threshold = self.gru_predictor.decision_threshold
        threshold_ratio = (score / decision_threshold) if decision_threshold else None
        moderate_ratio = 0.7

        if threshold_ratio is not None and threshold_ratio >= 1.0:
            risk_band = "high"
        elif threshold_ratio is not None and threshold_ratio >= moderate_ratio:
            risk_band = "moderate"
        else:
            risk_band = "low"

        if decision_threshold is not None and threshold_ratio is not None:
            detail = (
                f"Resp GRU risk score is {score:.3f} against decision threshold {decision_threshold:.3f} "
                f"({threshold_ratio:.2f}x threshold)."
            )
        else:
            detail = f"Resp GRU risk score is {score:.3f}."

        feature_contributions: dict[str, float] = {}
        explanation = ""
        try:
            weights = self.gru_predictor.temporal_saliency(sequence_tensor)
            n_steps = len(weights)
            feature_contributions = {
                f"t_{i + 1:02d}": float(w) for i, w in enumerate(weights)
            }
            top_steps = sorted(range(n_steps), key=lambda i: weights[i], reverse=True)[
                :3
            ]
            top_hours = sorted(h + 1 for h in top_steps)
            if len(top_hours) == 1:
                focus_str = f"hour {top_hours[0]}"
            elif (top_hours[-1] - top_hours[0]) == (len(top_hours) - 1):
                focus_str = f"hours {top_hours[0]}-{top_hours[-1]}"
            else:
                focus_str = "hours " + ", ".join(str(h) for h in top_hours)
            explanation = f"Resp model focused on observation {focus_str} of the {n_steps}h window."
        except Exception as exc:
            logger.warning(
                "RespFailureAgent: saliency computation failed — %s: %s",
                type(exc).__name__,
                exc,
            )

        result = ModelAgentResult(
            status="available",
            score=score,
            risk_band=risk_band,
            detail=detail,
            decision_threshold=decision_threshold,
            threshold_ratio=threshold_ratio,
            feature_contributions=feature_contributions,
            explanation=explanation,
        )
        return result, [
            AgentLogEntry(agent="Resp Failure Agent", message=result.detail)
        ]
