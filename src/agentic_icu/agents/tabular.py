from __future__ import annotations

import logging
from typing import Optional

from agentic_icu.domain.contracts import AgentLogEntry, ModelAgentResult
from agentic_icu.inference.explainer import TabularExplainer
from agentic_icu.inference.tabular import XGBoostInference
from agentic_icu.preprocessing.windowing import RuntimePreprocessor

logger = logging.getLogger(__name__)


class LabTabularAgent:
    def __init__(
        self,
        preprocessor: RuntimePreprocessor,
        predictor: XGBoostInference,
        explainer: Optional[TabularExplainer] = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.explainer = explainer

    def evaluate(self, records) -> tuple[ModelAgentResult, list[AgentLogEntry]]:
        if not self.preprocessor.available or not self.predictor.available:
            result = ModelAgentResult(
                status="unavailable",
                detail="Tabular model artifacts are not available yet.",
            )
            return result, [AgentLogEntry(agent="Lab Agent", message=result.detail)]

        features = self.preprocessor.build_tabular_features(records)

        # Check feature coverage — flag as degraded when >20% of model features are missing
        total_features = len(self.predictor.feature_columns)
        missing_count = sum(1 for col in self.predictor.feature_columns if col not in features or features[col] == 0.0)
        missing_fraction = missing_count / total_features if total_features > 0 else 0.0
        data_degraded = missing_fraction > 0.20

        score = self.predictor.predict(features)
        decision_threshold = self.predictor.decision_threshold
        threshold_ratio = (score / decision_threshold) if (decision_threshold is not None and decision_threshold > 0.0) else None
        moderate_ratio = 0.7

        if threshold_ratio is not None and threshold_ratio >= 1.0:
            risk_band = "high"
        elif threshold_ratio is not None and threshold_ratio >= moderate_ratio:
            risk_band = "moderate"
        else:
            risk_band = "low"

        if decision_threshold is not None and threshold_ratio is not None:
            detail = (
                f"Tabular model risk score is {score:.3f} against decision threshold {decision_threshold:.3f} "
                f"({threshold_ratio:.2f}x threshold)."
            )
        else:
            detail = f"Tabular model risk score is {score:.3f}."

        if data_degraded:
            detail += f" Data coverage degraded ({missing_fraction:.0%} of features missing — defaulted to 0)."
            logger.warning(
                "LabTabularAgent: %.0f%% of features missing (%d/%d) — model output may be unreliable.",
                missing_fraction * 100, missing_count, total_features,
            )

        # SHAP feature contributions — top-10 for a useful sidebar view
        feature_contributions: dict[str, float] = {}
        explanation = ""
        if self.explainer is not None:
            try:
                top_list, all_contributions = self.explainer.top_contributions(
                    features, n=10
                )
                feature_contributions = {
                    item["feature"]: item["shap_value"] for item in top_list
                }
                explanation = self.explainer.format_explanation(top_list[:3])
                if top_list:
                    top_label = top_list[0]["label"]
                    arrow = "↑" if top_list[0]["direction"] == "increases_risk" else "↓"
                    detail += f" Primary driver: {top_label} ({arrow})."
            except Exception as exc:
                logger.warning(
                    "LabTabularAgent: SHAP computation failed — %s: %s",
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
        return result, [AgentLogEntry(agent="Lab Agent", message=result.detail)]
