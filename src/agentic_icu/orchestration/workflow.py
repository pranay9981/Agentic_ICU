from __future__ import annotations

import logging
from typing import Optional

from agentic_icu.agents.reasoner import ClinicalReasoner
from agentic_icu.agents.resp_failure import RespFailureAgent
from agentic_icu.agents.signal_quality import SignalQualityAgent
from agentic_icu.agents.tabular import LabTabularAgent
from agentic_icu.agents.vitals import VitalsAgent
from agentic_icu.domain.contracts import EvaluatePatientRequest, EvaluatePatientResponse, ModelAgentResult
from agentic_icu.inference.ensemble import EnsembleInference

logger = logging.getLogger(__name__)


class AgenticICUWorkflow:
    def __init__(
        self,
        signal_quality_agent: SignalQualityAgent,
        vitals_agent: VitalsAgent,
        lab_agent: LabTabularAgent,
        reasoner: ClinicalReasoner,
        resp_failure_agent: Optional[RespFailureAgent] = None,
        ensemble: Optional[EnsembleInference] = None,
    ) -> None:
        self.signal_quality_agent = signal_quality_agent
        self.vitals_agent = vitals_agent
        self.lab_agent = lab_agent
        self.reasoner = reasoner
        self.resp_failure_agent = resp_failure_agent
        self.ensemble = ensemble

    def evaluate(self, request: EvaluatePatientRequest) -> EvaluatePatientResponse:
        window_values = [record.values for record in request.observation_window]

        # Signal quality runs first — on full suppression we short-circuit to
        # avoid wasting GRU/XGBoost compute and returning misleading SHAP output
        # for a window the reasoner will mark as Suppressed Artifact.
        signal_quality, signal_logs = self.signal_quality_agent.evaluate(window_values)

        if not signal_quality.signal_valid and signal_quality.suppression_recommendation:
            # Full suppression — skip model inference entirely
            suppressed_result = ModelAgentResult(
                status="unavailable",
                detail="Inference skipped: signal fully suppressed by Signal Quality Agent.",
            )
            resp_result = ModelAgentResult(
                status="unavailable",
                detail="Inference skipped: signal fully suppressed.",
            )
            clinical_decision, reasoner_logs, _, _, _ = self.reasoner.decide(
                signal_quality, suppressed_result, suppressed_result, None
            )
            return EvaluatePatientResponse(
                patient_id=request.patient_id,
                signal_quality=signal_quality,
                vitals_agent=suppressed_result,
                lab_agent=suppressed_result,
                resp_failure_agent=resp_result,
                clinical_decision=clinical_decision,
                reasoning_log=signal_logs + reasoner_logs,
            )

        vitals_result, vitals_logs = self.vitals_agent.evaluate(request.observation_window)
        lab_result, lab_logs = self.lab_agent.evaluate(request.observation_window)

        if self.resp_failure_agent is not None:
            resp_result, resp_logs = self.resp_failure_agent.evaluate(request.observation_window)
        else:
            resp_result = ModelAgentResult(status="unavailable", detail="Respiratory failure agent not configured.")
            resp_logs = []

        # Ensemble meta-learner: fuse GRU + XGBoost calibrated scores
        ensemble_score: float | None = None
        ensemble_result = ModelAgentResult(status="unavailable", detail="Ensemble meta-learner not configured.")
        if (
            self.ensemble is not None
            and self.ensemble.available
            and vitals_result.score is not None
            and lab_result.score is not None
        ):
            try:
                ensemble_score = self.ensemble.predict(vitals_result.score, lab_result.score)
                high_thresh = self.reasoner.policy.high_alert_ensemble_score_threshold or 0.80
                med_thresh = self.reasoner.policy.medium_alert_ensemble_score_threshold or 0.55
                ensemble_result = ModelAgentResult(
                    status="available",
                    score=ensemble_score,
                    risk_band=(
                        "high" if ensemble_score >= high_thresh
                        else ("moderate" if ensemble_score >= med_thresh else "low")
                    ),
                    detail=f"Ensemble (GRU+XGB) risk score is {ensemble_score:.3f}.",
                    decision_threshold=high_thresh,
                    threshold_ratio=round(ensemble_score / high_thresh, 3),
                )
            except Exception as exc:
                logger.warning("EnsembleInference.predict failed — %s: %s", type(exc).__name__, exc)

        clinical_decision, reasoner_logs, vitals_result, lab_result, resp_result = self.reasoner.decide(
            signal_quality, vitals_result, lab_result, resp_result, ensemble_score
        )

        return EvaluatePatientResponse(
            patient_id=request.patient_id,
            signal_quality=signal_quality,
            vitals_agent=vitals_result,
            lab_agent=lab_result,
            resp_failure_agent=resp_result,
            ensemble_agent=ensemble_result,
            clinical_decision=clinical_decision,
            reasoning_log=signal_logs + vitals_logs + lab_logs + resp_logs + reasoner_logs,
        )
