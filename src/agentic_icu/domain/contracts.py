from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Hard limits applied at the API boundary
MAX_WINDOW_ROWS = 168   # 7 days of hourly observations
MIN_WINDOW_ROWS = 1


class ObservationRecord(BaseModel):
    timestamp: Optional[str] = None
    values: Dict[str, float] = Field(default_factory=dict)


class EvaluatePatientRequest(BaseModel):
    patient_id: str
    observation_window: List[ObservationRecord] = Field(
        min_length=MIN_WINDOW_ROWS,
        max_length=MAX_WINDOW_ROWS,
    )

    @field_validator("patient_id")
    @classmethod
    def patient_id_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("patient_id must not be blank")
        return stripped


class AgentLogEntry(BaseModel):
    agent: str
    message: str


class SignalQualityResult(BaseModel):
    signal_valid: bool
    artifact_type: Optional[str] = None
    artifact_confidence: float = 0.0
    suppression_recommendation: bool = False
    artifact_affected_features: List[str] = Field(default_factory=list)
    suppression_mode: Literal["none", "partial", "full"] = "none"


class ModelAgentResult(BaseModel):
    status: Literal["available", "unavailable"]
    score: Optional[float] = None
    risk_band: Optional[str] = None
    detail: str
    decision_threshold: Optional[float] = None
    threshold_ratio: Optional[float] = None
    feature_contributions: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class ClinicalDecision(BaseModel):
    alert_triggered: bool
    alert_type: Optional[str] = None
    priority: Literal["low", "medium", "high"]
    rationale: str


class EvaluatePatientResponse(BaseModel):
    patient_id: str
    signal_quality: SignalQualityResult
    vitals_agent: ModelAgentResult
    lab_agent: ModelAgentResult
    resp_failure_agent: ModelAgentResult
    clinical_decision: ClinicalDecision
    reasoning_log: List[AgentLogEntry] = Field(default_factory=list)


class AgentExplanation(BaseModel):
    feature_contributions: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""
    status: Literal["available", "unavailable"]


class ExplainPatientResponse(BaseModel):
    patient_id: str
    lab_explanation: AgentExplanation
    vitals_explanation: AgentExplanation
