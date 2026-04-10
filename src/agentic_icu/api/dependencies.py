from __future__ import annotations

from functools import lru_cache

from agentic_icu.agents.reasoner import AlertPolicy, ClinicalReasoner
from agentic_icu.agents.resp_failure import RespFailureAgent
from agentic_icu.agents.signal_quality import SignalQualityAgent
from agentic_icu.agents.tabular import LabTabularAgent
from agentic_icu.agents.vitals import VitalsAgent
from agentic_icu.config import settings
from agentic_icu.inference.explainer import TabularExplainer
from agentic_icu.inference.sequence import SequenceInference
from agentic_icu.inference.tabular import XGBoostInference
from agentic_icu.orchestration.workflow import AgenticICUWorkflow
from agentic_icu.preprocessing.windowing import RuntimePreprocessor


@lru_cache(maxsize=1)
def get_workflow() -> AgenticICUWorkflow:
    preprocessor = RuntimePreprocessor(
        train_statistics_path=settings.train_statistics_path,
        pipeline_config_path=settings.pipeline_config_path,
    )
    vitals_predictor = SequenceInference(
        model_path=settings.sequence_model_path,
        metrics_path=settings.sequence_metrics_path,
        calibrator_path=settings.sequence_calibrator_path,
    )
    tabular_predictor = XGBoostInference(
        model_path=settings.xgboost_model_path,
        metrics_path=settings.xgboost_metrics_path,
        calibrator_path=settings.xgboost_calibrator_path,
    )
    tabular_explainer = TabularExplainer(tabular_predictor)
    resp_gru_predictor = SequenceInference(
        model_path=settings.resp_sequence_model_path,
        metrics_path=settings.resp_sequence_metrics_path,
        calibrator_path=settings.resp_sequence_calibrator_path,
    )
    resp_xgb_predictor = XGBoostInference(
        model_path=settings.resp_xgboost_model_path,
        metrics_path=settings.resp_xgboost_metrics_path,
        calibrator_path=settings.resp_xgboost_calibrator_path,
    )
    alert_policy = AlertPolicy.from_dict(settings.load_alert_policy())

    return AgenticICUWorkflow(
        signal_quality_agent=SignalQualityAgent(),
        vitals_agent=VitalsAgent(preprocessor, vitals_predictor),
        lab_agent=LabTabularAgent(preprocessor, tabular_predictor, explainer=tabular_explainer),
        resp_failure_agent=RespFailureAgent(preprocessor, resp_gru_predictor, resp_xgb_predictor),
        reasoner=ClinicalReasoner(policy=alert_policy),
    )
