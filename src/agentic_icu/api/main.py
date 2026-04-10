from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agentic_icu.api.dependencies import get_workflow
from agentic_icu.config import settings
from agentic_icu.domain.contracts import (
    AgentExplanation,
    EvaluatePatientRequest,
    EvaluatePatientResponse,
    ExplainPatientResponse,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Agentic-ICU Rebuild API",
    description="Clean runtime API and dashboard for the rebuilt multi-agent ICU workflow.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Error handlers ────────────────────────────────────────────────────────────

def _serializable_errors(errors: list) -> list:
    """Convert Pydantic error dicts to JSON-safe plain dicts (ctx values may be exceptions)."""
    safe = []
    for err in errors:
        item = {k: (str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v)
                for k, v in err.items() if k != "ctx"}
        if "ctx" in err:
            item["ctx"] = {k: str(v) for k, v in err["ctx"].items()}
        safe.append(item)
    return safe


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": _serializable_errors(exc.errors()),
        },
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )


def latest_alert_policy_report_path() -> Path:
    reports_dir = Path(settings.reports_dir)
    candidates = sorted(reports_dir.glob("alert_policy_comparison_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No alert policy comparison report was found.")
    return candidates[0]


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    workflow = get_workflow()
    preprocessing_ready = workflow.vitals_agent.preprocessor.available
    xgboost_present = workflow.lab_agent.predictor.available
    sequence_present = workflow.vitals_agent.predictor.available
    xgboost_loaded = False
    sequence_loaded = False
    load_latency_ms: float | None = None

    t0 = time.monotonic()
    if preprocessing_ready and xgboost_present:
        try:
            workflow.lab_agent.predictor.load()
            xgboost_loaded = True
        except Exception:
            xgboost_loaded = False

    if preprocessing_ready and sequence_present:
        try:
            workflow.vitals_agent.predictor.load()
            sequence_loaded = True
        except Exception:
            sequence_loaded = False
    load_latency_ms = round((time.monotonic() - t0) * 1000, 1)

    # Count available patient files for a rough data-readiness signal
    raw_dir = Path(settings.raw_data_dir)
    patient_count: int | None = None
    try:
        patient_count = sum(1 for _ in raw_dir.glob("*.psv"))
    except Exception:
        pass

    return {
        "status": "ok",
        "preprocessing_ready": preprocessing_ready,
        "xgboost_ready": xgboost_present and xgboost_loaded,
        "sequence_ready": sequence_present and sequence_loaded,
        "load_latency_ms": load_latency_ms,
        "patient_count": patient_count,
        "host": settings.host,
        "port": settings.port,
    }


@app.get("/runtime-config")
def runtime_config() -> dict:
    workflow = get_workflow()
    sequence_predictor = workflow.vitals_agent.predictor
    tabular_predictor = workflow.lab_agent.predictor

    sequence_threshold = sequence_predictor.decision_threshold if sequence_predictor.available else None
    tabular_threshold = tabular_predictor.decision_threshold if tabular_predictor.available else None

    return {
        "alert_policy": workflow.reasoner.policy.__dict__,
        "model_thresholds": {
            "sequence_threshold": sequence_threshold,
            "xgboost_threshold": tabular_threshold,
        },
    }


@app.get("/reports/alert-policy-latest")
def alert_policy_latest_report() -> dict:
    try:
        report_path = latest_alert_policy_report_path()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    profiles = payload.get("profiles", [])
    best_profile = None
    if profiles:
        best_profile = max(profiles, key=lambda profile: profile.get("metrics", {}).get("balanced_accuracy", 0.0))

    return {
        "report_name": report_path.name,
        "patients_evaluated": payload.get("patients_evaluated"),
        "observation_rows": payload.get("observation_rows"),
        "profiles": profiles,
        "best_profile_by_balanced_accuracy": best_profile["profile"] if best_profile else None,
    }


@app.get("/demo-patient/{patient_id}", response_model=EvaluatePatientRequest)
def demo_patient(patient_id: str, max_rows: int = 24) -> EvaluatePatientRequest:
    patient_path = Path(settings.raw_data_dir) / f"{patient_id}.psv"
    if not patient_path.exists():
        raise HTTPException(status_code=404, detail=f"Demo patient not found: {patient_id}")

    rows = []
    with patient_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            values = {}
            for key, value in row.items():
                if key == "SepsisLabel" or value in (None, ""):
                    continue
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    values[key] = numeric_value
            rows.append({"values": values})

    if not rows:
        raise HTTPException(status_code=422, detail=f"No usable observations found for patient {patient_id}")

    return EvaluatePatientRequest(patient_id=patient_id, observation_window=rows)


@app.get("/demo-patients")
def list_demo_patients() -> dict:
    """Return the canonical demo patient pool used by the dashboard."""
    pool = [
        {"id": "p000001", "label": "Stable",     "tone": "low"},
        {"id": "p000026", "label": "Watch",       "tone": "medium"},
        {"id": "p000028", "label": "High Risk",   "tone": "high"},
        {"id": "p000002", "label": "Stable",      "tone": "low"},
        {"id": "p000004", "label": "Stable",      "tone": "low"},
        {"id": "p000005", "label": "Stable",      "tone": "low"},
        {"id": "p000006", "label": "Stable",      "tone": "low"},
        {"id": "p000011", "label": "Suppressed",  "tone": "low"},
    ]
    raw_dir = Path(settings.raw_data_dir)
    available = [p for p in pool if (raw_dir / f"{p['id']}.psv").exists()]
    return {"patients": available}


# In-memory cache — built once on first request, reused for all subsequent calls
_patient_id_cache: list[str] | None = None


def _get_all_patient_ids() -> list[str]:
    global _patient_id_cache
    if _patient_id_cache is None:
        raw_dir = Path(settings.raw_data_dir)
        _patient_id_cache = sorted(p.stem for p in raw_dir.glob("*.psv"))
    return _patient_id_cache


@app.get("/patients")
def search_patients(search: str = "", limit: int = 100, offset: int = 0) -> dict:
    """Search all patient files in the raw data directory."""
    all_ids = _get_all_patient_ids()
    q = search.strip().lower()
    filtered = [pid for pid in all_ids if q in pid.lower()] if q else all_ids
    total = len(filtered)
    page = filtered[offset : offset + limit]
    return {
        "patients": [{"id": pid} for pid in page],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/model-metrics")
def model_metrics() -> dict:
    """Return structured training metrics for all trained models."""
    paths = {
        "sepsis_gru": Path(settings.sequence_metrics_path),
        "sepsis_xgb": Path(settings.xgboost_metrics_path),
        "resp_gru":   Path(settings.resp_sequence_metrics_path),
        "resp_xgb":   Path(settings.resp_xgboost_metrics_path),
    }

    def load_json(p: Path) -> dict:
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def cls_metrics(block: dict, key: str) -> dict:
        cr = block.get("classification_report", {})
        c  = cr.get(key, {})
        return {
            "precision": round(c.get("precision", 0), 4),
            "recall":    round(c.get("recall", 0), 4),
            "f1":        round(c.get("f1-score", 0), 4),
        }

    def summary(raw: dict, cls_key: str) -> dict:
        t = raw.get("test_metrics", {})
        s = raw.get("threshold_selection", {})
        c = cls_metrics(t, cls_key)
        return {
            "auc":               round(t.get("auc", 0), 4),
            "average_precision": round(t.get("average_precision", 0), 4),
            "brier_score":       round(t.get("brier_score", 0), 4),
            "precision": c["precision"],
            "recall":    c["recall"],
            "f1":        c["f1"],
            "threshold": round(s.get("threshold", 0), 4),
        }

    def training_history(raw: dict) -> list:
        return [
            {
                "epoch":      h["epoch"],
                "train_loss": round(h["train_loss"], 4),
                "val_loss":   round(h["val_loss"], 4),
                "val_auc":    round(h.get("val_auc", 0), 4),
                "val_ap":     round(h.get("val_average_precision", 0), 4),
            }
            for h in raw.get("history", [])
        ]

    def top_features(raw: dict, n: int = 10) -> list:
        fi = raw.get("feature_importance_gain", {})
        items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:n]
        mx = items[0][1] if items else 1.0
        return [{"feature": k, "gain": round(v, 1), "rel": round(v / mx, 4)} for k, v in items]

    raw = {k: load_json(p) for k, p in paths.items()}

    return {
        "sepsis_gru": {
            "name": "Sepsis GRU", "type": "sequence",
            "metrics":      summary(raw["sepsis_gru"], "1.0"),
            "architecture": raw["sepsis_gru"].get("architecture", {}),
            "history":      training_history(raw["sepsis_gru"]),
        },
        "sepsis_xgb": {
            "name": "Sepsis XGBoost", "type": "tabular",
            "metrics":       summary(raw["sepsis_xgb"], "1"),
            "feature_count": raw["sepsis_xgb"].get("feature_count", 0),
            "top_features":  top_features(raw["sepsis_xgb"]),
        },
        "resp_gru": {
            "name": "Resp GRU", "type": "sequence",
            "metrics":      summary(raw["resp_gru"], "1.0"),
            "architecture": raw["resp_gru"].get("architecture", {}),
            "history":      training_history(raw["resp_gru"]),
        },
        "resp_xgb": {
            "name": "Resp XGBoost", "type": "tabular",
            "metrics":       summary(raw["resp_xgb"], "1"),
            "feature_count": raw["resp_xgb"].get("feature_count", 0),
            "top_features":  top_features(raw["resp_xgb"]),
        },
    }


@app.post("/evaluate", response_model=EvaluatePatientResponse)
def evaluate_patient(request: EvaluatePatientRequest) -> EvaluatePatientResponse:
    return get_workflow().evaluate(request)


@app.post("/explain", response_model=ExplainPatientResponse)
def explain_patient(request: EvaluatePatientRequest) -> ExplainPatientResponse:
    """Return SHAP feature contributions (lab) and temporal saliency (vitals) without running the full decision pipeline."""
    workflow = get_workflow()
    records = request.observation_window

    # --- Lab / SHAP ---
    lab_agent = workflow.lab_agent
    lab_explanation = AgentExplanation(status="unavailable")
    if lab_agent.predictor.available and lab_agent.explainer is not None:
        try:
            features = lab_agent.preprocessor.build_tabular_features(records)
            contributions, _ = lab_agent.explainer.top_contributions(features, n=3)
            lab_explanation = AgentExplanation(
                status="available",
                feature_contributions={item["feature"]: item["shap_value"] for item in contributions},
                explanation=lab_agent.explainer.format_explanation(contributions),
            )
        except Exception:
            pass

    # --- Vitals / Temporal saliency ---
    vitals_agent = workflow.vitals_agent
    vitals_explanation = AgentExplanation(status="unavailable")
    if vitals_agent.preprocessor.available and vitals_agent.predictor.available:
        try:
            sequence_tensor = vitals_agent.preprocessor.build_sequence_tensor(records)
            weights = vitals_agent.predictor.temporal_saliency(sequence_tensor)
            n = len(weights)
            vitals_explanation = AgentExplanation(
                status="available",
                feature_contributions={f"t_{i + 1:02d}": float(w) for i, w in enumerate(weights)},
                explanation=f"Temporal saliency over {n} observation hours.",
            )
        except Exception:
            pass

    return ExplainPatientResponse(
        patient_id=request.patient_id,
        lab_explanation=lab_explanation,
        vitals_explanation=vitals_explanation,
    )
