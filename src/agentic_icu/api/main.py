from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Sequence

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from agentic_icu.api.dependencies import get_workflow
from agentic_icu.config import BASE_DIR, settings
from agentic_icu.domain.contracts import (
    MAX_WINDOW_ROWS,
    AgentExplanation,
    EvaluatePatientRequest,
    EvaluatePatientResponse,
    ExplainPatientResponse,
    ObservationRecord,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Optional API key auth. Set AGENTIC_ICU_API_KEY env var to enable.
_API_KEY: str = os.environ.get("AGENTIC_ICU_API_KEY", "").strip()
_AUTH_EXEMPT: set[str] = {"/", "/health", "/favicon.ico"}


class _ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if _API_KEY:
            path = request.url.path
            if path not in _AUTH_EXEMPT and not path.startswith("/static/"):
                if request.headers.get("X-API-Key", "") != _API_KEY:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": "unauthorized",
                            "detail": "Valid X-API-Key header required.",
                        },
                    )
        return await call_next(request)


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    _HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "object-src 'none'; "
            "frame-ancestors 'none'"
        ),
    }

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for k, v in self._HEADERS.items():
            response.headers.setdefault(k, v)
        return response


# Model quality floors — warn at startup if a model falls below these.
_MODEL_AUC_FLOOR = 0.70
_MODEL_AP_FLOOR = 0.05

# Patient ID list cached at startup — avoids re-scanning 40k files on every search.
# Rebuilt once when the lifespan starts; stale if files are added/removed at runtime.
_patient_ids: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm models and cache the patient ID list at startup."""
    global _patient_ids
    workflow = get_workflow()
    try:
        workflow.vitals_agent.predictor.load()
        workflow.lab_agent.predictor.load()
        if workflow.resp_failure_agent is not None:
            workflow.resp_failure_agent.gru_predictor.load()
        if workflow.ensemble is not None:
            workflow.ensemble.load()
        logger.info("Model warm-up complete.")
    except Exception as exc:
        logger.warning(
            "Model warm-up failed (%s: %s) — models will load lazily on first request.",
            type(exc).__name__,
            exc,
        )
    try:
        _patient_ids = sorted(p.stem for p in Path(settings.raw_data_dir).glob("*.psv"))
        logger.info("Patient ID cache built: %d patients.", len(_patient_ids))
    except Exception as exc:
        logger.warning(
            "Patient ID cache build failed (%s: %s).", type(exc).__name__, exc
        )

    # Model quality gate — warn at startup if any model is below minimum floors.
    try:
        checks: list[tuple[str, Any]] = [
            ("Sepsis GRU", workflow.vitals_agent.predictor),
            ("Sepsis XGBoost", workflow.lab_agent.predictor),
        ]
        if workflow.resp_failure_agent is not None:
            checks.append(("Resp GRU", workflow.resp_failure_agent.gru_predictor))
        for name, predictor in checks:
            if hasattr(predictor, "available") and predictor.available:
                tm = predictor.metrics.get("test_metrics", {})
                auc = tm.get("auc", 0.0)
                ap = tm.get("average_precision", 0.0)
                if auc < _MODEL_AUC_FLOOR:
                    logger.warning(
                        "Model quality gate: %s AUC=%.3f below floor %.2f",
                        name,
                        auc,
                        _MODEL_AUC_FLOOR,
                    )
                if ap < _MODEL_AP_FLOOR:
                    logger.warning(
                        "Model quality gate: %s AP=%.4f below floor %.2f",
                        name,
                        ap,
                        _MODEL_AP_FLOOR,
                    )
    except Exception as exc:
        logger.warning(
            "Model quality gate check failed (%s: %s).", type(exc).__name__, exc
        )

    # Artifact manifest check
    try:
        manifest_path = BASE_DIR / "artifacts" / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            gru_metrics = workflow.vitals_agent.predictor.metrics
            xgb_metrics = workflow.lab_agent.predictor.metrics
            manifest_checks = [
                ("sepsis_gru_input_size", gru_metrics.get("input_size"), manifest.get("sepsis_gru_input_size")),
                ("sepsis_xgb_feature_count", xgb_metrics.get("feature_count"), manifest.get("sepsis_xgb_feature_count")),
            ]
            for name, actual, expected in manifest_checks:
                if expected is not None and actual != expected:
                    logger.error(
                        "Artifact manifest mismatch: %s expected=%s actual=%s — wrong artifacts loaded!",
                        name, expected, actual,
                    )
                else:
                    logger.info("Artifact manifest OK: %s=%s", name, actual)
    except Exception as exc:
        logger.warning("Artifact manifest check failed (%s: %s).", type(exc).__name__, exc)

    async def _cleanup_rl_counters():
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            now = time.monotonic()
            with _rl_lock:
                stale = [k for k, q in _rl_counters.items() if all(now - t > _RL_WINDOW_S for t in q)]
                for k in stale:
                    del _rl_counters[k]

    _cleanup_task = asyncio.create_task(_cleanup_rl_counters())

    yield

    _cleanup_task.cancel()
    try:
        await _cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Agentic-ICU Rebuild API",
    description="Clean runtime API and dashboard for the rebuilt multi-agent ICU workflow.",
    lifespan=lifespan,
)
app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(_ApiKeyMiddleware)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        # We use a copy of the list to avoid mutations during iteration
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                # Cleanup dead connections lazily
                self.disconnect(connection)


manager = ConnectionManager()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico")


# ── Error handlers ────────────────────────────────────────────────────────────


def _serializable_errors(errors: Sequence[Any]) -> list:
    """Convert Pydantic error dicts to JSON-safe plain dicts."""
    safe = []
    for err in errors:
        item = {
            k: (
                str(v)
                if not isinstance(v, (str, int, float, bool, list, dict, type(None)))
                else v
            )
            for k, v in err.items()
            if k != "ctx"
        }
        if "ctx" in err:
            item["ctx"] = {k: str(v) for k, v in err["ctx"].items()}
        safe.append(item)
    return safe


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
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
        headers=dict(exc.headers) if exc.headers else None,
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = str(uuid.uuid4())[:8]
    logger.exception(
        "Unhandled error [request_id=%s] %s %s",
        request_id,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "An unexpected error occurred.",
            "request_id": request_id,
        },
    )


_PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# ── Rate limiting ─────────────────────────────────────────────────────────────
_RL_WINDOW_S: int = 60
_RL_MAX_REQUESTS: int = 30
_rl_lock = threading.Lock()
_rl_counters: dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request) -> None:
    """FastAPI dependency: sliding-window rate limit per client IP per path."""
    ip = request.client.host if request.client else "unknown"
    key = f"{ip}:{request.url.path}"
    now = time.monotonic()
    with _rl_lock:
        q = _rl_counters[key]
        while q and now - q[0] > _RL_WINDOW_S:
            q.popleft()
        if not q:
            del _rl_counters[key]  # prune exhausted key; q still holds the deque
        if len(q) >= _RL_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded — max {_RL_MAX_REQUESTS} requests per {_RL_WINDOW_S}s.",
                headers={"Retry-After": str(_RL_WINDOW_S)},
            )
        _rl_counters[key].append(now)  # defaultdict re-creates key if it was pruned


def latest_alert_policy_report_path() -> Path:
    """Find the latest pre-computed alert policy comparison report."""
    reports_dir = Path(settings.reports_dir)
    # Check for both timestamped reports and the baseline 'latest' report.
    candidates = sorted(
        reports_dir.glob("alert_policy_comparison_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    fallback = reports_dir / "alert-policy-latest.json"
    if fallback.exists():
        return fallback

    raise FileNotFoundError("No alert policy comparison report was found.")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    workflow = get_workflow()
    preprocessing_ready = workflow.vitals_agent.preprocessor.available

    # Check loaded state without calling load() — avoids expensive disk I/O on
    # every health ping.  Models are loaded lazily on first /evaluate request.
    xgboost_ready = workflow.lab_agent.predictor.available
    sequence_ready = workflow.vitals_agent.predictor.available
    resp_ready = (
        workflow.resp_failure_agent is not None
        and workflow.resp_failure_agent.gru_predictor.available
    )

    # Measure actual load time only once (models self-cache after first load)
    load_latency_ms: float | None = None
    if preprocessing_ready:
        t0 = time.monotonic()
        try:
            workflow.lab_agent.predictor.load()
            workflow.vitals_agent.predictor.load()
            if workflow.resp_failure_agent is not None:
                workflow.resp_failure_agent.gru_predictor.load()
        except Exception:
            logger.debug("Lazy loading failed in health check, continuing.")
        load_latency_ms = round((time.monotonic() - t0) * 1000, 1)

    patient_count = len(_patient_ids) if _patient_ids else None

    return {
        "status": "ok",
        "preprocessing_ready": preprocessing_ready,
        "xgboost_ready": xgboost_ready,
        "sequence_ready": sequence_ready,
        "resp_ready": resp_ready,
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

    sequence_threshold = (
        sequence_predictor.decision_threshold if sequence_predictor.available else None
    )
    tabular_threshold = (
        tabular_predictor.decision_threshold if tabular_predictor.available else None
    )

    ensemble_threshold = workflow.reasoner.policy.high_alert_ensemble_score_threshold

    return {
        "alert_policy": workflow.reasoner.policy.__dict__,
        "model_thresholds": {
            "sequence_threshold": sequence_threshold,
            "xgboost_threshold": tabular_threshold,
            "ensemble_threshold": ensemble_threshold,
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
        best_profile = max(
            profiles,
            key=lambda profile: profile.get("metrics", {}).get(
                "balanced_accuracy", 0.0
            ),
        )

    return {
        "report_name": report_path.name,
        "patients_evaluated": payload.get("patients_evaluated"),
        "observation_rows": payload.get("observation_rows"),
        "profiles": profiles,
        "best_profile_by_balanced_accuracy": (
            best_profile["profile"] if best_profile else None
        ),
    }


@app.get("/demo-patient/{patient_id}", response_model=EvaluatePatientRequest)
def demo_patient(
    patient_id: str,
    max_rows: int = Query(default=24, ge=1, le=MAX_WINDOW_ROWS),
) -> EvaluatePatientRequest:
    if not _PATIENT_ID_RE.match(patient_id):
        raise HTTPException(
            status_code=422, detail=f"Invalid patient ID format: {patient_id!r}"
        )
    raw_dir = Path(settings.raw_data_dir).resolve()
    patient_path = (raw_dir / f"{patient_id}.psv").resolve()
    if not patient_path.is_relative_to(raw_dir) or not patient_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Demo patient not found: {patient_id}"
        )

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
                try:
                    numeric_value = float(value)
                except ValueError:
                    continue
                if math.isfinite(numeric_value):
                    values[key] = numeric_value
            rows.append({"values": values})

    if not rows:
        raise HTTPException(
            status_code=422,
            detail=f"No usable observations found for patient {patient_id}",
        )


    observation_window = [ObservationRecord(values=r["values"]) for r in rows]
    return EvaluatePatientRequest(
        patient_id=patient_id, observation_window=observation_window
    )


@app.get("/demo-patients")
def list_demo_patients() -> dict:
    """Return the canonical demo patient pool used by the dashboard."""
    pool = [
        {"id": "p000001", "label": "Stable", "tone": "low"},
        {"id": "p000026", "label": "Watch", "tone": "medium"},
        {"id": "p000028", "label": "High Risk", "tone": "high"},
        {"id": "p000002", "label": "Stable", "tone": "low"},
        {"id": "p000004", "label": "Stable", "tone": "low"},
        {"id": "p000005", "label": "Stable", "tone": "low"},
        {"id": "p000006", "label": "Stable", "tone": "low"},
        {"id": "p000011", "label": "Suppressed", "tone": "low"},
    ]
    raw_dir = Path(settings.raw_data_dir)
    available = [p for p in pool if (raw_dir / f"{p['id']}.psv").exists()]
    return {"patients": available}


@app.get("/patients")
def search_patients(
    search: str = "",
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Search all patient files using the startup-cached ID list."""
    q = search.strip().lower()
    filtered = [pid for pid in _patient_ids if q in pid.lower()] if q else _patient_ids
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
        "resp_gru": Path(settings.resp_sequence_metrics_path),
        "resp_xgb": Path(settings.resp_xgboost_metrics_path),
        "ensemble": Path(settings.ensemble_metrics_path),
    }

    def load_json(p: Path) -> dict:
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def cls_metrics(block: dict, key: str) -> dict:
        cr = block.get("classification_report", {})
        c = cr.get(key, {})
        return {
            "precision": round(c.get("precision", 0), 4),
            "recall": round(c.get("recall", 0), 4),
            "f1": round(c.get("f1-score", 0), 4),
        }

    def summary(raw: dict, cls_key: str) -> dict:
        t = raw.get("test_metrics", {})
        s = raw.get("threshold_selection", {})
        c = cls_metrics(t, cls_key)
        return {
            "auc": round(t.get("auc", 0), 4),
            "average_precision": round(t.get("average_precision", 0), 4),
            "brier_score": round(t.get("brier_score", 0), 4),
            "precision": c["precision"],
            "recall": c["recall"],
            "f1": c["f1"],
            "threshold": round(s.get("threshold", 0), 4),
        }

    def training_history(raw: dict) -> list:
        return [
            {
                "epoch": h["epoch"],
                "train_loss": round(h["train_loss"], 4),
                "val_loss": round(h["val_loss"], 4),
                "val_auc": round(h.get("val_auc", 0), 4),
                "val_ap": round(h.get("val_average_precision", 0), 4),
            }
            for h in raw.get("history", [])
        ]

    def top_features(raw: dict, n: int = 10) -> list:
        fi = raw.get("feature_importance_gain", {})
        items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:n]
        mx = items[0][1] if items else 1.0
        return [
            {"feature": k, "gain": round(v, 1), "rel": round(v / mx, 4)}
            for k, v in items
        ]

    raw = {k: load_json(p) for k, p in paths.items()}

    ens = raw.get("ensemble", {})
    return {
        "sepsis_gru": {
            "name": "Sepsis GRU",
            "type": "sequence",
            "metrics": summary(raw["sepsis_gru"], "1.0"),
            "architecture": raw["sepsis_gru"].get("architecture", {}),
            "history": training_history(raw["sepsis_gru"]),
        },
        "sepsis_xgb": {
            "name": "Sepsis XGBoost",
            "type": "tabular",
            "metrics": summary(raw["sepsis_xgb"], "1"),
            "feature_count": raw["sepsis_xgb"].get("feature_count", 0),
            "top_features": top_features(raw["sepsis_xgb"]),
        },
        "resp_gru": {
            "name": "Resp GRU",
            "type": "sequence",
            "metrics": summary(raw["resp_gru"], "1.0"),
            "architecture": raw["resp_gru"].get("architecture", {}),
            "history": training_history(raw["resp_gru"]),
        },
        "resp_xgb": {
            "name": "Resp XGBoost",
            "type": "tabular",
            "metrics": summary(raw["resp_xgb"], "1"),
            "feature_count": raw["resp_xgb"].get("feature_count", 0),
            "top_features": top_features(raw["resp_xgb"]),
        },
        "ensemble": {
            "name": "Sepsis Ensemble (GRU+XGB)",
            "type": "ensemble",
            "metrics": {
                "auc": round(ens.get("val_auc", 0), 4),
                "average_precision": round(ens.get("val_auprc", 0), 4),
            },
            "formula": "sigmoid(coef_gru * gru_score + coef_xgb * xgb_score + intercept)",
            "coef_gru": round(ens.get("coef_gru", 0), 4),
            "coef_xgb": round(ens.get("coef_xgb", 0), 4),
            "intercept": round(ens.get("intercept", 0), 4),
        },
    }


@app.get("/model-metrics/calibration")
def model_calibration_curves() -> dict:
    """Return isotonic calibration transformation curves for all models.
    Samples each fitted calibrator at 20 raw score points to show the raw→calibrated mapping.
    """
    workflow = get_workflow()
    import numpy as np

    def sample_calibrator(predictor, name: str) -> dict:
        if not predictor.available:
            return {"available": False}
        try:
            predictor.load()
        except Exception:
            return {"available": False}
        if not predictor.calibrated:
            return {"available": False, "reason": "No calibrator loaded"}
        raw_points = np.linspace(0.0, 1.0, 20).tolist()
        cal_points = [float(predictor._calibrator.predict([r])[0]) for r in raw_points]
        return {
            "available": True,
            "raw": [round(r, 4) for r in raw_points],
            "calibrated": [round(c, 4) for c in cal_points],
            "method": "isotonic_regression",
        }

    result = {
        "sepsis_gru": sample_calibrator(workflow.vitals_agent.predictor, "sepsis_gru"),
        "sepsis_xgb": sample_calibrator(workflow.lab_agent.predictor, "sepsis_xgb"),
    }
    if workflow.resp_failure_agent:
        result["resp_gru"] = sample_calibrator(workflow.resp_failure_agent.gru_predictor, "resp_gru")
    return result


@app.post(
    "/evaluate",
    response_model=EvaluatePatientResponse,
    dependencies=[Depends(_rate_limit)],
)
async def evaluate_patient(request: EvaluatePatientRequest) -> EvaluatePatientResponse:
    result = get_workflow().evaluate(request)

    # WebSocket Broadcast: Notify all clients if a High Priority alert is detected.
    # This allows other clinical stations to see real-time updates.
    if result.clinical_decision.priority == "high":
        try:
            payload = {
                "type": "CLINICAL_ALERT",
                "patient_id": result.patient_id,
                "alert_type": result.clinical_decision.alert_type,
                "priority": result.clinical_decision.priority,
                "timestamp": time.time(),
            }
            await manager.broadcast(json.dumps(payload))
        except Exception as exc:
            logger.warning("Failed to broadcast WebSocket alert: %s", exc)

    return result


@app.post(
    "/explain",
    response_model=ExplainPatientResponse,
    dependencies=[Depends(_rate_limit)],
)
def explain_patient(request: EvaluatePatientRequest) -> ExplainPatientResponse:
    """Return SHAP feature contributions (lab) and temporal saliency (vitals) without running the full decision pipeline."""  # noqa: E501
    workflow = get_workflow()
    records = request.observation_window

    # Signal quality gate — running SHAP on a fully suppressed (artifacted) window
    # would return misleading contributions that contradict the /evaluate decision.
    signal_quality, _ = workflow.signal_quality_agent.evaluate(
        [record.values for record in records]
    )
    if not signal_quality.signal_valid and signal_quality.suppression_recommendation:
        _suppressed = AgentExplanation(
            status="unavailable",
            explanation="Signal fully suppressed by Signal Quality Agent — explanation unavailable on an artifacted window.",  # noqa: E501
        )
        return ExplainPatientResponse(
            patient_id=request.patient_id,
            lab_explanation=_suppressed,
            vitals_explanation=_suppressed,
        )

    # --- Lab / SHAP ---
    lab_agent = workflow.lab_agent
    lab_explanation = AgentExplanation(status="unavailable")
    if lab_agent.predictor.available and lab_agent.explainer is not None:
        try:
            features = lab_agent.preprocessor.build_tabular_features(records)
            contributions, _ = lab_agent.explainer.top_contributions(features, n=3)
            lab_explanation = AgentExplanation(
                status="available",
                feature_contributions={
                    item["feature"]: item["shap_value"] for item in contributions
                },
                explanation=lab_agent.explainer.format_explanation(contributions),
            )
        except Exception as exc:
            logger.warning(
                "/explain: SHAP computation failed — %s: %s", type(exc).__name__, exc
            )

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
                feature_contributions={
                    f"t_{i + 1:02d}": float(w) for i, w in enumerate(weights)
                },
                explanation=f"Temporal saliency over {n} observation hours.",
            )
        except Exception as exc:
            logger.warning(
                "/explain: saliency computation failed — %s: %s",
                type(exc).__name__,
                exc,
            )

    return ExplainPatientResponse(
        patient_id=request.patient_id,
        lab_explanation=lab_explanation,
        vitals_explanation=vitals_explanation,
    )


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    logger.info("WebSocket connected: %s", client_id)
    try:
        while True:
            # Keep connection alive; can also receive heartbeats here
            data = await websocket.receive_text()
            # Echo or process client messages if needed
            await websocket.send_text(f"ACK: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected: %s", client_id)
    except Exception as e:
        logger.warning("WebSocket error for %s: %s", client_id, e)
        manager.disconnect(websocket)
