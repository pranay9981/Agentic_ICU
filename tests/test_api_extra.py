from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Mock the warmup runs to 0 before importing app to avoid lifespan delay
import agentic_icu.api.main as main_mod

main_mod._WARMUP_RUNS = 0

from agentic_icu.api.main import app


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock heavy dependencies globally for these tests."""
    from agentic_icu.agents.reasoner import AlertPolicy

    mock_wf = MagicMock()
    mock_wf.vitals_agent.model = MagicMock()
    mock_wf.lab_agent.model = MagicMock()

    # Give the reasoner a real policy so runtime-config can serialize it
    real_policy = AlertPolicy()
    mock_wf.reasoner.policy = real_policy

    # Give predictors real threshold values so runtime-config returns numeric data
    mock_wf.vitals_agent.predictor.available = True
    mock_wf.vitals_agent.predictor.decision_threshold = 0.5
    mock_wf.lab_agent.predictor.available = True
    mock_wf.lab_agent.predictor.decision_threshold = 0.5
    mock_wf.resp_failure_agent = None

    with patch("agentic_icu.api.main.get_workflow", return_value=mock_wf):
        yield mock_wf

@pytest.fixture
def client():
    # Set raise_server_exceptions=False to test error handlers
    return TestClient(app, raise_server_exceptions=False)

def test_dashboard_endpoint(client):
    """Test the dashboard (index.html) endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_health_latency_measurement(client):
    """Test health check latency logic."""
    # Patch time.monotonic in the actual module with an infinite side effect
    t = [10.0]
    def mock_monotonic():
        t[0] += 0.05
        return t[0]

    with patch("agentic_icu.api.main.time.monotonic", side_effect=mock_monotonic):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "load_latency_ms" in data

def test_alert_policy_latest_report_error(client):
    """Test 404 when no alert policy report exists."""
    with patch("agentic_icu.api.main.latest_alert_policy_report_path", side_effect=FileNotFoundError("No report")):
        response = client.get("/reports/alert-policy-latest")
        assert response.status_code == 404
        assert response.json()["detail"] == "No report"

def test_alert_policy_latest_report_success(client, tmp_path):
    """Test successful report loading and best profile logic."""
    report_data = {
        "patients_evaluated": 10,
        "observation_rows": 100,
        "profiles": [
            {"profile": {"p1": 1}, "metrics": {"balanced_accuracy": 0.5}},
            {"profile": {"p2": 2}, "metrics": {"balanced_accuracy": 0.8}}
        ]
    }
    report_file = tmp_path / "alert_policy_comparison_2026.json"
    report_file.write_text(json.dumps(report_data))

    with patch("agentic_icu.api.main.latest_alert_policy_report_path", return_value=report_file):
        response = client.get("/reports/alert-policy-latest")
        assert response.status_code == 200
        body = response.json()
        assert body["best_profile_by_balanced_accuracy"] == {"p2": 2}

def test_demo_patient_value_parsing_error(client, tmp_path):
    """Test demo patient with unparseable values."""
    patient_dir = tmp_path / "raw"
    patient_dir.mkdir()
    patient_file = patient_dir / "p999999.psv"
    patient_file.write_text("ICULOS|HR|Temp\n1.0|80|invalid\n")

    with patch("agentic_icu.config.settings.raw_data_dir", str(patient_dir)):
        with patch("agentic_icu.api.main._patient_ids", ["p999999"]):
            response = client.get("/demo-patient/p999999")
            assert response.status_code == 200
            body = response.json()
            # ICULOS is handled separately, HR is 80.0, Temp is skipped
            assert body["observation_window"][0]["values"]["HR"] == 80.0
            assert "Temp" not in body["observation_window"][0]["values"]

def test_demo_patient_no_usable_rows(client, tmp_path):
    """Test demo patient with no usable numeric rows."""
    patient_dir = tmp_path / "raw"
    patient_dir.mkdir()
    patient_file = patient_dir / "pEmpty.psv"
    patient_file.write_text("HR|Temp\ninvalid|invalid\n")

    with patch("agentic_icu.config.settings.raw_data_dir", str(patient_dir)):
        response = client.get("/demo-patient/pEmpty")
        assert response.status_code == 200
        # Hardened API preserves the structure but values are empty for unparseable rows
        window = response.json()["observation_window"]
        assert len(window) > 0
        assert window[0]["values"] == {}

def test_model_metrics_missing_files(client, tmp_path):
    """Test model metrics when some JSON files are missing."""
    with patch("agentic_icu.config.settings.sequence_metrics_path", str(tmp_path / "miss.json")):
        response = client.get("/model-metrics")
        assert response.status_code == 200

def test_explain_patient_signal_quality_suppression(client, mock_dependencies):
    """Test /explain behavior when signal is fully suppressed."""
    mock_sq = MagicMock()
    # Explicitly set boolean values to avoid MagicMock truthiness confusion
    mock_sq.signal_valid = False
    mock_sq.suppression_recommendation = True

    mock_dependencies.signal_quality_agent.evaluate.return_value = (mock_sq, [])

    response = client.post("/explain", json={
        "patient_id": "suppressed",
        "observation_window": [{"values": {"HR": 80.0, "ICULOS": 1.0}}]
    })
    assert response.status_code == 200
    # Both explanations should be marked as unavailable due to signal suppression
    data = response.json()
    assert data["lab_explanation"]["status"] == "unavailable"
    assert data["vitals_explanation"]["status"] == "unavailable"
    assert "explanation" in data["lab_explanation"]

def test_runtime_config_endpoint(client):
    """Test the /runtime-config endpoint."""
    response = client.get("/runtime-config")
    assert response.status_code == 200
    body = response.json()
    assert "alert_policy" in body

def test_demo_patients_logic(client, tmp_path):
    """Test demo-patients pool filtering."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "p000001.psv").write_text("dummy")

    with patch("agentic_icu.config.settings.raw_data_dir", str(raw_dir)):
        response = client.get("/demo-patients")
        assert response.status_code == 200
        patients = response.json()["patients"]
        assert len(patients) == 1


def test_alert_policy_latest_report_corrupt_json(client, tmp_path):
    """Test 500 response when report file contains corrupt JSON."""
    corrupt_file = tmp_path / "bad_report.json"
    corrupt_file.write_text("NOT VALID JSON {{{")
    with patch("agentic_icu.api.main.latest_alert_policy_report_path", return_value=corrupt_file):
        response = client.get("/reports/alert-policy-latest")
        assert response.status_code == 500


def test_connection_manager_connect_disconnect():
    """Unit test for ConnectionManager connect/disconnect lifecycle."""
    import asyncio

    from agentic_icu.api.main import ConnectionManager

    manager = ConnectionManager()

    mock_ws = MagicMock()
    async def _accept(): pass
    mock_ws.accept = _accept

    async def run():
        await manager.connect(mock_ws)
        assert mock_ws in manager.active_connections
        manager.disconnect(mock_ws)
        assert mock_ws not in manager.active_connections
        # Double disconnect should be a no-op (no KeyError)
        manager.disconnect(mock_ws)

    asyncio.run(run())


def test_connection_manager_broadcast_removes_dead_connections():
    """Broadcast should remove connections that raise on send."""
    import asyncio

    from agentic_icu.api.main import ConnectionManager

    manager = ConnectionManager()

    bad_ws = MagicMock()
    async def fail_send(msg):
        raise RuntimeError("connection closed")
    bad_ws.send_text = fail_send
    manager.active_connections.append(bad_ws)

    async def run():
        await manager.broadcast("test message")
        assert bad_ws not in manager.active_connections

    asyncio.run(run())


def test_websocket_connect_and_receive(client):
    """Test the /ws/{client_id} WebSocket endpoint connects and echoes."""
    with client.websocket_connect("/ws/test-client-001") as ws:
        ws.send_text("ping")
        data = ws.receive_text()
        assert "ping" in data or "ACK" in data


def test_serializable_errors_helper():
    """Direct unit test for the _serializable_errors helper."""
    from agentic_icu.api.main import _serializable_errors

    errors = [
        {"type": "value_error", "msg": "bad value", "ctx": {"error": ValueError("oops")}},
        {"type": "missing", "msg": "field required"},
    ]
    result = _serializable_errors(errors)
    assert len(result) == 2
    assert result[0]["ctx"]["error"] == "oops"
    assert result[1]["msg"] == "field required"


def test_lifespan_startup_runs_warmup(tmp_path):
    """Test that lifespan startup successfully pre-warms workflow models."""
    from fastapi.testclient import TestClient

    from agentic_icu.api.main import app

    mock_wf = MagicMock()
    mock_wf.vitals_agent.predictor.load = MagicMock()
    mock_wf.lab_agent.predictor.load = MagicMock()
    mock_wf.resp_failure_agent = None
    mock_wf.ensemble = None
    # Ensure predictors report as unavailable so quality gate is skipped cleanly
    mock_wf.vitals_agent.predictor.available = False
    mock_wf.lab_agent.predictor.available = False

    with patch("agentic_icu.api.main.get_workflow", return_value=mock_wf):
        with patch("agentic_icu.api.main.settings.raw_data_dir", str(tmp_path)):
            with TestClient(app, raise_server_exceptions=True) as c:
                response = c.get("/health")
                assert response.status_code == 200

    # load() is called once during lifespan startup
    mock_wf.vitals_agent.predictor.load.assert_called()
    mock_wf.lab_agent.predictor.load.assert_called()


def test_patients_search_endpoint(client):
    """Test /patients search endpoint with search and pagination params."""
    response = client.get("/patients?search=p000&limit=5&offset=0")
    assert response.status_code == 200
    body = response.json()
    assert "patients" in body
    assert "total" in body
    assert len(body["patients"]) <= 5


def test_favicon_endpoint(client):
    """Test /favicon.ico returns a file response."""
    response = client.get("/favicon.ico")
    # Should be 200 or 404 but not a 500
    assert response.status_code in (200, 404)
