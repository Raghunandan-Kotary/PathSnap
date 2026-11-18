"""
Automated Unit and Integration Tests for PathSnap FastAPI Backend (Person 4).
Verifies:
1. Health check endpoint (/health)
2. Trips list endpoint (/trips)
3. Prediction endpoint (/predict/{trip_id})
4. Error handling (404 for invalid trip)
5. Ground truth isolation (verifies ground-truth routes never leak to client)
6. Evaluator audit logging
"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app
from api.config import settings

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    """Verifies that the API service is alive and healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data

def test_trips_endpoint(client):
    """Verifies that the /trips endpoint returns the 10 evaluation trips."""
    response = client.get("/trips")
    assert response.status_code == 200
    data = response.json()
    assert "trips" in data
    assert data["count"] == 10
    # Evaluator trips are listed when installed, otherwise the student-test trips
    expected = settings.EVALUATOR_TEST_DIR if settings.EVALUATOR_TEST_DIR.exists() else settings.STUDENT_TEST_DIR
    if expected.exists():
        assert data["trips"] == sorted(d.name for d in expected.iterdir() if d.is_dir())

def test_predict_valid_trip(client):
    """Verifies that the prediction pipeline executes and returns valid metrics."""
    trip_id = "00000000"
    response = client.post(f"/predict/{trip_id}")
    assert response.status_code == 200
    data = response.json()
    
    assert data["trip_id"] == trip_id
    assert data["status"] == "success"
    assert data["num_observations"] > 0
    assert len(data["predicted_arcs"]) == data["num_observations"]
    
    # Check metrics
    metrics = data["metrics"]
    assert "pointwise_top1_accuracy" in metrics
    assert "route_continuity" in metrics
    assert 0.0 <= metrics["pointwise_top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["route_continuity"] <= 1.0
    
    # Check timing breakdown
    timing = data["timing"]
    assert "data_load_ms" in timing
    assert "feature_extraction_ms" in timing
    assert "model_inference_ms" in timing
    assert "viterbi_decode_ms" in timing
    assert "total_latency_ms" in timing
    assert timing["total_latency_ms"] > 0

def test_ground_truth_isolation(client):
    """
    CRITICAL EVALUATOR RULE:
    Ground truth reference routes must remain strictly server-side.
    Asserts that the response payload NEVER includes the raw reference route.
    """
    trip_id = "00000007"
    response = client.post(f"/predict/{trip_id}")
    assert response.status_code == 200
    data = response.json()
    
    # Confirm ground truth keys are not exposed in prediction payload
    assert "ground_truth_route" not in data
    assert "reference_route" not in data
    assert "raw_route" not in data
    assert "gt_arcs" not in data
    # The map sent to the UI must not draw the reference route either
    assert "Reference route" not in (data.get("map_html") or "")

def test_predict_invalid_trip_returns_404(client):
    """Verifies that non-existent trips trigger HTTP 404."""
    response = client.post("/predict/99999999")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data

def test_evaluator_request_logging(client):
    """Verifies that requests write to api/logs/requests.log."""
    log_file = settings.LOG_FILE
    assert log_file.exists()
    
    initial_size = log_file.stat().st_size
    client.post("/predict/00000009")
    new_size = log_file.stat().st_size
    
    assert new_size > initial_size
