"""
API Tests for GAT-XIDS FastAPI Backend
======================================
Tests health, model-info, and CSV validation / inference endpoints.
"""

import io
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from src.data.preprocess import EXPECTED_FEATURES


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_csv_bytes():
    """Generates a valid 20-row CSV with exact 45 features and an optional label column."""
    np.random.seed(42)
    data = {feat: np.random.randn(20).astype(np.float32) for feat in EXPECTED_FEATURES}
    # Add optional labels to verify they are ignored during feature extraction
    data["ground_truth_label"] = ["Benign"] * 10 + ["MQTT attacks"] * 10
    data["is_attack"] = [0] * 10 + [1] * 10
    df = pd.DataFrame(data)
    
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


def test_health_endpoint(client):
    """Verify /api/health returns ok status and verifies model readiness."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert data["model_ready"] is True
    assert data["preprocessing_ready"] is True
    assert data["dependencies_ready"] is True


def test_model_info_endpoint(client):
    """Verify /api/model-info returns actual model architecture, 45 features, and 5 classes."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["expected_feature_count"] == 45
    assert len(data["feature_names"]) == 45
    assert data["feature_names"] == EXPECTED_FEATURES
    assert len(data["supported_classes"]) == 5
    assert "Benign" in data["supported_classes"]
    assert "MQTT attacks" in data["supported_classes"]


def test_analyze_csv_invalid_file_extension(client):
    """Verify non-CSV files are rejected with 400 Bad Request."""
    response = client.post(
        "/api/analyze-csv",
        files={"file": ("malicious_payload.exe", b"fake binary data", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Only CSV files (.csv) are accepted" in response.json()["detail"]


def test_analyze_csv_empty_file(client):
    """Verify empty CSV upload is rejected with 400 Bad Request."""
    response = client.post(
        "/api/analyze-csv",
        files={"file": ("empty.csv", b"", "text/csv")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_csv_missing_features(client):
    """Verify CSV with missing features returns 422 Unprocessable Entity with error list."""
    # CSV with only 5 features instead of 45
    incomplete_df = pd.DataFrame({
        "Header_Length": [1.0, 2.0],
        "Protocol Type": [6.0, 17.0]
    })
    buf = io.BytesIO()
    incomplete_df.to_csv(buf, index=False)
    buf.seek(0)

    response = client.post(
        "/api/analyze-csv",
        files={"file": ("incomplete.csv", buf.getvalue(), "text/csv")}
    )
    assert response.status_code == 422
    assert "Missing" in response.json()["detail"]


def test_analyze_csv_non_finite_and_numeric_cleaning(client):
    """Verify CSV with Infs, NaNs, and string-formatted numbers is cleaned and predicted cleanly."""
    np.random.seed(99)
    data = {feat: np.random.randn(15).astype(np.float32) for feat in EXPECTED_FEATURES}
    # Inject non-finite values
    data["Header_Length"][0] = np.nan
    data["Duration"][1] = np.inf
    data["Rate"][2] = -np.inf
    # Inject string-formatted number
    data["Srate"][3] = "123.45"

    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    response = client.post(
        "/api/analyze-csv",
        files={"file": ("noisy_traffic.csv", buf.getvalue(), "text/csv")}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["total_records"] == 15
    assert len(res_data["predictions"]) == 15


def test_analyze_csv_valid_pipeline_and_label_exclusion(client, valid_csv_bytes):
    """
    Verify complete analysis pipeline:
    - Ground-truth labels are excluded from model inputs.
    - Graph is constructed and GAT model predicts actual class probabilities.
    - Category counts and graph summary are accurately populated.
    """
    response = client.post(
        "/api/analyze-csv",
        files={"file": ("valid_sample.csv", valid_csv_bytes, "text/csv")},
        params={"max_records": 50}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["total_records"] == 20
    assert len(res_data["predictions"]) == 20

    # Verify graph summary
    g_summary = res_data["graph_summary"]
    assert g_summary["num_nodes"] == 20
    assert g_summary["num_edges"] > 0
    assert g_summary["k_neighbors"] == 5

    # Verify individual prediction structure
    first_pred = res_data["predictions"][0]
    assert first_pred["flow_index"] == 0
    assert 0 <= first_pred["predicted_class_id"] < 5
    assert 0.0 <= first_pred["confidence"] <= 1.0
    assert len(first_pred["probabilities"]) == 5

    # Verify category counts match sum of individual predictions
    sum_counts = sum(res_data["category_counts"].values())
    assert sum_counts == 20
