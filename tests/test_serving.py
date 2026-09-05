import os
import pytest
import torch
from fastapi.testclient import TestClient
from serving.app import app, get_inference_service
import serving.app as serving_app
from tgn.train import TGN

@pytest.fixture
def mock_checkpoint(tmp_path):
    ckpt_path = tmp_path / "test_model.pt"
    device = torch.device("cpu")
    model = TGN(
        num_nodes=10,
        edge_feat_dim=4,
        memory_dim=16,
        message_dim=16,
        time_dim=8,
        embedding_dim=16,
        predictor_hidden_dim=16,
        device=device
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "model": {
                "memory_dim": 16,
                "message_dim": 16,
                "time_dim": 8,
                "embedding_dim": 16,
                "predictor_hidden_dim": 16
            }
        },
        "num_nodes": 10,
        "edge_feat_dim": 4
    }, str(ckpt_path))
    return str(ckpt_path)

def test_serving_endpoints(mock_checkpoint, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_PATH", mock_checkpoint)
    serving_app._inference_service = None  # Reset global state

    client = TestClient(app)

    # Test GET /health
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    # Test POST /predict
    payload = {
        "src_node_id": 1,
        "dst_node_id": 2,
        "timestamp": 100.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "probability" in res_data
    assert 0.0 <= res_data["probability"] <= 1.0
