import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tgn.inference import TGNInferenceService

app = FastAPI(title="TGN Inference Service API", version="1.0.0")

_inference_service: Optional[TGNInferenceService] = None

class PredictRequest(BaseModel):
    src_node_id: int
    dst_node_id: int
    timestamp: float

class PredictResponse(BaseModel):
    probability: float

def get_inference_service() -> Optional[TGNInferenceService]:
    global _inference_service
    if _inference_service is None:
        checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
        if os.path.exists(checkpoint_path):
            try:
                _inference_service = TGNInferenceService(checkpoint_path)
            except Exception as e:
                _inference_service = None
    return _inference_service

@app.get("/health")
def health_check():
    service = get_inference_service()
    status = "healthy" if service is not None else "degraded (no checkpoint loaded)"
    return {"status": status}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    service = get_inference_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service unavailable or checkpoint not loaded.")

    probs = service.predict_link_probabilities(
        src_ids=[request.src_node_id],
        dst_ids=[request.dst_node_id],
        timestamps=[request.timestamp]
    )
    return PredictResponse(probability=probs[0])
