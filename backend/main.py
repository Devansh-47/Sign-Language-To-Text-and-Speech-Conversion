import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from typing import List
import uvicorn

from backend.schemas import PredictionRequest
from backend.preprocessing import Preprocessor

app = FastAPI(title="HandSpeak AI Inference Engine", version="1.0.0")

# Global Variables
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "asl_model_v2.onnx")
ort_session = None
preprocessor = Preprocessor(target_frames=30)
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

@app.on_event("startup")
def load_model():
    global ort_session
    if os.path.exists(MODEL_PATH):
        try:
            ort_session = ort.InferenceSession(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}. Prediction endpoints will fail.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if ort_session else "degraded",
        "model_loaded": ort_session is not None
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    if not ort_session:
        raise HTTPException(status_code=503, detail="Model not loaded. Please upload 'asl_model_v2.onnx'.")

    try:
        # 1. Preprocess
        # Convert Pydantic models to list of lists
        raw_sequence = [frame.landmarks for frame in request.sequence]
        
        # Returns (30, 21, 3)
        processed = preprocessor.process(raw_sequence)
        
        # Flatten for model: (1, 30, 63)
        input_tensor = processed.reshape(1, 30, 63).astype(np.float32)

        # 2. Inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # 3. Post-process
        logits = ort_outs[0][0] # (26,)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        letter = CLASSES[class_idx]

        return {
            "letter": letter,
            "confidence": confidence,
            "all_probs": {k: float(v) for k, v in zip(CLASSES, probs)}
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
