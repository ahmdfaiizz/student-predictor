from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from app.schemas import StudentInput, PredictionOut
from app.ml.features import FEATURES

MODEL_PATH = Path(__file__).parent / "ml" / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Student Performance Predictor")

def to_level(p: float) -> str:
    if p < 0.35:
        return "low"
    if p < 0.65:
        return "medium"
    return "high"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(student: StudentInput):
    row = pd.DataFrame([student.model_dump()])[FEATURES]
    prob = float(model.predict_proba(row)[0, 1])
    return PredictionOut(at_risk_probability=round(prob, 3),
                         risk_level=to_level(prob))