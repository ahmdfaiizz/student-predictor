from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.db import get_db, Prediction
from app.schemas import StudentInput, PredictionOut, HistoryItem
from app.ml.features import FEATURES
from app.ml.explain import Explainer

MODEL_PATH = Path(__file__).parent / "ml" / "model.joblib"
model = joblib.load(MODEL_PATH)
explainer = Explainer(model)

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
def predict(student: StudentInput, db: Session = Depends(get_db)):
    data = student.model_dump()
    row = pd.DataFrame([data])[FEATURES]
    prob = round(float(model.predict_proba(row)[0, 1]), 3)
    level = to_level(prob)

    db.add(Prediction(inputs=data, probability=prob, risk_level=level))
    db.commit()

    return PredictionOut(at_risk_probability=prob, risk_level=level,
                         top_factors=explainer.top_factors(row))

@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = 20, db: Session = Depends(get_db)):
    return (db.query(Prediction)
              .order_by(Prediction.id.desc())
              .limit(min(limit, 100))
              .all())