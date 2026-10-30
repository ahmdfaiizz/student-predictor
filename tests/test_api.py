from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

BASE = dict(age=17, sex="F", Medu=2, Fedu=2, studytime=2, failures=0,
            absences=2, goout=3, health=3, freetime=3, G1=16,
            higher="yes", internet="yes", schoolsup="no", famsup="yes")

def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_low_risk_student():
    r = client.post("/predict", json=BASE)
    assert r.status_code == 200
    assert r.json()["risk_level"] == "low"

def test_high_risk_student():
    r = client.post("/predict", json={**BASE, "G1": 6, "failures": 2, "absences": 15})
    assert r.status_code == 200
    assert r.json()["risk_level"] == "high"
    assert len(r.json()["top_factors"]) == 3

def test_invalid_input_rejected():
    r = client.post("/predict", json={**BASE, "G1": 99})
    assert r.status_code == 422

def test_history_returns_list():
    client.post("/predict", json=BASE)
    r = client.get("/history")
    assert r.status_code == 200
    assert isinstance(r.json(), list) and len(r.json()) >= 1