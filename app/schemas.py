from typing import Literal
from pydantic import BaseModel, Field

YesNo = Literal["yes", "no"]

class StudentInput(BaseModel):
    age: int = Field(ge=15, le=22)
    sex: Literal["F", "M"]
    Medu: int = Field(ge=0, le=4, description="Mother's education (0-4)")
    Fedu: int = Field(ge=0, le=4, description="Father's education (0-4)")
    studytime: int = Field(ge=1, le=4, description="Weekly study time (1-4)")
    failures: int = Field(ge=0, le=3, description="Past class failures")
    absences: int = Field(ge=0, le=100)
    goout: int = Field(ge=1, le=5)
    health: int = Field(ge=1, le=5)
    freetime: int = Field(ge=1, le=5)
    G1: int = Field(ge=0, le=20, description="First-period grade (0-20)")
    higher: YesNo
    internet: YesNo
    schoolsup: YesNo
    famsup: YesNo

class PredictionOut(BaseModel):
    at_risk_probability: float
    risk_level: Literal["low", "medium", "high"]