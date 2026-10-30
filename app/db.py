from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///./predictions.db",
                       connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    inputs = Column(JSON, nullable=False)
    probability = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()