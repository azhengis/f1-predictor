from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from inference.pipeline import predict_race, FEATURE_COLS

app = FastAPI(title="F1 Race Predictor")

class DriverInput(BaseModel):
    driver_name: str
    grid: float
    quali_position: float
    gap_to_pole: float
    quali_teammate_delta: float
    recent_form: float
    dnf_rate: float
    champ_position: float
    champ_points_before: float
    team_podium_rate: float
    pit_mean: float
    pit_std: float
    overtaking_difficulty: float
    driver_circuit_avg: float

class RaceInput(BaseModel):
    drivers: list[DriverInput]

@app.post("/predict")
def predict(race: RaceInput):
    df = pd.DataFrame([d.dict() for d in race.drivers])
    return {"predictions": predict_race(df)}

@app.get("/health")
def health():
    return {"status": "ok"}