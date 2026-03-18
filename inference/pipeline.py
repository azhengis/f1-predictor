import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from config import MODEL_PATH
from inference.explainer import get_shap_explanations

FEATURE_COLS = [
    "grid",
    "quali_position", "gap_to_pole", "quali_teammate_delta",
    "recent_form", "dnf_rate",
    "champ_position", "champ_points_before",
    "team_podium_rate", "pit_mean", "pit_std",
    "overtaking_difficulty", "driver_circuit_avg",
]

model = joblib.load(MODEL_PATH)


def predict_race(input_df: pd.DataFrame) -> list:
    X = input_df[FEATURE_COLS].fillna(input_df[FEATURE_COLS].median())
    scores = model.predict(X)
    explanations = get_shap_explanations(model, X)

    # regression: lower score = better position (ascending sort)
    ranked = sorted(
        zip(input_df["driver_name"].values, scores, explanations),
        key=lambda x: x[1],
        reverse=False
    )

    return [
        {
            "position": i + 1,
            "driver": name,
            "score": round(float(score), 3),
            "shap": shap
        }
        for i, (name, score, shap) in enumerate(ranked)
    ]