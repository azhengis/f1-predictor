import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from config import DATA_PROCESSED, MODEL_PATH, TRAIN_YEARS, VAL_YEARS, TEST_YEARS

FEATURE_COLS = [
    "grid",
    "quali_position", "gap_to_pole", "quali_teammate_delta",
    "recent_form", "dnf_rate",
    "champ_position", "champ_points_before",
    "team_podium_rate", "pit_mean", "pit_std",
    "overtaking_difficulty", "driver_circuit_avg",
]
TARGET = "positionOrder"

def train():
    df = pd.read_csv(DATA_PROCESSED / "features.csv")

    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    val_df   = df[df["year"].isin(VAL_YEARS)].copy()
    test_df  = df[df["year"].isin(TEST_YEARS)].copy()

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_val,   y_val   = val_df[FEATURE_COLS],   val_df[TARGET]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=0.5,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    from models.evaluate import evaluate_model
    evaluate_model(model, test_df, FEATURE_COLS)

if __name__ == "__main__":
    train()