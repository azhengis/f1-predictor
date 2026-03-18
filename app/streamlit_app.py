import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import streamlit as st
features_path = Path("data/processed/features.csv")
model_path = Path("models/artifacts/lgbm_model.pkl")

if not features_path.exists() or not model_path.exists():
    with st.spinner("First run: building features and training model. This takes 3-5 minutes..."):
        import subprocess
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models/artifacts", exist_ok=True)
        subprocess.run(["python", "features/pipeline.py"], check=True)
        subprocess.run(["python", "models/train.py"], check=True)
    st.success("Model ready! Reloading...")
    st.rerun()


import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from inference.pipeline import predict_race, FEATURE_COLS
from config import DATA_PROCESSED

st.set_page_config(page_title="F1 Race Predictor", layout="wide")
st.title("F1 Race Prediction System")

@st.cache_data
def load_features():
    return pd.read_csv(DATA_PROCESSED / "features.csv")

features_df = load_features()

@st.cache_data
def load_driver_names():
    drivers = pd.read_csv("data/raw/drivers.csv")
    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
    # integer id map:{1: "Lewis Hamilton", ...}
    int_map = drivers[["driverId", "driver_name"]].set_index("driverId")["driver_name"].to_dict()
    # string id map: {"hamilton": "Lewis Hamilton", ...}
    str_map = drivers[["driverRef", "driver_name"]].set_index("driverRef")["driver_name"].to_dict()
    # merge both
    combined = {str(k): v for k, v in int_map.items()}
    combined.update(str_map)
    return combined

@st.cache_data  
def load_race_names():
    races = pd.read_csv("data/raw/races.csv")
    races["year"] = pd.to_numeric(races["year"], errors="coerce").fillna(0).astype(int)
    races["label"] = races["year"].astype(str) + " - " + races["name"]
    return races[["raceId", "label"]].set_index("raceId")["label"].to_dict()

driver_names = load_driver_names()
race_labels  = load_race_names()

st.sidebar.header("Select a Race to Predict")
years = sorted(features_df["year"].dropna().astype(int).unique(), reverse=True)
selected_year = st.sidebar.selectbox("Year", years)

race_ids = features_df[features_df["year"] == selected_year]["raceId"].unique()
race_label_map = {rid: race_labels.get(rid, str(rid)) for rid in race_ids}
selected_race = st.sidebar.selectbox(
    "Race", 
    options=sorted(race_ids),
    format_func=lambda x: race_label_map[x]
)
if st.sidebar.button("Predict Race", type="primary"):
    race_data = features_df[features_df["raceId"] == selected_race].copy()
    race_data["driver_name"] = race_data["driverId"].map(driver_names).fillna(race_data["driverId"].astype(str))

    predictions = predict_race(race_data)
    pred_df = pd.DataFrame(predictions)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Predicted Finishing Order")
        display_df = pred_df[["position", "driver", "score"]].copy()
        display_df.columns = ["Position", "Driver", "Model Score"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Score Distribution")
        fig = px.bar(
            pred_df, x="driver", y="score",
            color="score", color_continuous_scale="RdYlGn_r",
            labels={"driver": "Driver", "score": "Predicted Position Score"},
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("SHAP Explanations - Top 3 Drivers")
    for pred in predictions[:3]:
        with st.expander(f"P{pred['position']} — Driver {pred['driver']}"):
            shap_df = pd.DataFrame(
                list(pred["shap"].items()),
                columns=["Feature", "SHAP Value"]
            ).sort_values("SHAP Value")

            fig2 = px.bar(
                shap_df, x="SHAP Value", y="Feature",
                orientation="h",
                color="SHAP Value",
                color_continuous_scale="RdBu",
                title=f"Feature impact for driver {pred['driver']}",
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### Actual Result vs Prediction")
    actual = race_data.sort_values("positionOrder").head(20).reset_index(drop=True)
    actual["Driver"] = actual["driverId"].map(driver_names).fillna(actual["driverId"].astype(str))
    actual["Actual Position"] = actual["positionOrder"].astype(int)
    actual["Predicted Position"] = actual["Driver"].map(
        {p["driver"]: p["position"] for p in predictions}
    )
    actual["Delta"] = actual["Actual Position"] - actual["Predicted Position"]
    actual["Delta"] = actual["Delta"].apply(
        lambda x: f"+{int(x)}" if pd.notna(x) and x > 0 
        else (str(int(x)) if pd.notna(x) else "—")
    )
    st.dataframe(
        actual[["Actual Position", "Driver", "Predicted Position", "Delta"]],
        use_container_width=True,
        hide_index=True
    )