import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from config import RECENT_RACES_WINDOW, DECAY_FACTOR


def add_recent_form(df):
    df = df.sort_values(["driverId", "raceId"])
    records = []
    for driver_id, group in df.groupby("driverId"):
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            past = group.iloc[max(0, i - RECENT_RACES_WINDOW):i]
            if len(past) == 0:
                records.append(10.0)
                continue
            weights = [DECAY_FACTOR ** j for j in range(len(past)-1, -1, -1)]
            score = np.average(past["positionOrder"].fillna(20), weights=weights)
            records.append(score)
    df["recent_form"] = records
    return df


def add_dnf_rate(df):
    df = df.sort_values(["driverId", "raceId"])
    records = []
    for driver_id, group in df.groupby("driverId"):
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            past = group.iloc[max(0, i-10):i]
            if len(past) == 0:
                records.append(0.1)
                continue
            dnf = (past["statusId"] != 1).sum()
            records.append(dnf / len(past))
    df["dnf_rate"] = records
    return df


def add_quali_teammate_delta(qualifying_df, results_df):
    # qualifying already has constructorId and use it directly
    q = qualifying_df[["raceId", "driverId", "constructorId", "q3"]].copy()
    q["q3"] = pd.to_numeric(q["q3"], errors="coerce")
    
    teammate_avg = q.groupby(["raceId", "constructorId"])["q3"].transform("mean")
    q["quali_teammate_delta"] = q["q3"] - teammate_avg
    return q[["raceId", "driverId", "quali_teammate_delta"]]

def add_championship_position(results_df, races_df):
    if "year" not in results_df.columns or "round" not in results_df.columns:
        results_df = results_df.merge(
            races_df[["raceId", "year", "round"]], on="raceId", how="left"
        )

    merged = results_df.sort_values(["year", "round", "driverId"])
    records = []

    for year, year_group in merged.groupby("year"):
        cum_points = {}
        for rnd, race in year_group.groupby("round"):
            for _, row in race.iterrows():
                pts_so_far = cum_points.get(row["driverId"], 0)
                records.append({
                    "raceId": row["raceId"],
                    "driverId": row["driverId"],
                    "champ_points_before": pts_so_far
                })
            for _, row in race.iterrows():
                cum_points[row["driverId"]] = (
                    cum_points.get(row["driverId"], 0) + row["points"]
                )

    champ_df = pd.DataFrame(records)
    champ_df["champ_position"] = champ_df.groupby("raceId")["champ_points_before"].rank(
        ascending=False, method="min"
    )
    return champ_df

def add_positions_gained(df):
    # rolling average of positions gained/lost from grid to finish
    df = df.sort_values(["driverId", "raceId"])
    records = []
    for driver_id, group in df.groupby("driverId"):
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            past = group.iloc[max(0, i-5):i]
            if len(past) == 0:
                records.append(0.0)
                continue
            gained = (past["grid"] - past["positionOrder"]).mean()
            records.append(gained)
    df["avg_positions_gained"] = records
    return df

def add_quali_trend(df):
    # whether driver's qualifying position improving or worsening over last 3 races
    df = df.sort_values(["driverId", "raceId"])
    records = []
    for driver_id, group in df.groupby("driverId"):
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            past = group.iloc[max(0, i-3):i]
            if len(past) < 2:
                records.append(0.0)
                continue
            # positive = getting worse, negative = improving
            trend = past["grid"].diff().mean()
            records.append(trend if not pd.isna(trend) else 0.0)
    df["quali_trend"] = records
    return df