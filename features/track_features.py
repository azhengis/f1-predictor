import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

CIRCUIT_META = {
    "bahrain":        {"track_type": "permanent", "overtaking_difficulty": 2},
    "jeddah":         {"track_type": "street",    "overtaking_difficulty": 4},
    "albert_park":    {"track_type": "street",    "overtaking_difficulty": 3},
    "baku":           {"track_type": "street",    "overtaking_difficulty": 2},
    "miami":          {"track_type": "street",    "overtaking_difficulty": 3},
    "monaco":         {"track_type": "street",    "overtaking_difficulty": 5},
    "catalunya":      {"track_type": "permanent", "overtaking_difficulty": 4},
    "villeneuve":     {"track_type": "permanent", "overtaking_difficulty": 2},
    "red_bull_ring":  {"track_type": "permanent", "overtaking_difficulty": 2},
    "silverstone":    {"track_type": "permanent", "overtaking_difficulty": 2},
    "hungaroring":    {"track_type": "permanent", "overtaking_difficulty": 4},
    "spa":            {"track_type": "permanent", "overtaking_difficulty": 1},
    "zandvoort":      {"track_type": "permanent", "overtaking_difficulty": 4},
    "monza":          {"track_type": "permanent", "overtaking_difficulty": 1},
    "marina_bay":     {"track_type": "street",    "overtaking_difficulty": 4},
    "suzuka":         {"track_type": "permanent", "overtaking_difficulty": 3},
    "losail":         {"track_type": "permanent", "overtaking_difficulty": 2},
    "americas":       {"track_type": "permanent", "overtaking_difficulty": 2},
    "rodriguez":      {"track_type": "permanent", "overtaking_difficulty": 3},
    "interlagos":     {"track_type": "permanent", "overtaking_difficulty": 2},
    "vegas":          {"track_type": "street",    "overtaking_difficulty": 2},
    "yas_marina":     {"track_type": "permanent", "overtaking_difficulty": 3},
}

def add_circuit_meta(df, circuits_df):
    circuits_df = circuits_df[["circuitId", "circuitRef"]].copy()
    circuits_df["track_type"] = circuits_df["circuitRef"].map(
        lambda x: CIRCUIT_META.get(x, {}).get("track_type", "permanent")
    )
    circuits_df["overtaking_difficulty"] = circuits_df["circuitRef"].map(
        lambda x: CIRCUIT_META.get(x, {}).get("overtaking_difficulty", 3)
    )
    return df.merge(
        circuits_df[["circuitId", "track_type", "overtaking_difficulty"]],
        on="circuitId", how="left"
    )

def add_driver_circuit_history(results_df, races_df):
    # historical avg finish position per driver and circuit
    # merge circuitId and year if not present
    cols_needed = ["raceId", "circuitId", "year"]
    merge_cols = [c for c in cols_needed if c not in results_df.columns]
    if merge_cols:
        results_df = results_df.merge(
            races_df[["raceId", "circuitId", "year"]],
            on="raceId", how="left"
        )

    merged = results_df.sort_values(["driverId", "year", "raceId"])

    records = []
    for (driver_id, circuit_id), group in merged.groupby(["driverId", "circuitId"]):
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            past = group.iloc[:i]
            avg = past["positionOrder"].mean() if len(past) > 0 else 10.0
            records.append({
                "raceId": row["raceId"],
                "driverId": row["driverId"],
                "driver_circuit_avg": avg
            })

    return pd.DataFrame(records)