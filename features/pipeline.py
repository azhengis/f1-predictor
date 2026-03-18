import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data.loader import load_raw
from features.driver_features import (
    add_recent_form, add_dnf_rate,
    add_quali_teammate_delta, add_championship_position
)
from features.constructor_features import add_team_rolling_podium_rate, add_pit_stop_features
from features.track_features import add_circuit_meta, add_driver_circuit_history
from config import DATA_PROCESSED
from features.driver_features import (
    add_recent_form, add_dnf_rate,
    add_quali_teammate_delta, add_championship_position,
    add_positions_gained, add_quali_trend
)
from features.constructor_features import (
    add_team_rolling_podium_rate, add_pit_stop_features,
    add_constructor_avg_finish
)


def build_feature_matrix():
    races, results, qualifying, drivers, constructors, circuits, pit_stops = load_raw()

    # normalize all key columns to string across every dataframe
    def norm(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        return df

    results     = norm(results,     ["raceId","driverId","constructorId"])
    qualifying  = norm(qualifying,  ["raceId","driverId","constructorId"])
    pit_stops   = norm(pit_stops,   ["raceId","driverId"])
    races       = norm(races,       ["raceId","circuitId"])
    circuits    = norm(circuits,    ["circuitId"])

    # normalize year and round
    races["year"]  = pd.to_numeric(races["year"],  errors="coerce").astype("Int64")
    races["round"] = pd.to_numeric(races["round"], errors="coerce").astype("Int64")

    # drop any pre-existing year/round/circuitId from results to avoid merge conflicts
    results = results.drop(columns=[c for c in ["year","round","circuitId","raceName"] if c in results.columns])

    # merge values onto results
    results = results.merge(
        races[["raceId","year","round","circuitId"]],
        on="raceId", how="left"
    )

    print(f"After race merge: {len(results)} rows, 2025 rows: {len(results[results['year']==2025])}")

    # driver features
    results = add_recent_form(results)
    results = add_dnf_rate(results)
    results = add_positions_gained(results)
    results = add_quali_trend(results)

    # constructor features
    results = add_team_rolling_podium_rate(results)
    results = add_constructor_avg_finish(results)
    results = add_pit_stop_features(pit_stops, results)

    # championship position
    champ_df = add_championship_position(results, races)
    results = results.merge(
        champ_df[["raceId","driverId","champ_position","champ_points_before"]],
        on=["raceId","driverId"], how="left"
    )

    # circuit features 
    results = add_circuit_meta(results, circuits)
    circuit_hist = add_driver_circuit_history(results, races)
    results = results.merge(circuit_hist, on=["raceId","driverId"], how="left")

    # qualifying features 
    results_for_quali = results[["raceId","driverId","constructorId"]].drop_duplicates()

    quali_agg = qualifying.groupby(["raceId","driverId"]).agg(
        quali_position=("position","min"),
        best_quali_time=("q3","min"),
    ).reset_index()

    pole_times = quali_agg.groupby("raceId")["best_quali_time"].min().reset_index()
    pole_times.columns = ["raceId","pole_time"]
    quali_agg = quali_agg.merge(pole_times, on="raceId", how="left")
    quali_agg["gap_to_pole"] = quali_agg["best_quali_time"] - quali_agg["pole_time"]

    teammate_delta = add_quali_teammate_delta(qualifying, results_for_quali)
    quali_agg = quali_agg.merge(teammate_delta, on=["raceId","driverId"], how="left")

    results = results.merge(
        quali_agg[["raceId","driverId","quali_position","gap_to_pole","quali_teammate_delta"]],
        on=["raceId","driverId"], how="left"
    )

    # final feature selection
    feature_cols = [
        "raceId", "driverId", "constructorId", "year",
        "grid",
        "quali_position", "gap_to_pole", "quali_teammate_delta", "quali_trend",
        "recent_form", "dnf_rate", "avg_positions_gained",
        "champ_position", "champ_points_before",
        "team_podium_rate", "constructor_avg_finish", "pit_mean", "pit_std",
        "overtaking_difficulty", "driver_circuit_avg",
        "positionOrder",
    ]

    feature_df = results[feature_cols].dropna(subset=["positionOrder","quali_position"])
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    feature_df.to_csv(DATA_PROCESSED / "features.csv", index=False)
    print(f"Feature matrix saved: {len(feature_df)} rows")
    print(f"2025 rows in features: {len(feature_df[feature_df['year']==2025])}")
    return feature_df


if __name__ == "__main__":
    build_feature_matrix()