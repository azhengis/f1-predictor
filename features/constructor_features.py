import pandas as pd

def add_team_rolling_podium_rate(results_df, window=10):
    results_df = results_df.sort_values(["constructorId","raceId"])
    results_df["is_podium"] = results_df["positionOrder"] <= 3

    records = []
    for constructor_id, group in results_df.groupby("constructorId"):
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            past = group.iloc[max(0, i-window):i]
            rate = past["is_podium"].mean() if len(past) > 0 else 0.1
            records.append(rate)

    results_df["team_podium_rate"] = records
    return results_df

def add_pit_stop_features(pit_stops_df, results_df):
    # converting both driverId columns to string
    pit_stops_df = pit_stops_df.copy()
    results_df = results_df.copy()
    pit_stops_df["driverId"] = pit_stops_df["driverId"].astype(str)
    results_df["driverId"] = results_df["driverId"].astype(str)
    pit_stops_df["raceId"] = pit_stops_df["raceId"].astype(str)
    results_df["raceId"] = results_df["raceId"].astype(str)

    pit_merged = pit_stops_df.merge(
        results_df[["raceId", "driverId", "constructorId"]].drop_duplicates(),
        on=["raceId", "driverId"], how="left"
    )
    pit_merged["milliseconds"] = pd.to_numeric(pit_merged["milliseconds"], errors="coerce")

    pit_stats = pit_merged.groupby(["raceId", "constructorId"])["milliseconds"].agg(
        pit_mean="mean", pit_std="std"
    ).reset_index()
    pit_stats["pit_std"] = pit_stats["pit_std"].fillna(0)
    pit_stats["pit_mean"] = pit_stats["pit_mean"].fillna(25000)

    results_df["raceId"] = results_df["raceId"].astype(str)
    pit_stats["raceId"] = pit_stats["raceId"].astype(str)

    return results_df.merge(pit_stats, on=["raceId", "constructorId"], how="left")

def add_constructor_avg_finish(results_df, window=10):
    """Rolling average finish position per constructor."""
    results_df = results_df.sort_values(["constructorId", "raceId"])
    records = []
    for constructor_id, group in results_df.groupby("constructorId"):
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            past = group.iloc[max(0, i-window):i]
            avg = past["positionOrder"].mean() if len(past) > 0 else 10.0
            records.append(avg)
    results_df["constructor_avg_finish"] = records
    return results_df