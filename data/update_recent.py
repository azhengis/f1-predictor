import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from config import DATA_RAW

BASE = "https://api.jolpi.ca/ergast/f1"

def to_sec(t):
    try:
        m, s = str(t).split(":")
        return float(m) * 60 + float(s)
    except:
        return None

def fetch_results(year):
    all_results = []
    for round_num in range(1, 30):
        try:
            url = f"{BASE}/{year}/{round_num}/results.json"
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                break
            data = r.json()
            races = data["MRData"]["RaceTable"]["Races"]
            if not races:
                break
            race = races[0]
            race_id = int(f"9{year}{round_num:02d}")
            for res in race["Results"]:
                try:
                    pos = int(res["position"])
                except:
                    pos = 20
                all_results.append({
                    "raceId":        race_id,
                    "year":          year,
                    "round":         round_num,
                    "circuitId":     race["Circuit"]["circuitId"],
                    "raceName":      race["raceName"],
                    "driverId":      res["Driver"]["driverId"],
                    "constructorId": res["Constructor"]["constructorId"],
                    "grid":          int(res.get("grid", 0)),
                    "positionOrder": pos,
                    "points":        float(res.get("points", 0)),
                    "statusId":      1 if res["status"] == "Finished" else 0,
                })
            print(f"  {year} Round {round_num}: {race['raceName']} ✓")
        except Exception as e:
            print(f"  {year} Round {round_num}: skipped ({e})")
            break
    return pd.DataFrame(all_results)

def fetch_qualifying(year):
    all_quali = []
    for round_num in range(1, 30):
        try:
            url = f"{BASE}/{year}/{round_num}/qualifying.json"
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                break
            data = r.json()
            races = data["MRData"]["RaceTable"]["Races"]
            if not races:
                break
            race = races[0]
            race_id = int(f"9{year}{round_num:02d}")
            for res in race["QualifyingResults"]:
                all_quali.append({
                    "raceId":        race_id,
                    "driverId":      res["Driver"]["driverId"],
                    "constructorId": res["Constructor"]["constructorId"],
                    "position":      int(res.get("position", 20)),
                    "q1":            to_sec(res.get("Q1", "")),
                    "q2":            to_sec(res.get("Q2", "")),
                    "q3":            to_sec(res.get("Q3", "")),
                })
        except Exception as e:
            print(f"  Quali {year} Round {round_num}: skipped ({e})")
            break
    return pd.DataFrame(all_quali)

def update():
    print("Fetching 2024 data...")
    results_2024 = fetch_results(2024)
    quali_2024   = fetch_qualifying(2024)

    print("\nFetching 2025 data...")
    results_2025 = fetch_results(2025)
    quali_2025   = fetch_qualifying(2025)

    # Load existing
    existing_results = pd.read_csv(DATA_RAW / "results.csv")
    existing_quali   = pd.read_csv(DATA_RAW / "qualifying.csv")

    # Append and deduplicate
    new_results = pd.concat([existing_results, results_2024, results_2025], ignore_index=True)
    new_quali   = pd.concat([existing_quali, quali_2024, quali_2025], ignore_index=True)
    new_results = new_results.drop_duplicates(subset=["raceId", "driverId"])
    new_quali   = new_quali.drop_duplicates(subset=["raceId", "driverId"])

    new_results.to_csv(DATA_RAW / "results.csv", index=False)
    new_quali.to_csv(DATA_RAW / "qualifying.csv", index=False)

    print(f"\n✓ Added {len(results_2024)} result rows for 2024")
    print(f"✓ Added {len(results_2025)} result rows for 2025")
    print("\nNow run:")
    print("  python features/pipeline.py")
    print("  python models/train.py")

if __name__ == "__main__":
    def update():
        print("Fetching 2024 data...")
        results_2024 = fetch_results(2024)
        quali_2024   = fetch_qualifying(2024)

        print("\nFetching 2025 data...")
        results_2025 = fetch_results(2025)
        quali_2025   = fetch_qualifying(2025)

        # Load existing CSVs
        existing_results = pd.read_csv(DATA_RAW / "results.csv")
        existing_quali   = pd.read_csv(DATA_RAW / "qualifying.csv")
        existing_races   = pd.read_csv(DATA_RAW / "races.csv")

        # Build races rows for new data
        new_race_rows = []
        for df, year in [(results_2024, 2024), (results_2025, 2025)]:
            if df.empty:
                continue
            for _, group in df.groupby(["raceId", "round", "raceName", "circuitId"]):
                pass
            for (race_id, round_num, race_name, circuit_id), group in df.groupby(
                ["raceId", "round", "raceName", "circuitId"]
            ):
                new_race_rows.append({
                    "raceId":     race_id,
                    "year":       year,
                    "round":      round_num,
                    "circuitId":  circuit_id,
                    "name":       race_name,
                    "date":       f"{year}-01-01",
                    "time":       "00:00:00",
                    "url":        "",
                })

        if new_race_rows:
            new_races_df = pd.DataFrame(new_race_rows)
            existing_races = pd.concat([existing_races, new_races_df], ignore_index=True)
            existing_races = existing_races.drop_duplicates(subset=["raceId"])
            existing_races.to_csv(DATA_RAW / "races.csv", index=False)
            print(f"\n✓ Added {len(new_race_rows)} race entries to races.csv")

        # Append results and qualifying
        new_results = pd.concat([existing_results, results_2024, results_2025], ignore_index=True)
        new_quali   = pd.concat([existing_quali, quali_2024, quali_2025], ignore_index=True)
        new_results = new_results.drop_duplicates(subset=["raceId", "driverId"])
        new_quali   = new_quali.drop_duplicates(subset=["raceId", "driverId"])

        new_results.to_csv(DATA_RAW / "results.csv", index=False)
        new_quali.to_csv(DATA_RAW / "qualifying.csv", index=False)

        print(f"✓ Added {len(results_2024)} result rows for 2024")
        print(f"✓ Added {len(results_2025)} result rows for 2025")
        print("\nNow run:")
        print("  python features/pipeline.py")
        print("  python models/train.py")