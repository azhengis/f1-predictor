import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import DATA_RAW


def time_to_sec(t):
    try:
        parts = str(t).strip().split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except:
        return None


def load_raw():
    races        = pd.read_csv(DATA_RAW / "races.csv")
    results      = pd.read_csv(DATA_RAW / "results.csv")
    qualifying   = pd.read_csv(DATA_RAW / "qualifying.csv")
    drivers      = pd.read_csv(DATA_RAW / "drivers.csv")
    constructors = pd.read_csv(DATA_RAW / "constructors.csv")
    circuits     = pd.read_csv(DATA_RAW / "circuits.csv")
    pit_stops    = pd.read_csv(DATA_RAW / "pit_stops.csv")

    # clean results
    results["positionOrder"] = pd.to_numeric(results["positionOrder"], errors="coerce")
    results["grid"]          = pd.to_numeric(results["grid"], errors="coerce")
    results["points"]        = pd.to_numeric(results["points"], errors="coerce")

    # clean qualifying: convert time strings to seconds
    qualifying["position"] = pd.to_numeric(qualifying["position"], errors="coerce")
    for col in ["q1", "q2", "q3"]:
        qualifying[col] = qualifying[col].apply(time_to_sec)

    return races, results, qualifying, drivers, constructors, circuits, pit_stops