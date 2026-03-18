import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "artifacts" / "lgbm_model.pkl"
CACHE_DIR = BASE_DIR / "data" / "fastf1_cache"

os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(BASE_DIR / "models" / "artifacts", exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

RECENT_RACES_WINDOW = 5
DECAY_FACTOR = 0.8
TRAIN_YEARS = list(range(2010, 2022))
VAL_YEARS = [2022, 2023]
TEST_YEARS = [2024, 2025]