import pickle
from pathlib import Path
import os

ARTIFACTS_DIR = Path("artifacts")
PICKLE_FILES = [
    "xgboost_calibrator.pkl",
    "sequence_gru_calibrator.pkl",
    "sequence_resp_gru_calibrator.pkl",
    "xgboost_resp_calibrator.pkl",
    "ensemble_meta.pkl"
]

def resave_pickles():
    for filename in PICKLE_FILES:
        path = ARTIFACTS_DIR / filename
        if path.exists():
            print(f"Resaving {path}...")
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Re-save with same name
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"Successfully resaved {filename}")
        else:
            print(f"Skipping {filename} (not found)")

if __name__ == "__main__":
    resave_pickles()
