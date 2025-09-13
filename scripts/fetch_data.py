import os
from pathlib import Path
from app.services.loader import load_timeseries

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = DATA_DIR / "timeseries.csv"

def main():
    df = load_timeseries(os.getenv("CITY", "Paris"))
    df.to_csv(OUT, index=True)
    print(f"Saved {len(df)} rows to {OUT}")

if __name__ == "__main__":
    main()
