import os
import pandas as pd
import fastf1
from fastf1 import get_event_schedule, get_session
from tqdm import tqdm

fastf1.Cache.enable_cache("cache")

SEASON = 2024
OUT_PATH = "data/raw/quali_laps_2024.csv"

os.makedirs("data/raw", exist_ok=True)

schedule = get_event_schedule(SEASON)
schedule = schedule[schedule["RoundNumber"] > 0]

all_rows = []

for _, row in tqdm(schedule.iterrows(), total=len(schedule), desc="Fetching Qualifying sessions"):
    event_name = row["EventName"]

    try:
        session = get_session(SEASON, event_name, "Q")
        session.load()

        race_name = session.event["EventName"]
        circuit_name = race_name
        location = session.event["Location"]
        country = session.event["Country"]

        laps = session.laps.pick_quicklaps()

        keep_cols = [
            "Driver", "Team", "LapNumber",
            "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
            "Compound", "Stint", "Position",
            "TrackStatus", "IsAccurate"
        ]

        laps = laps[keep_cols].copy()

        laps["Season"] = SEASON
        laps["RaceName"] = race_name
        laps["Circuit"] = circuit_name
        laps["Location"] = location
        laps["Country"] = country
        laps["Session"] = "Qualifying"

        all_rows.append(laps)

    except Exception as e:
        print(f"[SKIPPED] {event_name} (Qualifying) -> {e}")

quali_df = pd.concat(all_rows, ignore_index=True)
quali_df.to_csv(OUT_PATH, index=False)

print(f"\n? Saved: {OUT_PATH}")
print("Rows:", len(quali_df))
