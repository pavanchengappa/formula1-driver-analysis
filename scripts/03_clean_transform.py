import os
import pandas as pd

RACE_PATH = "data/raw/race_laps_2024.csv"
QUALI_PATH = "data/raw/quali_laps_2024.csv"
OUT_PATH = "data/processed/driver_performance_2024.csv"

os.makedirs("data/processed", exist_ok=True)

race = pd.read_csv(RACE_PATH)
quali = pd.read_csv(QUALI_PATH)

def to_seconds(series):
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()

for df in [race, quali]:
    df["LapTimeSec"] = to_seconds(df["LapTime"])
    df["Sector1Sec"] = to_seconds(df["Sector1Time"])
    df["Sector2Sec"] = to_seconds(df["Sector2Time"])
    df["Sector3Sec"] = to_seconds(df["Sector3Time"])

def validate_lap_data(df, label):
    initial_rows = len(df)

    # Remove inaccurate laps
    df = df[df["IsAccurate"] == True]

    # Remove missing lap times
    df = df.dropna(subset=["LapTimeSec"])

    # Remove extreme outliers (very slow laps)
    df = df[df["LapTimeSec"] < df["LapTimeSec"].quantile(0.99)]

    removed = initial_rows - len(df)
    print(f"Data validation ({label}) removed {removed} rows")

    return df

race = validate_lap_data(race, "Race")
quali = validate_lap_data(quali, "Qualifying")

# Degradation helper only makes sense for Race laps
race["LapInStint"] = (
    race.groupby(["Season", "RaceName", "Driver", "Stint"])
    .cumcount() + 1
)
quali["LapInStint"] = None

# Compute fastest lap per event and session
race["SessionFastestLap"] = (
    race.groupby(["Season", "RaceName", "Session"])["LapTimeSec"]
    .transform("min")
)
quali["SessionFastestLap"] = (
    quali.groupby(["Season", "RaceName", "Session"])["LapTimeSec"]
    .transform("min")
)

# Normalized pace (lower = faster) for both sessions
race["NormalizedPace"] = race["LapTimeSec"] / race["SessionFastestLap"]
quali["NormalizedPace"] = quali["LapTimeSec"] / quali["SessionFastestLap"]

TRACK_TYPE_MAP = {
    "Monaco Grand Prix": "Street",
    "Singapore Grand Prix": "Street",
    "Las Vegas Grand Prix": "Street",
    "Azerbaijan Grand Prix": "Street",
    "Italian Grand Prix": "High-Speed",
    "Saudi Arabian Grand Prix": "High-Speed",
    "Belgian Grand Prix": "High-Speed",
    "Hungarian Grand Prix": "Technical",
    "Japanese Grand Prix": "Technical",
    "British Grand Prix": "Technical",
    "Dutch Grand Prix": "Technical",
}

race["TrackType"] = race["Circuit"].map(TRACK_TYPE_MAP).fillna("Mixed")
quali["TrackType"] = quali["Circuit"].map(TRACK_TYPE_MAP).fillna("Mixed")

final_df = pd.concat([race, quali], ignore_index=True)

# Keep only clean columns for Power BI
final_df = final_df[[
    "Season", "RaceName", "Circuit", "Session",
    "Driver", "Team",
    "LapNumber", "LapInStint",
    "Compound", "Stint",
    "LapTimeSec", "NormalizedPace",
    "Sector1Sec", "Sector2Sec", "Sector3Sec",
    "Position", "TrackStatus",
    "TrackType"
]]

final_df.to_csv(OUT_PATH, index=False)

print(f"\n? Final dataset saved: {OUT_PATH}")
print("Rows:", len(final_df))
print(final_df.head())
