import os
import numpy as np
import pandas as pd

IN_PATH = "data/processed/driver_performance_2024.csv"
OUT_PATH = "data/processed/driver_metrics_2024.csv"

os.makedirs("data/processed", exist_ok=True)


df = pd.read_csv(IN_PATH)

# Keep valid rows for pace and consistency metrics.
base = df.dropna(subset=["Driver", "Team", "Session", "LapTimeSec", "NormalizedPace", "TrackType"]).copy()
base["Compound"] = base["Compound"].astype(str).str.upper()
base["TrackType"] = base["TrackType"].astype(str)

# Phase 2A: Driver pace ranking.
pace = (
    base.groupby("Driver", as_index=False)
    .agg(
        Team=("Team", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        AvgNormalizedPace=("NormalizedPace", "mean"),
        TotalLaps=("LapTimeSec", "size"),
    )
)
pace["PaceRank"] = pace["AvgNormalizedPace"].rank(method="min", ascending=True).astype(int)

# Teammate benchmark (driver average pace against team average pace).
team_avg = pace.groupby("Team", as_index=False).agg(TeamAvgNormalizedPace=("AvgNormalizedPace", "mean"))
pace = pace.merge(team_avg, on="Team", how="left")
pace["DeltaToTeamAvg"] = pace["AvgNormalizedPace"] - pace["TeamAvgNormalizedPace"]

# Phase 2B: Consistency analysis.
consistency = base.groupby("Driver", as_index=False).agg(ConsistencyStdSec=("LapTimeSec", "std"))
consistency["ConsistencyRank"] = consistency["ConsistencyStdSec"].rank(method="min", ascending=True).astype(int)

metrics = pace.merge(consistency, on="Driver", how="left")

# Phase 2C: Pace vs consistency quadrants.
pace_threshold = metrics["AvgNormalizedPace"].mean()
consistency_threshold = metrics["ConsistencyStdSec"].mean()


def classify_quadrant(row):
    fast = row["AvgNormalizedPace"] <= pace_threshold
    consistent = row["ConsistencyStdSec"] <= consistency_threshold
    if fast and consistent:
        return "Fast-Consistent"
    if fast and not consistent:
        return "Fast-Inconsistent"
    if not fast and consistent:
        return "Slow-Consistent"
    return "Slow-Inconsistent"


metrics["PaceConsistencyQuadrant"] = metrics.apply(classify_quadrant, axis=1)

# Phase 5: Session intelligence (Quali vs Race pace delta).
session_pace = (
    base.groupby(["Driver", "Session"], as_index=False)
    .agg(AvgNormalizedPace=("NormalizedPace", "mean"))
)

pivot = session_pace.pivot(index="Driver", columns="Session", values="AvgNormalizedPace").reset_index()
pivot = pivot.rename(columns={"Qualifying": "QualiAvgNormalizedPace", "Race": "RaceAvgNormalizedPace"})

if "QualiAvgNormalizedPace" not in pivot.columns:
    pivot["QualiAvgNormalizedPace"] = pd.NA
if "RaceAvgNormalizedPace" not in pivot.columns:
    pivot["RaceAvgNormalizedPace"] = pd.NA

pivot["RaceMinusQualiDelta"] = pivot["RaceAvgNormalizedPace"] - pivot["QualiAvgNormalizedPace"]
pivot["SundayOverperformer"] = pivot["RaceMinusQualiDelta"] < 0

metrics = metrics.merge(
    pivot[["Driver", "QualiAvgNormalizedPace", "RaceAvgNormalizedPace", "RaceMinusQualiDelta", "SundayOverperformer"]],
    on="Driver",
    how="left",
)

# Phase 3A: Tire degradation modeling (Race only).
race = base[(base["Session"] == "Race") & (base["LapInStint"].notna())].copy()
race = race.dropna(subset=["Stint", "LapInStint"]).copy()
race["LapInStint"] = pd.to_numeric(race["LapInStint"], errors="coerce")
race = race.dropna(subset=["LapInStint"])


def slope_sec_per_lap(group):
    x = group["LapInStint"].to_numpy(dtype=float)
    y = group["LapTimeSec"].to_numpy(dtype=float)
    if len(group) < 3 or np.ptp(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


stint_rows = []
for (driver, race_name, stint), g in race.groupby(["Driver", "RaceName", "Stint"]):
    stint_rows.append(
        {
            "Driver": driver,
            "RaceName": race_name,
            "Stint": stint,
            "DegradationSlopeSecPerLap": slope_sec_per_lap(g),
            "StintLaps": len(g),
        }
    )

stint_slopes = pd.DataFrame(stint_rows)
stint_slopes = stint_slopes.dropna(subset=["DegradationSlopeSecPerLap"])
stint_slopes = stint_slopes[stint_slopes["StintLaps"] >= 3]

degradation = (
    stint_slopes.groupby("Driver", as_index=False)
    .agg(
        AvgDegradationSlopeSecPerLap=("DegradationSlopeSecPerLap", "mean"),
        MedianDegradationSlopeSecPerLap=("DegradationSlopeSecPerLap", "median"),
        StintsUsedForDegradation=("DegradationSlopeSecPerLap", "size"),
    )
)
degradation["TireManagementRank"] = degradation["AvgDegradationSlopeSecPerLap"].rank(method="min", ascending=True).astype(int)

metrics = metrics.merge(degradation, on="Driver", how="left")

# Phase 3B: Compound performance (dry compounds).
dry = base[base["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
compound_pace = (
    dry.groupby(["Driver", "Compound"], as_index=False)
    .agg(
        AvgCompoundNormalizedPace=("NormalizedPace", "mean"),
        CompoundLaps=("NormalizedPace", "size"),
    )
)

compound_pivot = compound_pace.pivot(index="Driver", columns="Compound", values="AvgCompoundNormalizedPace").reset_index()
compound_pivot = compound_pivot.rename(
    columns={
        "SOFT": "AvgNormPaceSoft",
        "MEDIUM": "AvgNormPaceMedium",
        "HARD": "AvgNormPaceHard",
    }
)

lap_pivot = compound_pace.pivot(index="Driver", columns="Compound", values="CompoundLaps").reset_index()
lap_pivot = lap_pivot.rename(
    columns={
        "SOFT": "SoftLaps",
        "MEDIUM": "MediumLaps",
        "HARD": "HardLaps",
    }
)

compound_metrics = compound_pivot.merge(lap_pivot, on="Driver", how="left")


def best_dry_compound(row):
    options = {
        "SOFT": row.get("AvgNormPaceSoft"),
        "MEDIUM": row.get("AvgNormPaceMedium"),
        "HARD": row.get("AvgNormPaceHard"),
    }
    valid = {k: v for k, v in options.items() if pd.notna(v)}
    if not valid:
        return pd.NA
    return min(valid, key=valid.get)


compound_metrics["BestDryCompound"] = compound_metrics.apply(best_dry_compound, axis=1)
metrics = metrics.merge(compound_metrics, on="Driver", how="left")

# Phase 4A: Track-type sensitivity (pace and consistency by track type).
track_stats = (
    base.groupby(["Driver", "TrackType"], as_index=False)
    .agg(
        TrackTypeAvgNormPace=("NormalizedPace", "mean"),
        TrackTypeConsistencyStdSec=("LapTimeSec", "std"),
        TrackTypeLaps=("LapTimeSec", "size"),
    )
)

pace_by_track = track_stats.pivot(index="Driver", columns="TrackType", values="TrackTypeAvgNormPace").reset_index()
pace_by_track = pace_by_track.rename(
    columns={
        "Street": "AvgNormPaceStreet",
        "Technical": "AvgNormPaceTechnical",
        "High-Speed": "AvgNormPaceHighSpeed",
        "Mixed": "AvgNormPaceMixed",
    }
)

cons_by_track = track_stats.pivot(index="Driver", columns="TrackType", values="TrackTypeConsistencyStdSec").reset_index()
cons_by_track = cons_by_track.rename(
    columns={
        "Street": "ConsistencyStdStreet",
        "Technical": "ConsistencyStdTechnical",
        "High-Speed": "ConsistencyStdHighSpeed",
        "Mixed": "ConsistencyStdMixed",
    }
)

laps_by_track = track_stats.pivot(index="Driver", columns="TrackType", values="TrackTypeLaps").reset_index()
laps_by_track = laps_by_track.rename(
    columns={
        "Street": "LapsStreet",
        "Technical": "LapsTechnical",
        "High-Speed": "LapsHighSpeed",
        "Mixed": "LapsMixed",
    }
)

track_features = pace_by_track.merge(cons_by_track, on="Driver", how="outer").merge(laps_by_track, on="Driver", how="outer")


def best_track_type(row):
    options = {
        "Street": row.get("AvgNormPaceStreet"),
        "Technical": row.get("AvgNormPaceTechnical"),
        "High-Speed": row.get("AvgNormPaceHighSpeed"),
        "Mixed": row.get("AvgNormPaceMixed"),
    }
    valid = {k: v for k, v in options.items() if pd.notna(v)}
    if not valid:
        return pd.NA
    return min(valid, key=valid.get)


track_features["SpecialistTrackType"] = track_features.apply(best_track_type, axis=1)

metrics = metrics.merge(track_features, on="Driver", how="left")

# Phase 4B: Teammate comparison by track type.
driver_track_pace = (
    base.groupby(["Driver", "Team", "TrackType"], as_index=False)
    .agg(DriverTrackAvgNormPace=("NormalizedPace", "mean"))
)
team_track_pace = (
    base.groupby(["Team", "TrackType"], as_index=False)
    .agg(TeamTrackAvgNormPace=("NormalizedPace", "mean"))
)

teammate_track_delta = driver_track_pace.merge(team_track_pace, on=["Team", "TrackType"], how="left")
teammate_track_delta["DeltaToTeamByTrackType"] = (
    teammate_track_delta["DriverTrackAvgNormPace"] - teammate_track_delta["TeamTrackAvgNormPace"]
)

delta_pivot = teammate_track_delta.pivot(index="Driver", columns="TrackType", values="DeltaToTeamByTrackType").reset_index()
delta_pivot = delta_pivot.rename(
    columns={
        "Street": "DeltaToTeamStreet",
        "Technical": "DeltaToTeamTechnical",
        "High-Speed": "DeltaToTeamHighSpeed",
        "Mixed": "DeltaToTeamMixed",
    }
)

# Lower absolute value means more car-driven; larger positive/negative means more driver effect.
driver_effect = (
    teammate_track_delta.groupby("Driver", as_index=False)
    .agg(AvgAbsTrackTypeTeamDelta=("DeltaToTeamByTrackType", lambda s: float(np.mean(np.abs(s)))))
)

delta_pivot = delta_pivot.merge(driver_effect, on="Driver", how="left")
metrics = metrics.merge(delta_pivot, on="Driver", how="left")

# Optional teammate comparison field requested in Phase 2A.
metrics["TeamMateDelta"] = metrics["DeltaToTeamAvg"]

# Round for BI readability.
round_cols = [
    "AvgNormalizedPace",
    "TeamAvgNormalizedPace",
    "DeltaToTeamAvg",
    "ConsistencyStdSec",
    "QualiAvgNormalizedPace",
    "RaceAvgNormalizedPace",
    "RaceMinusQualiDelta",
    "TeamMateDelta",
    "AvgDegradationSlopeSecPerLap",
    "MedianDegradationSlopeSecPerLap",
    "AvgNormPaceSoft",
    "AvgNormPaceMedium",
    "AvgNormPaceHard",
    "AvgNormPaceStreet",
    "AvgNormPaceTechnical",
    "AvgNormPaceHighSpeed",
    "AvgNormPaceMixed",
    "ConsistencyStdStreet",
    "ConsistencyStdTechnical",
    "ConsistencyStdHighSpeed",
    "ConsistencyStdMixed",
    "DeltaToTeamStreet",
    "DeltaToTeamTechnical",
    "DeltaToTeamHighSpeed",
    "DeltaToTeamMixed",
    "AvgAbsTrackTypeTeamDelta",
]
for c in round_cols:
    if c in metrics.columns:
        metrics[c] = metrics[c].astype(float).round(6)

int_cols = [
    "TotalLaps",
    "PaceRank",
    "ConsistencyRank",
    "StintsUsedForDegradation",
    "TireManagementRank",
    "SoftLaps",
    "MediumLaps",
    "HardLaps",
    "LapsStreet",
    "LapsTechnical",
    "LapsHighSpeed",
    "LapsMixed",
]
for c in int_cols:
    if c in metrics.columns:
        metrics[c] = metrics[c].astype("Int64")

metrics = metrics.sort_values(["PaceRank", "Driver"]).reset_index(drop=True)
metrics.to_csv(OUT_PATH, index=False)

print(f"Saved: {OUT_PATH}")
print("Rows:", len(metrics))
print("Columns:", len(metrics.columns))
print("NaN counts (top 15):")
print(metrics.isna().sum().sort_values(ascending=False).head(15).to_dict())
print("SpecialistTrackType counts:")
print(metrics["SpecialistTrackType"].value_counts(dropna=False).to_dict())
