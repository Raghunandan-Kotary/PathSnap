"""
Generates observation-level table for Person 2 and evaluators:
trip_id, split, obs_idx, n_candidates, n_positive, gps_lon, gps_lat, gps_time
Crucially includes all GPS observations, including any points with zero candidates.
"""
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "experiments" / "results"
RAW_DIR = BASE_DIR / "data" / "raw"

def build_observation_table():
    print("=" * 80)
    print("Building Observation-Level Audit Table (Person 1 -> Person 2)")
    print("=" * 80)

    train_cands_path = PROCESSED_DIR / "candidates_features_train.parquet"
    test_cands_path = PROCESSED_DIR / "candidates_features_test.parquet"

    print("Loading candidate feature tables...")
    train_cands = pd.read_parquet(train_cands_path, columns=["trip_id", "obs_idx", "label"])
    test_cands = pd.read_parquet(test_cands_path, columns=["trip_id", "obs_idx", "label"])

    obs_records = []

    for split, cands_df in [("train", train_cands), ("student_test", test_cands)]:
        split_dir = RAW_DIR / split
        trip_ids = sorted([d.name for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
        
        # Group candidates by (trip_id, obs_idx)
        grouped = cands_df.groupby(["trip_id", "obs_idx"]).agg(
            n_candidates=("label", "count"),
            n_positive=("label", "sum")
        ).reset_index()
        
        # Index for fast lookup: (trip_id, obs_idx) -> (n_cands, n_pos)
        lookup = {}
        for _, row in grouped.iterrows():
            lookup[(str(row["trip_id"]), int(row["obs_idx"]))] = (int(row["n_candidates"]), int(row["n_positive"]))

        print(f"Scanning raw {split} tracks ({len(trip_ids)} trips) to capture all points (including zero-candidate observations)...")
        for tid in trip_ids:
            track_file = split_dir / tid / f"{tid}.track"
            track = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "time"])
            
            for obs_idx, row in track.iterrows():
                key = (tid, obs_idx)
                n_cands, n_pos = lookup.get(key, (0, 0))
                
                obs_records.append({
                    "split": split,
                    "trip_id": tid,
                    "obs_idx": obs_idx,
                    "n_candidates": n_cands,
                    "n_positive": n_pos,
                    "gps_lon": row["lon"],
                    "gps_lat": row["lat"],
                    "gps_time": row["time"]
                })

    obs_df = pd.DataFrame(obs_records)
    print(f"\nGenerated observation table: {len(obs_df):,} total GPS points across 90 trips.")
    print(f"Observations with 0 candidates: {(obs_df['n_candidates'] == 0).sum():,}")
    print(f"Observations with >=1 positive: {(obs_df['n_positive'] >= 1).sum():,} ({((obs_df['n_positive'] >= 1).mean()*100):.2f}%)")
    print(f"Average candidates per point:   {obs_df['n_candidates'].mean():.2f}")

    # Save to Parquet and CSV
    train_obs = obs_df[obs_df["split"] == "train"].drop(columns=["split"])
    test_obs = obs_df[obs_df["split"] == "student_test"].drop(columns=["split"])

    train_out = PROCESSED_DIR / "observation_summary_train.parquet"
    test_out = PROCESSED_DIR / "observation_summary_test.parquet"
    csv_out = RESULTS_DIR / "observation_level_summary.csv"

    train_obs.to_parquet(train_out, index=False)
    test_obs.to_parquet(test_out, index=False)
    obs_df.to_csv(csv_out, index=False)

    print(f"\nSaved:")
    print(f"  -> {train_out} ({len(train_obs):,} observations)")
    print(f"  -> {test_out} ({len(test_obs):,} observations)")
    print(f"  -> {csv_out} ({len(obs_df):,} total rows)")
    print("=" * 80)

if __name__ == "__main__":
    build_observation_table()
