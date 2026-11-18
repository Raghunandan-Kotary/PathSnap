"""
Batch feature table builder for Person 1.
Generates:
1. data/processed/candidates_features_train.parquet (80 training trips with 5-fold CV group assignment)
2. data/processed/candidates_features_test.parquet (10 student-testing trips)
3. experiments/results/candidate_miss_rates.csv (audit of candidate generation quality)
"""
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data.loader import load_trip
from features.extract_features import generate_trip_candidates_and_features

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def process_partition(split: str, trip_ids: list, search_radius_m: float = 60.0):
    all_dfs = []
    miss_logs = []
    
    print(f"\nProcessing {len(trip_ids)} trips for '{split}' partition...")
    t0 = time.time()
    
    for idx, tid in enumerate(trip_ids):
        t_start = time.time()
        trip_dir = RAW_DIR / split / tid
        trip = load_trip(trip_dir, tid)
        
        feats_df = generate_trip_candidates_and_features(
            track_df=trip["track"],
            nodes_df=trip["nodes"],
            arcs_df=trip["arcs"],
            route_arcs=trip["route"],
            search_radius_m=search_radius_m
        )
        
        feats_df["trip_id"] = tid
        all_dfs.append(feats_df)
        
        # Calculate miss rate for this trip
        total_obs = len(trip["track"])
        pos_obs = feats_df[feats_df["label"] == 1]["obs_idx"].nunique()
        miss_count = total_obs - pos_obs
        miss_rate = miss_count / max(total_obs, 1)
        
        elapsed = time.time() - t_start
        miss_logs.append({
            "split": split,
            "trip_id": tid,
            "observations": total_obs,
            "candidates_total": len(feats_df),
            "positives": int((feats_df["label"] == 1).sum()),
            "matched_observations": pos_obs,
            "miss_count": miss_count,
            "miss_rate_pct": round(miss_rate * 100, 2),
            "processing_seconds": round(elapsed, 2)
        })
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(trip_ids):
            print(f"  [{idx+1}/{len(trip_ids)}] {tid}: {len(feats_df)} cands | miss: {miss_rate*100:.1f}% ({elapsed:.2f}s)")
            
    combined_df = pd.concat(all_dfs, ignore_index=True)
    total_time = time.time() - t0
    print(f"Finished '{split}' partition: {len(combined_df):,} total candidate rows in {total_time:.1f}s")
    
    return combined_df, pd.DataFrame(miss_logs)

def main():
    print("=" * 80)
    print("Person 1: Feature Table Generation & Group-Aware CV Fold Assignment")
    print("=" * 80)

    # 1. Discover training trips
    train_dir = RAW_DIR / "train"
    train_trips = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    
    # 2. Discover student test trips
    test_dir = RAW_DIR / "student_test"
    test_trips = sorted([d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    
    print(f"Discovered: {len(train_trips)} train trips, {len(test_trips)} student test trips.")

    # 3. Process train partition
    train_df, train_miss = process_partition("train", train_trips, search_radius_m=60.0)

    # 4. Assign 5-Fold GroupKFold CV on trip_id
    print("\nAssigning 5-Fold GroupKFold CV on 80 training trips...")
    gkf = GroupKFold(n_splits=5)
    train_df["fold"] = -1
    
    groups = train_df["trip_id"].values
    X_dummy = np.zeros(len(train_df))
    y_dummy = train_df["label"].values
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, y_dummy, groups)):
        train_df.loc[val_idx, "fold"] = fold
        val_trips = train_df.loc[val_idx, "trip_id"].nunique()
        val_rows = len(val_idx)
        print(f"  Fold {fold}: {val_trips} trips, {val_rows:,} candidate rows")

    # 5. Process student test partition
    test_df, test_miss = process_partition("student_test", test_trips, search_radius_m=60.0)
    test_df["fold"] = -1

    # 6. Reorder columns to match INTERFACES.md strictly
    col_order = [
        "trip_id", "obs_idx", "gps_lon", "gps_lat", "gps_time",
        "arc_id", "from_node", "to_node",
        "feat_dist_to_arc", "feat_arc_length", "feat_heading_diff",
        "feat_speed", "feat_time_gap", "feat_prev_dist",
        "label", "fold"
    ]
    train_df = train_df[col_order]
    test_df = test_df[col_order]

    # 7. Save Parquet and CSV artifacts
    train_parquet = PROCESSED_DIR / "candidates_features_train.parquet"
    test_parquet = PROCESSED_DIR / "candidates_features_test.parquet"
    miss_csv = RESULTS_DIR / "candidate_miss_rates.csv"

    print("\nWriting processed datasets...")
    train_df.to_parquet(train_parquet, index=False)
    test_df.to_parquet(test_parquet, index=False)

    all_miss = pd.concat([train_miss, test_miss], ignore_index=True)
    all_miss.to_csv(miss_csv, index=False)

    print(f"  -> Saved {train_parquet} ({len(train_df):,} rows)")
    print(f"  -> Saved {test_parquet} ({len(test_df):,} rows)")
    print(f"  -> Saved {miss_csv}")
    print(f"\nAverage Candidate Miss Rate: {all_miss['miss_rate_pct'].mean():.2f}% across all {len(all_miss)} trips.")
    print("=" * 80)
    print("Feature Freeze Complete! Handing off to Person 2.")
    print("=" * 80)

if __name__ == "__main__":
    main()
