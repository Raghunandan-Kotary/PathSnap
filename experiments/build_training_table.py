"""
Builds the frozen candidate + feature table that every model trains on,
per docs/INTERFACES.md Interface 1 (owned by Person 1 -- this script exists
here because nothing built it yet and Person 3's training is blocked
without it; Person 1 should take over or confirm it).

For each trip: loads raw files, generates labeled candidates (route_arcs
passed in so `label` is set), engineers features, and assigns trip-grouped
CV folds (0-4) over the 80 training trips so no trip's observations leak
across folds. The 10 student_test trips also get labeled candidates (their
ground truth is allowed for student-side comparison/tuning -- only the 10
evaluator_test trips keep ground truth server-side only) but no fold
column, since they are not used for cross-validation.

Run from the repo root:
    python experiments/build_training_table.py

Writes:
    data/processed/candidates_features_train.csv
    data/processed/candidates_features_student_test.csv
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import GroupKFold

from data.loader import load_arcs, load_nodes, load_route, load_track
from features.candidate_generator import generate_candidates
from features.feature_engineering import engineer_candidate_features

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed"
N_FOLDS = 5
SEARCH_RADIUS_M = 60.0


def build_trip_table(trip_dir: Path, trip_id: str) -> pd.DataFrame:
    track_df = load_track(trip_dir / f"{trip_id}.track")
    nodes_df = load_nodes(trip_dir / f"{trip_id}.nodes")
    arcs_df = load_arcs(trip_dir / f"{trip_id}.arcs")
    route = load_route(trip_dir / f"{trip_id}.route")

    cand_result = generate_candidates(track_df, nodes_df, arcs_df, route_arcs=route, radius_m=SEARCH_RADIUS_M)
    candidates_df = cand_result["candidates_df"]
    if candidates_df.empty:
        return candidates_df

    features_df = engineer_candidate_features(candidates_df, nodes_df, arcs_df)
    features_df["trip_id"] = trip_id
    return features_df


def build_split(split: str) -> pd.DataFrame:
    split_dir = RAW_DIR / split
    tables = []
    t0 = time.time()
    for trip_dir in sorted(split_dir.iterdir()):
        if not trip_dir.is_dir():
            continue
        trip_id = trip_dir.name
        df = build_trip_table(trip_dir, trip_id)
        if not df.empty:
            tables.append(df)
        else:
            print(f"  WARNING: no candidates at all for {split}/{trip_id}")
    print(f"{split}: {len(tables)} trips, {time.time() - t0:.1f}s")
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Trip-grouped 5-fold assignment: every observation of a trip lands in the same fold."""
    trip_ids = df["trip_id"].unique()
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_of_trip = {}
    for fold, (_, val_trip_idx) in enumerate(gkf.split(trip_ids, groups=trip_ids)):
        for i in val_trip_idx:
            fold_of_trip[trip_ids[i]] = fold
    df["fold"] = df["trip_id"].map(fold_of_trip)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = build_split("train")
    train_df = assign_folds(train_df)
    train_out = OUT_DIR / "candidates_features_train.csv"
    train_df.to_csv(train_out, index=False)
    print(f"Wrote {len(train_df)} rows, {train_df['trip_id'].nunique()} trips -> {train_out}")

    test_df = build_split("student_test")
    test_out = OUT_DIR / "candidates_features_student_test.csv"
    test_df.to_csv(test_out, index=False)
    print(f"Wrote {len(test_df)} rows, {test_df['trip_id'].nunique()} trips -> {test_out}")


if __name__ == "__main__":
    main()
