"""Adapt Person 1's frozen feature tables (team schema, docs/INTERFACES.md) for the shared harness.

Person 1 writes two files (train / student_test) without a `partition` column and without an
observation table. This script
  * adds `partition`,
  * validates the table against the rules the harness relies on,
  * builds the observation table (one row per GPS point of every trip, INCLUDING points with zero
    candidates) from the raw track lengths in eda/outputs/dataset_inventory.csv, independently of
    Person 1's own observation summary,
  * prints the headline statistics so they can be compared with what the table's author reported.

    python experiments/adapt_team_table.py --train <candidates_features_train.parquet> \
        --test <candidates_features_test.parquet> --out-dir experiments/team_table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402

FEATURES = ["feat_dist_to_arc", "feat_arc_length", "feat_heading_diff", "feat_speed", "feat_time_gap", "feat_prev_dist"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=config.REPO_ROOT / "experiments" / "team_table")
    a = ap.parse_args()

    tr, te = pd.read_parquet(a.train), pd.read_parquet(a.test)
    for d in (tr, te):
        d["trip_id"] = d["trip_id"].astype(str).str.zfill(8)
    tr["partition"], te["partition"] = "train", "student_test"
    df = pd.concat([tr, te], ignore_index=True)

    inv = pd.read_csv(config.EDA_OUT / "dataset_inventory.csv", dtype={"trip_id": str})
    part_of = dict(zip(inv.trip_id, inv.partition))
    problems = []
    if set(tr.trip_id) & set(te.trip_id):
        problems.append("a trip appears in both train and student_test tables")
    if any(part_of.get(t) != "train" for t in tr.trip_id.unique()):
        problems.append("train table contains trips that are not in the train partition")
    if any(part_of.get(t) != "student_test" for t in te.trip_id.unique()):
        problems.append("test table contains trips that are not in the student_test partition")
    if tr.fold.isin(range(5)).all() is False or not tr.fold.between(0, 4).all():
        problems.append("train folds are not all in 0..4")
    if (tr.groupby("trip_id").fold.nunique() > 1).any():
        problems.append("a train trip is spread over several folds (leakage)")
    if not (te.fold == -1).all():
        problems.append("student_test rows should have fold = -1")
    if df[FEATURES].isna().any().any():
        problems.append("NaN in feature columns")
    const = [c for c in FEATURES if df[c].nunique() <= 1]
    if const:
        problems.append(f"constant feature columns: {const}")
    if (df.groupby(["trip_id", "obs_idx", "arc_id"]).size() > 1).any():
        problems.append("duplicate (trip, obs, arc) candidate rows")
    missing_trips = sorted(set(inv.trip_id) - set(df.trip_id))
    if missing_trips:
        problems.append(f"trips with no candidate rows at all: {missing_trips}")

    # observation table: every GPS point of every trip, zero-candidate points included
    grid = pd.concat([pd.DataFrame({"trip_id": t, "obs_idx": np.arange(n, dtype="int64"), "partition": p})
                      for t, n, p in zip(inv.trip_id, inv.n_gps, inv.partition)], ignore_index=True)
    cnt = df.groupby(["trip_id", "obs_idx"]).agg(n_candidates=("label", "size"), n_positive=("label", "sum")).reset_index()
    obs = grid.merge(cnt, on=["trip_id", "obs_idx"], how="left")
    obs[["n_candidates", "n_positive"]] = obs[["n_candidates", "n_positive"]].fillna(0).astype("int32")
    low = pd.read_csv(config.EDA_OUT / "trips_low_route_coverage.csv", dtype={"trip_id": str}).trip_id
    obs["trip_low_route_coverage"] = obs.trip_id.isin(set(low))
    extra = set(zip(df.trip_id, df.obs_idx)) - set(zip(grid.trip_id, grid.obs_idx))
    if extra:
        problems.append(f"{len(extra)} candidate rows reference obs_idx beyond the track length")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out_dir / "candidates_features.parquet", index=False)
    obs.to_parquet(a.out_dir / "observations.parquet", index=False)

    n = len(obs)
    print(f"candidate rows: train {len(tr):,} ({tr.trip_id.nunique()} trips)  student_test {len(te):,} ({te.trip_id.nunique()} trips)")
    print(f"observations: {n:,} (train {int((obs.partition == 'train').sum()):,}, student_test {int((obs.partition == 'student_test').sum()):,})")
    print(f"obs with 0 candidates: {int((obs.n_candidates == 0).sum()):,}")
    print(f"obs with >=1 positive: {int((obs.n_positive >= 1).sum()):,} ({(obs.n_positive >= 1).mean():.2%})  -> candidate miss rate {(obs.n_positive == 0).mean():.2%}")
    print(f"obs with exactly 1 positive: {(obs.n_positive == 1).mean():.2%};  with >=2: {(obs.n_positive >= 2).mean():.2%}")
    print(f"mean candidates per obs: {obs.n_candidates.mean():.2f};  candidate positive rate: {df.label.mean():.4f}")
    per_trip_miss = obs.groupby("trip_id").apply(lambda g: (g.n_positive == 0).mean(), include_groups=False)
    print(f"miss rate: pooled over points {(obs.n_positive == 0).mean():.2%}  vs unweighted mean of per-trip rates {per_trip_miss.mean():.2%}")
    print("rows per fold:", tr.groupby("fold").size().to_dict())
    print("feature ranges:\n" + df[FEATURES].describe().loc[["min", "50%", "max"]].round(3).to_string())
    print("\nPROBLEMS:" if problems else "\nno structural problems found")
    for p in problems:
        print("  -", p)
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
