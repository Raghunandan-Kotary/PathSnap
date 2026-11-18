"""Context features (v2) for (GPS point, candidate arc) pairs.

Error analysis of the frozen 5-feature model on the student-test trips (classifier-only, tuned XGBoost):
of the wrong picks, 48% are an ADJACENT arc (shares a node with the true arc: the GPS point is near a
junction) and 30% are the REVERSE-DIRECTION twin of the true arc. The five frozen features cannot
separate those cases: they say nothing about where along an arc the point projects, the heading is a
noisy one-step difference, and every candidate is scored without looking at the other candidates of the
same GPS point or at the neighbouring GPS points.

The features below are computed from the raw .track / .nodes files and the candidate list only (no labels,
no reference route), so the API can compute them at inference time exactly as in training:

    f2_proj_frac       where along the arc the GPS point projects (0 = start node, 1 = end node)
    f2_end_dist        distance from that projection to the nearer end of the arc (m)
    f2_cos_heading     cosine between the arc direction and the GPS heading smoothed over points i-2..i+2
                       (+1 = same direction, -1 = opposite direction, 0 when the vehicle barely moves)
    f2_ctx_dist        mean distance of the neighbouring GPS points (i-2, i-1, i+1, i+2) to the arc (m)
    f2_dist_minus_min  this candidate's distance minus the smallest candidate distance for the same point (m)
    f2_ctx_minus_min   the same for f2_ctx_dist
    f2_dist_rank       rank of the candidate's distance among the candidates of the same point (1 = nearest)
    f2_n_cand          number of candidates of the same point

    python experiments/context_features.py --table experiments/team_table/candidates_features.parquet \
        --data-dir "<folder with train/ and student_test/>" --out experiments/team_table_v2/candidates_features.parquet
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2.geo import LocalProjector  # noqa: E402

V2_FEATURES = ["f2_proj_frac", "f2_end_dist", "f2_cos_heading", "f2_ctx_dist", "f2_dist_minus_min",
               "f2_ctx_minus_min", "f2_dist_rank", "f2_n_cand"]
HEADING_HALF_WINDOW = 2
STATIONARY_M = 2.0


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t, np.sqrt(L2)


def add_context_features(cands: pd.DataFrame, track: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """cands: one trip's candidate rows with obs_idx, from_node, to_node, feat_dist_to_arc.
    track: lon, lat, time (row = obs_idx). nodes: lon, lat (row = node id). Returns cands with V2_FEATURES added."""
    out = cands.copy()
    proj = LocalProjector.from_lonlat(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    nx_, ny_ = proj.to_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    gx, gy = proj.to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    n = len(track)
    i = out["obs_idx"].to_numpy().astype(int)
    u, v = out["from_node"].to_numpy().astype(int), out["to_node"].to_numpy().astype(int)
    ax, ay, bx, by = nx_[u], ny_[u], nx_[v], ny_[v]

    d, t, L = _seg_dist(gx[i], gy[i], ax, ay, bx, by)
    out["f2_proj_frac"] = t.astype("float32")
    out["f2_end_dist"] = (np.minimum(t, 1 - t) * L).astype("float32")

    k = HEADING_HALF_WINDOW
    lo, hi = np.maximum(np.arange(n) - k, 0), np.minimum(np.arange(n) + k, n - 1)
    hx, hy = gx[hi] - gx[lo], gy[hi] - gy[lo]
    hn = np.hypot(hx, hy)
    ux, uy = bx - ax, by - ay
    un = np.maximum(np.hypot(ux, uy), 1e-9)
    cos = (hx[i] * ux + hy[i] * uy) / (np.maximum(hn[i], 1e-9) * un)
    out["f2_cos_heading"] = np.where(hn[i] >= STATIONARY_M, cos, 0.0).astype("float32")

    ctx = np.zeros(len(out))
    for off in (-2, -1, 1, 2):
        j = np.clip(i + off, 0, n - 1)
        ctx += _seg_dist(gx[j], gy[j], ax, ay, bx, by)[0]
    out["f2_ctx_dist"] = (ctx / 4).astype("float32")

    g = out.groupby("obs_idx")
    out["f2_dist_minus_min"] = (out["feat_dist_to_arc"] - g["feat_dist_to_arc"].transform("min")).astype("float32")
    out["f2_ctx_minus_min"] = (out["f2_ctx_dist"] - g["f2_ctx_dist"].transform("min")).astype("float32")
    out["f2_dist_rank"] = g["feat_dist_to_arc"].rank(method="first").astype("float32")
    out["f2_n_cand"] = g["feat_dist_to_arc"].transform("size").astype("float32")
    return out


def _trip_dir(data_dir: Path, trip: str) -> Path:
    for part in ("train", "student_test", "evaluator_test"):
        if (data_dir / part / trip).is_dir():
            return data_dir / part / trip
    raise FileNotFoundError(trip)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, type=Path, help="adapted candidate table (experiments/adapt_team_table.py)")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    df = pd.read_parquet(a.table)
    parts = []
    for trip, c in df.groupby("trip_id", sort=False):
        d = _trip_dir(a.data_dir, trip)
        track = pd.read_csv(d / f"{trip}.track", sep=r"\s+", header=None, names=["lon", "lat", "time"])
        nodes = pd.read_csv(d / f"{trip}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
        parts.append(add_context_features(c, track, nodes))
    out = pd.concat(parts).loc[df.index]  # keep the frozen table's row order exactly
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    obs = a.table.with_name("observations.parquet")
    if obs.exists():
        shutil.copy2(obs, a.out.with_name("observations.parquet"))
    print(f"wrote {a.out}: {len(out):,} rows, added {V2_FEATURES}")
    print(out[V2_FEATURES].describe().loc[["min", "50%", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
