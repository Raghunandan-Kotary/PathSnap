"""Shared EDA checklist (Person 2's own run).

Covers: inventory, GPS quality, extents (projected, not degree-Euclidean), road-network size
and connectivity, route/arc relationship, and reference-route limits.

Every row/plot states its partition. Output goes to eda/outputs/.

    python eda/run_eda.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.geo import LocalProjector  # noqa: E402


def read_ws(path: Path, names: list[str], dtypes: dict) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtypes)


def trip_ids(partition: str) -> list[str]:
    return sorted(p.name for p in config.PARTITION_DIRS[partition].iterdir() if p.is_dir())


def analyse_trip(partition: str, trip: str) -> dict:
    d = config.PARTITION_DIRS[partition] / trip
    files = {ext: d / f"{trip}.{ext}" for ext in ("track", "nodes", "arcs", "route")}
    row: dict = {"partition": partition, "trip_id": trip}
    for ext, p in files.items():
        row[f"{ext}_exists"] = p.exists()
        row[f"{ext}_bytes"] = p.stat().st_size if p.exists() else 0

    trk = read_ws(files["track"], ["lon", "lat", "t"], {"lon": "float64", "lat": "float64", "t": "float64"})
    nodes = read_ws(files["nodes"], ["lon", "lat"], {"lon": "float64", "lat": "float64"})
    arcs = read_ws(files["arcs"], ["u", "v"], {"u": "int64", "v": "int64"})
    route = pd.read_csv(files["route"], header=None, names=["arc"], dtype={"arc": "int64"})["arc"].to_numpy()

    # ---------------- GPS track ----------------
    row["n_gps"] = len(trk)
    row["gps_nan"] = int(trk.isna().any(axis=1).sum())
    row["gps_lon_min"], row["gps_lon_max"] = trk.lon.min(), trk.lon.max()
    row["gps_lat_min"], row["gps_lat_max"] = trk.lat.min(), trk.lat.max()
    row["gps_coords_in_valid_range"] = bool(trk.lon.between(-180, 180).all() and trk.lat.between(-90, 90).all())
    dt = np.diff(trk.t.to_numpy())
    row["t_first"], row["t_last"] = trk.t.iloc[0], trk.t.iloc[-1]
    row["dt_is_1s_frac"] = float(np.mean(np.isclose(dt, 1.0)))
    row["dt_non_positive"] = int((dt <= 0).sum())
    row["dt_gt_1s"] = int((dt > 1.0 + 1e-9).sum())
    row["dt_max_s"] = float(dt.max()) if len(dt) else np.nan
    row["dt_gap_gt_5s"] = int((dt > 5).sum())
    proj = LocalProjector.from_lonlat(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    row["proj_lon0"], row["proj_lat0"] = proj.lon0, proj.lat0
    gx, gy = proj.to_xy(trk.lon.to_numpy(), trk.lat.to_numpy())
    step = np.hypot(np.diff(gx), np.diff(gy))
    speed = step / np.where(dt > 0, dt, np.nan)
    row["gps_extent_x_km"] = (gx.max() - gx.min()) / 1000
    row["gps_extent_y_km"] = (gy.max() - gy.min()) / 1000
    row["gps_path_len_km"] = step.sum() / 1000
    row["step_median_m"] = float(np.median(step))
    row["step_p99_m"] = float(np.percentile(step, 99))
    row["step_max_m"] = float(step.max())
    row["speed_max_ms"] = float(np.nanmax(speed))
    row["jump_gt_50m_per_step"] = int((step > 50).sum())
    row["duplicate_consecutive_pts"] = int(((np.diff(trk.lon) == 0) & (np.diff(trk.lat) == 0)).sum())

    # ---------------- road network ----------------
    n_nodes, n_arcs = len(nodes), len(arcs)
    row["n_nodes"], row["n_arcs"] = n_nodes, n_arcs
    row["nodes_nan"] = int(nodes.isna().any(axis=1).sum())
    row["node_lon_min"], row["node_lon_max"] = nodes.lon.min(), nodes.lon.max()
    row["node_lat_min"], row["node_lat_max"] = nodes.lat.min(), nodes.lat.max()
    row["arc_ids_in_range"] = bool((arcs.u.between(0, n_nodes - 1) & arcs.v.between(0, n_nodes - 1)).all())
    row["self_loops"] = int((arcs.u == arcs.v).sum())
    row["duplicate_arcs"] = int(arcs.duplicated().sum())
    pair = set(zip(arcs.u.tolist(), arcs.v.tolist()))
    row["arcs_with_reverse_frac"] = float(np.mean([(v, u) in pair for u, v in zip(arcs.u, arcs.v)]))
    deg_nodes = np.zeros(n_nodes, dtype=bool)
    deg_nodes[arcs.u.to_numpy()] = True
    deg_nodes[arcs.v.to_numpy()] = True
    row["isolated_nodes"] = int((~deg_nodes).sum())
    g = coo_matrix((np.ones(n_arcs, dtype=np.int8), (arcs.u, arcs.v)), shape=(n_nodes, n_nodes)).tocsr()
    n_weak, lab_weak = connected_components(g, directed=True, connection="weak")
    n_strong, lab_strong = connected_components(g, directed=True, connection="strong")
    sizes_w = np.bincount(lab_weak)
    sizes_s = np.bincount(lab_strong)
    row["weak_components"], row["weak_largest_frac"] = n_weak, float(sizes_w.max() / n_nodes)
    row["strong_components"], row["strong_largest_frac"] = n_strong, float(sizes_s.max() / n_nodes)
    row["gps_inside_network_bbox"] = bool(
        trk.lon.between(nodes.lon.min(), nodes.lon.max()).all() and trk.lat.between(nodes.lat.min(), nodes.lat.max()).all()
    )
    nx_, ny_ = proj.to_xy(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    arc_len = np.hypot(nx_[arcs.u] - nx_[arcs.v], ny_[arcs.u] - ny_[arcs.v])
    row["arc_len_median_m"], row["arc_len_max_m"] = float(np.median(arc_len)), float(arc_len.max())
    row["network_extent_x_km"] = (nx_.max() - nx_.min()) / 1000
    row["network_extent_y_km"] = (ny_.max() - ny_.min()) / 1000

    # ---------------- reference route ----------------
    row["n_route_arcs"] = len(route)
    row["route_ids_in_range"] = bool(((route >= 0) & (route < n_arcs)).all())
    row["route_unique_arcs"] = int(len(np.unique(route)))
    row["route_repeated_arcs"] = int(len(route) - len(np.unique(route)))
    if row["route_ids_in_range"] and len(route) > 1:
        u_r, v_r = arcs.u.to_numpy()[route], arcs.v.to_numpy()[route]
        connected = v_r[:-1] == u_r[1:]
        row["route_consecutive_connected_frac"] = float(connected.mean())
        row["route_breaks"] = int((~connected).sum())
        row["route_len_km"] = float(arc_len[route].sum() / 1000)
        lab_r = lab_weak[u_r]
        row["route_in_one_weak_component"] = bool((lab_r == lab_r[0]).all())
        row["route_in_largest_weak_component"] = bool((lab_r == np.argmax(sizes_w)).all())
        # How far is each GPS point from the reference route polyline? (noise + route limitation)
        import shapely

        segs = shapely.linestrings(
            np.stack([np.stack([nx_[u_r], ny_[u_r]], 1), np.stack([nx_[v_r], ny_[v_r]], 1)], 1)
        )
        mls = shapely.multilinestrings(segs)  # one MultiLineString of the whole reference route
        pts = shapely.points(gx, gy)
        dist = shapely.distance(pts, mls)
        row["gps_to_route_median_m"] = float(np.median(dist))
        row["gps_to_route_p95_m"] = float(np.percentile(dist, 95))
        row["gps_to_route_max_m"] = float(dist.max())
        row["gps_gt_30m_from_route"] = int((dist > 30).sum())
        # route start / end vs first / last GPS
        row["first_gps_to_route_start_m"] = float(np.hypot(gx[0] - nx_[u_r[0]], gy[0] - ny_[u_r[0]]))
        row["last_gps_to_route_end_m"] = float(np.hypot(gx[-1] - nx_[v_r[-1]], gy[-1] - ny_[v_r[-1]]))
    return row


def main() -> None:
    config.EDA_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for part in ("train", "student_test"):
        for trip in trip_ids(part):
            rows.append(analyse_trip(part, trip))
            print(f"[{part}] {trip} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = config.EDA_OUT / "dataset_inventory.csv"
    df.to_csv(out, index=False)
    print("wrote", out, df.shape)


if __name__ == "__main__":
    main()
