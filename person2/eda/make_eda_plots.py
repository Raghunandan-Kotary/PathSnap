"""EDA plots + Folium maps from eda/outputs/dataset_inventory.csv and the raw files.

    python eda/make_eda_plots.py                 # all plots + default maps
    python eda/make_eda_plots.py --map train 00000003

Partition is printed in every figure title / map caption.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.geo import LocalProjector  # noqa: E402

OUT = config.EDA_OUT


def load_inventory() -> pd.DataFrame:
    df = pd.read_csv(OUT / "dataset_inventory.csv", dtype={"trip_id": str})
    df["gps_within_30m_of_route_frac"] = 1 - df.gps_gt_30m_from_route / df.n_gps
    return df


def plots(df: pd.DataFrame) -> None:
    tr, st = df[df.partition == "train"], df[df.partition == "student_test"]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].hist(np.log10(tr.n_gps), bins=20, alpha=0.8, label="train (80)")
    ax[0, 0].hist(np.log10(st.n_gps), bins=20, alpha=0.8, label="student_test (10)")
    ax[0, 0].set(xlabel="log10(GPS points per trip)", ylabel="trips", title="GPS points per trip")
    ax[0, 0].legend()

    ax[0, 1].hist(tr.dt_is_1s_frac, bins=20, alpha=0.8, label="train")
    ax[0, 1].hist(st.dt_is_1s_frac, bins=20, alpha=0.8, label="student_test")
    ax[0, 1].set(xlabel="fraction of steps with dt == 1 s", ylabel="trips", title="Sampling regularity per trip")
    ax[0, 1].legend()

    ax[0, 2].scatter(tr.dt_max_s, tr.dt_gap_gt_5s, s=18, label="train")
    ax[0, 2].scatter(st.dt_max_s, st.dt_gap_gt_5s, s=28, marker="s", label="student_test")
    ax[0, 2].set(xscale="log", xlabel="largest time gap (s, log)", ylabel="# steps with dt > 5 s", title="Time gaps")
    ax[0, 2].legend()

    ax[1, 0].scatter(tr.n_nodes, tr.n_arcs, s=18, label="train")
    ax[1, 0].scatter(st.n_nodes, st.n_arcs, s=28, marker="s", label="student_test")
    ax[1, 0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road network size per trip")
    ax[1, 0].legend()

    ax[1, 1].scatter(tr.weak_components, tr.weak_largest_frac, s=18, label="train")
    ax[1, 1].scatter(st.weak_components, st.weak_largest_frac, s=28, marker="s", label="student_test")
    ax[1, 1].set(xscale="log", xlabel="weakly-connected components", ylabel="largest component / nodes",
                 title="Network connectivity (no isolated nodes in any trip)")
    ax[1, 1].legend()

    ax[1, 2].scatter(tr.gps_path_len_km, tr.route_len_km, s=18, label="train")
    ax[1, 2].scatter(st.gps_path_len_km, st.route_len_km, s=28, marker="s", label="student_test")
    lim = [0, max(df.gps_path_len_km.max(), df.route_len_km.max())]
    ax[1, 2].plot(lim, lim, "k--", lw=0.8, label="route = GPS path")
    ax[1, 2].set(xlabel="GPS path length (km)", ylabel="reference route length (km)",
                 title="Reference route vs GPS path length")
    ax[1, 2].legend()
    fig.suptitle("PathSnap EDA: dataset inventory (train = 80 trips, student_test = 10 trips)")
    fig.tight_layout()
    fig.savefig(OUT / "eda_overview.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(tr.gps_to_route_median_m, bins=30, alpha=0.8, label="train")
    ax[0].hist(st.gps_to_route_median_m, bins=30, alpha=0.8, label="student_test")
    ax[0].set(xscale="linear", xlabel="median GPS-to-reference-route distance per trip (m)", ylabel="trips",
              title="GPS noise relative to the reference route")
    ax[0].legend()
    ax[1].hist(tr.gps_within_30m_of_route_frac, bins=20, alpha=0.8, label="train")
    ax[1].hist(st.gps_within_30m_of_route_frac, bins=20, alpha=0.8, label="student_test")
    ax[1].set(xlabel="fraction of GPS points within 30 m of the reference route", ylabel="trips",
              title="Trips whose reference route does not cover the GPS trace")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "eda_route_coverage.png", dpi=130)
    plt.close(fig)

    low = df[df.gps_within_30m_of_route_frac < 0.9].sort_values("gps_within_30m_of_route_frac")
    low[["partition", "trip_id", "n_gps", "gps_to_route_median_m", "gps_within_30m_of_route_frac",
         "gps_path_len_km", "route_len_km"]].to_csv(OUT / "trips_low_route_coverage.csv", index=False)
    print("trips with <90% of GPS within 30 m of route:", len(low))


def folium_map(partition: str, trip: str, buffer_m: float = 400.0, max_gps_markers: int = 2500) -> Path:
    import folium

    d = config.PARTITION_DIRS[partition] / trip
    trk = pd.read_csv(d / f"{trip}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(d / f"{trip}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(d / f"{trip}.arcs", sep=r"\s+", header=None, names=["u", "v"])
    route = pd.read_csv(d / f"{trip}.route", header=None, names=["arc"])["arc"].to_numpy()

    proj = LocalProjector.from_lonlat(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    nx_, ny_ = proj.to_xy(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    gx, gy = proj.to_xy(trk.lon.to_numpy(), trk.lat.to_numpy())
    x0, x1, y0, y1 = gx.min() - buffer_m, gx.max() + buffer_m, gy.min() - buffer_m, gy.max() + buffer_m
    inside = (nx_ >= x0) & (nx_ <= x1) & (ny_ >= y0) & (ny_ <= y1)
    keep = inside[arcs.u.to_numpy()] & inside[arcs.v.to_numpy()]
    a = arcs[keep]

    def seg(u, v):
        return [[nodes.lat[u], nodes.lon[u]], [nodes.lat[v], nodes.lon[v]]]

    m = folium.Map(location=[trk.lat.mean(), trk.lon.mean()], zoom_start=13, tiles="OpenStreetMap")
    m.fit_bounds([[trk.lat.min(), trk.lon.min()], [trk.lat.max(), trk.lon.max()]])

    fg_net = folium.FeatureGroup(name=f"Road network ({len(a):,} arcs within {int(buffer_m)} m of GPS)", show=True)
    folium.PolyLine([seg(u, v) for u, v in zip(a.u.to_numpy(), a.v.to_numpy())], color="#888888", weight=1,
                    opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)

    fg_route = folium.FeatureGroup(name=f"Reference route ({len(route)} arcs)", show=True)
    folium.PolyLine([seg(arcs.u[r], arcs.v[r]) for r in route], color="#1a9850", weight=4, opacity=0.9).add_to(fg_route)
    fg_route.add_to(m)

    fg_gps = folium.FeatureGroup(name=f"Raw GPS ({len(trk):,} points)", show=True)
    idx = np.linspace(0, len(trk) - 1, min(len(trk), max_gps_markers)).astype(int)
    for i in idx:
        folium.CircleMarker([trk.lat[i], trk.lon[i]], radius=2, color="#2166ac", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"obs {i}  t={trk.t[i]:.0f}s").add_to(fg_gps)
    fg_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
        padding: 8px 10px; border: 1px solid #888; font: 12px sans-serif;">
        <b>PathSnap EDA &ndash; {partition} trip {trip}</b><br>
        <span style="color:#2166ac">&#9679;</span> raw GPS<br>
        <span style="color:#1a9850">&#9644;</span> reference route<br>
        <span style="color:#888">&#9644;</span> road network (clipped)<br>
        (predicted route layer added by the decoder maps)</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    out = OUT / f"folium_{partition}_{trip}.html"
    m.save(out)
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs=2, metavar=("PARTITION", "TRIP"))
    args = ap.parse_args()
    if args.map:
        folium_map(*args.map)
    else:
        plots(load_inventory())
        folium_map("train", "00000003")  # a well-behaved trip (821 points, route covers the trace)
        folium_map("train", "00000054")  # a problem trip (route covers only a small part of the GPS trace)
