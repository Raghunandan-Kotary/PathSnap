"""EDA figures for report section 2, each labelled with the partition it uses, plus the numbers quoted in the text.

    python person2/eda/make_report_eda_figures.py

Writes docs/figures/report_eda_*.png and person2/eda/outputs/report_eda_numbers.json:
  * sampling intervals (share of 1 s steps) per partition,
  * distance from every GPS point to its trip's reference route per partition (all points),
  * one training trip drawn with its GPS points, nearby road network and reference route,
  * a screenshot of the interactive Folium map person2/eda/outputs/folium_train_00000003.html.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from person2 import config  # noqa: E402
from person2.geo import LocalProjector  # noqa: E402

FIG = config.REPO_ROOT.parent / "docs" / "figures"
LABEL = {"train": "train (80 trips)", "student_test": "student_test (10 trips)"}


def trip_arrays(part: str, trip: str):
    d = config.PARTITION_DIRS[part] / trip
    trk = pd.read_csv(d / f"{trip}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(d / f"{trip}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(d / f"{trip}.arcs", sep=r"\s+", header=None, names=["u", "v"])
    route = pd.read_csv(d / f"{trip}.route", header=None)[0].to_numpy()
    return trk, nodes, arcs, route


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    dts, dists = {}, {}
    for part, pdir in config.PARTITION_DIRS.items():
        dts[part], dists[part] = [], []
        for tdir in sorted(p for p in pdir.iterdir() if p.is_dir()):
            trk, nodes, arcs, route = trip_arrays(part, tdir.name)
            dts[part].append(np.diff(trk.t.to_numpy()))
            pr = LocalProjector.from_lonlat(nodes.lon.to_numpy(), nodes.lat.to_numpy())
            nx_, ny_ = pr.to_xy(nodes.lon.to_numpy(), nodes.lat.to_numpy())
            gx, gy = pr.to_xy(trk.lon.to_numpy(), trk.lat.to_numpy())
            u, v = arcs.u.to_numpy()[route], arcs.v.to_numpy()[route]
            mls = shapely.multilinestrings(shapely.linestrings(np.stack([np.stack([nx_[u], ny_[u]], 1), np.stack([nx_[v], ny_[v]], 1)], 1)))
            dists[part].append(shapely.distance(shapely.points(gx, gy), mls))
        dts[part], dists[part] = np.concatenate(dts[part]), np.concatenate(dists[part])

    nums = {}
    for part in dts:
        nums[part] = {"steps": int(len(dts[part])), "share_1s": float(np.isclose(dts[part], 1.0).mean()),
                      "gps_points": int(len(dists[part])), "median_gps_to_route_m": float(np.median(dists[part])),
                      "p95_gps_to_route_m": float(np.percentile(dists[part], 95)),
                      "share_within_30m": float((dists[part] <= 30).mean())}
    alld = np.concatenate(list(dts.values()))
    nums["all_known"] = {"share_1s": float(np.isclose(alld, 1.0).mean()),
                         "median_gps_to_route_m": float(np.median(np.concatenate(list(dists.values()))))}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for part, c in (("train", "C0"), ("student_test", "C1")):
        x = dts[part][dts[part] <= 10]
        ax[0].hist(x, bins=np.arange(0.5, 10.6, 1), alpha=0.6, color=c, density=True,
                   label=f"{LABEL[part]}: {100 * nums[part]['share_1s']:.1f}% of steps are 1 s")
        d = dists[part][dists[part] <= 50]
        ax[1].hist(d, bins=50, alpha=0.6, color=c, density=True,
                   label=f"{LABEL[part]}: median {nums[part]['median_gps_to_route_m']:.1f} m (all points)")
    ax[0].set(yscale="log", xlabel="time between consecutive GPS points (s), shown up to 10 s", ylabel="share of steps (log)", title="Sampling intervals")
    ax[1].set(xlabel="distance from GPS point to the reference route (m), shown up to 50 m", ylabel="density", title="GPS noise relative to the reference route")
    for a_ in ax:
        a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "report_eda_sampling_and_noise.png", dpi=150)
    plt.close(fig)

    trk, nodes, arcs, route = trip_arrays("train", "00000003")
    pr = LocalProjector.from_lonlat(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    nx_, ny_ = pr.to_xy(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    gx, gy = pr.to_xy(trk.lon.to_numpy(), trk.lat.to_numpy())
    pad = 300
    inside = (nx_ > gx.min() - pad) & (nx_ < gx.max() + pad) & (ny_ > gy.min() - pad) & (ny_ < gy.max() + pad)
    keep = inside[arcs.u.to_numpy()] & inside[arcs.v.to_numpy()]
    fig, ax = plt.subplots(figsize=(8, 8))
    for u, v in zip(arcs.u[keep], arcs.v[keep]):
        ax.plot([nx_[u], nx_[v]], [ny_[u], ny_[v]], color="#bbbbbb", lw=0.5)
    for u, v in zip(arcs.u.to_numpy()[route], arcs.v.to_numpy()[route]):
        ax.plot([nx_[u], nx_[v]], [ny_[u], ny_[v]], color="#1a9850", lw=2.5)
    ax.scatter(gx, gy, s=3, color="#2166ac", zorder=3)
    ax.plot([], [], color="#bbbbbb", label=f"road network ({int(keep.sum()):,} arcs within {pad} m)")
    ax.plot([], [], color="#1a9850", lw=2.5, label=f"reference route ({len(route)} arcs)")
    ax.scatter([], [], s=10, color="#2166ac", label=f"GPS points ({len(trk):,})")
    ax.set(aspect="equal", xlabel="x (m, local projection)", ylabel="y (m, local projection)",
           title="Training trip 00000003 (train partition): GPS trajectory, road network and reference route")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "report_eda_trajectory_train_00000003.png", dpi=150)
    plt.close(fig)

    html = config.EDA_OUT / "folium_train_00000003.html"
    if html.exists():
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except Exception:
                b = p.chromium.launch(channel="msedge")
            page = b.new_page(viewport={"width": 1100, "height": 760}, device_scale_factor=2)
            page.goto(html.resolve().as_uri(), wait_until="networkidle")
            page.wait_for_timeout(2000)
            page.screenshot(path=str(FIG / "report_eda_folium_train_00000003.png"))
            b.close()

    (config.EDA_OUT / "report_eda_numbers.json").write_text(json.dumps(nums, indent=2), encoding="utf-8")
    print(json.dumps(nums, indent=2))


if __name__ == "__main__":
    main()
