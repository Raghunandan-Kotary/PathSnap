"""
Report maps (Deliverables section 6): for one student_test trip, the raw GPS
points, nearby road network, reference route, classifier-only route and
HMM/Viterbi route, as interactive Folium HTML plus PNG screenshots for the PDF.

    python maps/export_report_maps.py --proba-dir <harness run folder> [--trip 00000009]

Without --trip it picks the student_test trip where the HMM changed the most
observations. The zoomed view frames the 90-observation window with the most
observations the HMM corrected. Uses the decoder weights in hmm/config.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation.route_metrics import classifier_only_arcs, compute_route_metrics, load_trip_reference  # noqa: E402
from hmm.decoder import RoadIndex, ViterbiRouteDecoder  # noqa: E402
from maps.folium_generator import _build_map  # noqa: E402

WINDOW = 90


def screenshot(html_path: Path, png_path: Path) -> None:
    """PNG for the PDF report. Uses Playwright (Python package, or the Node one via `node`)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        import subprocess
        js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();"
              "const p=await b.newPage({viewport:{width:1100,height:760},deviceScaleFactor:2});"
              f"await p.goto('{html_path.resolve().as_uri()}',{{waitUntil:'networkidle'}});"
              f"await p.waitForTimeout(1500);await p.screenshot({{path:'{png_path.resolve()}'}});await b.close();}})();")
        subprocess.run(["node", "-e", js], check=True)
        return
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 760}, device_scale_factor=2)
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(png_path))
        browser.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=REPO_ROOT / "person2/experiments/team_table/candidates_features.parquet")
    ap.add_argument("--proba-dir", type=Path, required=True)
    ap.add_argument("--trip", default=None)
    ap.add_argument("--html-dir", type=Path, default=REPO_ROOT / "maps" / "html")
    ap.add_argument("--png-dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    a = ap.parse_args()

    df = pd.read_parquet(a.table, columns=["trip_id", "obs_idx", "arc_id", "gps_lon", "gps_lat", "gps_time", "partition"])
    df = df[df.partition == "student_test"]
    df["trip_id"] = df["trip_id"].astype(str).str.zfill(8)
    probs = pd.read_parquet(a.proba_dir / "proba_student_test.parquet")
    probs["trip_id"] = probs["trip_id"].astype(str).str.zfill(8)
    df = df.merge(probs, on=["trip_id", "obs_idx", "arc_id"], how="left", validate="one_to_one")
    decoder = ViterbiRouteDecoder.from_config(REPO_ROOT / "hmm" / "config.yaml")

    results = {}
    for tid, g in df.groupby("trip_id"):
        if a.trip and tid != a.trip:
            continue
        g = g.reset_index(drop=True)
        ref = load_trip_reference(tid)
        n = len(ref["track"])
        only = np.asarray(classifier_only_arcs(g, n))
        pred = decoder.decode(g, g["proba"].to_numpy(), RoadIndex.from_frames(ref["nodes"], ref["arcs"]))
        hmm = np.full(n, -1)
        hmm[: len(pred)] = pred
        gt = np.asarray(ref["gt"])
        results[tid] = dict(ref=ref, only=only, hmm=hmm, fixed=(only != gt) & (hmm == gt))

    tid = a.trip or max(results, key=lambda t: results[t]["fixed"].sum())
    r, ref = results[tid], results[tid]["ref"]
    a.html_dir.mkdir(parents=True, exist_ok=True)
    a.png_dir.mkdir(parents=True, exist_ok=True)

    m_only = compute_route_metrics(r["only"].tolist(), ref["gt"], ref["arcs"])
    m_hmm = compute_route_metrics(r["hmm"].tolist(), ref["gt"], ref["arcs"])
    print(f"trip {tid}: classifier-only acc {m_only['pointwise_accuracy']:.4f} -> HMM {m_hmm['pointwise_accuracy']:.4f}; "
          f"observations fixed by HMM: {int(r['fixed'].sum())}")

    # Full trip
    fmap = _build_map(ref["track"], ref["nodes"], ref["arcs"], r["hmm"].tolist(), ref["route"],
                      classifier_only_route=r["only"].tolist())
    full_html = a.html_dir / f"person3_trip_{tid}_full.html"
    fmap.save(str(full_html))

    # Zoom on the window where the HMM corrected the most points
    fixed = r["fixed"].astype(int)
    start = int(np.argmax(np.convolve(fixed, np.ones(WINDOW, dtype=int), mode="valid")))
    win = ref["track"].iloc[start:start + WINDOW]
    zmap = _build_map(win, ref["nodes"], ref["arcs"], r["hmm"][start:start + WINDOW].tolist(), ref["route"],
                      classifier_only_route=r["only"][start:start + WINDOW].tolist())
    pad = 0.0004
    zmap.fit_bounds([[win.lat.min() - pad, win.lon.min() - pad], [win.lat.max() + pad, win.lon.max() + pad]])
    zoom_html = a.html_dir / f"person3_trip_{tid}_zoom.html"
    zmap.save(str(zoom_html))

    for html in (full_html, zoom_html):
        png = a.png_dir / html.with_suffix(".png").name
        screenshot(html, png)
        print("wrote", html, "and", png)
    print(f"zoom window: observations {start}-{start + WINDOW - 1}, "
          f"{int(fixed[start:start + WINDOW].sum())} of them corrected by the HMM")


if __name__ == "__main__":
    main()
