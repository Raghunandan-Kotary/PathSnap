"""Explain the API latency numbers: cold vs warm request, with and without the Folium map (report section 7.3).

    python person2/experiments/measure_api_latency.py --trip 00000015

Runs api/pipeline.py in one fresh process: request 1 (cold, with map), request 2 (warm, with map),
request 3 (warm, without map, which is how person2/experiments/evaluate_pipeline.py measures).
Writes person2/experiments/results/api_latency.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="00000015")
    a = ap.parse_args()
    t0 = time.perf_counter()
    from api.pipeline import pipeline_instance  # noqa: E402  (model and decoder load happen here)
    load_s = time.perf_counter() - t0
    rows = []
    for name, include_map in (("cold_with_map", True), ("warm_with_map", True), ("warm_without_map", False)):
        t = time.perf_counter()
        r = pipeline_instance.run_pipeline(a.trip, include_map=include_map)
        wall = (time.perf_counter() - t) * 1000
        tm = r.timing
        parts = tm.data_load_ms + tm.feature_extraction_ms + tm.model_inference_ms + tm.viterbi_decode_ms
        rows.append({"request": name, "wall_ms": wall, "reported_total_ms": tm.total_latency_ms, "data_load_ms": tm.data_load_ms,
                     "feature_extraction_ms": tm.feature_extraction_ms, "model_inference_ms": tm.model_inference_ms,
                     "viterbi_decode_ms": tm.viterbi_decode_ms, "sum_of_breakdown_ms": parts,
                     "not_in_breakdown_ms": tm.total_latency_ms - parts, "observations": r.num_observations})
        print(f"{name:18s} total {tm.total_latency_ms:8.1f} ms | breakdown sum {parts:7.1f} ms | rest (map + scoring) {tm.total_latency_ms - parts:7.1f} ms")
    out = {"trip": a.trip, "server_start_load_s": load_s, "requests": rows}
    (ROOT / "person2" / "experiments" / "results" / "api_latency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
