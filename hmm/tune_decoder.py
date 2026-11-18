"""
Tunes the HMM/Viterbi decoder and compares it with classifier-only predictions
(report section 5; Models and Training Guidelines section 9).

Tuning uses the 80 TRAINING trips with OUT-OF-FOLD classifier probabilities
(each trip scored by a model that never saw it), so the decoder weights are not
fitted to in-sample scores. The 10 student_test trips are only scored, never
used to choose a parameter; the evaluator partition is never touched.

    python hmm/tune_decoder.py \
        --table person2/experiments/team_table/candidates_features.parquet \
        --proba-dir person2/experiments/logs/<xgboost run> --write-config

Outputs (all small, committed):
    hmm/tuning/grid_search.csv           every weight combination, train-trip score
    hmm/tuning/best_params.json          chosen weights + how they were chosen
    experiments/results/person3_hmm_comparison_per_trip.csv
    experiments/results/person3_hmm_comparison_summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from evaluation.route_metrics import (NO_CANDIDATE_ARC, classifier_only_arcs, compute_route_metrics,  # noqa: E402
                                      load_trip_reference, pool_route_metrics)
from hmm.decoder import RoadIndex, ViterbiRouteDecoder  # noqa: E402

# emission_weight > 0: the route reconstruction must use the selected classifier's outputs
# (Models and Training Guidelines, section 8). The distance-only decoder (w = 0) is still
# evaluated alongside as a reference in the comparison.
GRID = {
    "emission_weight": [0.25, 0.5, 0.75, 1.0],
    "distance_scale_m": [7.0, 10.0, 15.0],
    "transition_scale_m": [10.0, 20.0, 50.0],
}
PENALTIES = [10.0, 25.0, 50.0]
# These two change the prepared transition costs / segmentation, so each value is re-prepared.
GAP_THRESHOLDS = [10.0, 30.0]
BACKTRACK_TOLERANCES = [10.0, 25.0, 50.0]
KEY = ["trip_id", "obs_idx", "arc_id"]

_TRIPS: dict = {}  # filled before forking so grid workers share it copy-on-write


def load_inputs(table_path: Path, proba_dir: Path) -> pd.DataFrame:
    cols = KEY + ["gps_lon", "gps_lat", "gps_time", "partition"]
    df = pd.read_parquet(table_path, columns=cols)
    df["trip_id"] = df["trip_id"].astype(str).str.zfill(8)
    probs = pd.concat([pd.read_parquet(proba_dir / "oof_proba_train.parquet"),
                       pd.read_parquet(proba_dir / "proba_student_test.parquet")], ignore_index=True)
    probs["trip_id"] = probs["trip_id"].astype(str).str.zfill(8)
    df = df.merge(probs, on=KEY, how="left", validate="one_to_one")
    if df["proba"].isna().any():
        raise ValueError("some candidates have no probability: the proba files do not match this table")
    return df


def prepare_trips(df: pd.DataFrame, decoder: ViterbiRouteDecoder) -> dict:
    trips = {}
    for tid, g in df.groupby("trip_id", sort=True):
        g = g.reset_index(drop=True)
        ref = load_trip_reference(tid)
        t0 = time.perf_counter()
        road = RoadIndex.from_frames(ref["nodes"], ref["arcs"])
        prepared = decoder.prepare(g, road)
        trips[tid] = {
            "partition": g["partition"].iat[0], "candidates": g, "proba": g["proba"].to_numpy(),
            "prepared": prepared, "prepare_s": time.perf_counter() - t0,
            "gt": np.asarray(ref["gt"], dtype=np.int64), "arcs_df": ref["arcs"], "n_obs": len(ref["track"]),
        }
    return trips


def _full_length(pred: list, prepared, n_obs: int) -> np.ndarray:
    out = np.full(n_obs, NO_CANDIDATE_ARC, dtype=np.int64)
    out[prepared.obs_min:prepared.obs_max + 1] = pred
    return out


def hmm_arcs(trip: dict, decoder: ViterbiRouteDecoder, use_proba: bool = True) -> np.ndarray:
    pred = decoder.decode_prepared(trip["prepared"], trip["proba"] if use_proba else None)
    return _full_length(pred, trip["prepared"], trip["n_obs"])


def _score_combo(params: dict) -> dict:
    """Pooled pointwise accuracy on the training trips for one weight combination (grid worker)."""
    dec = ViterbiRouteDecoder(**params)
    hit = total = 0
    for trip in _TRIPS.values():
        if trip["partition"] != "train":
            continue
        pred = hmm_arcs(trip, dec)
        hit += int(((pred == trip["gt"]) & (pred != NO_CANDIDATE_ARC)).sum())
        total += trip["n_obs"]
    return {**params, "train_pointwise_accuracy": hit / total}


def grid_search(base: dict, combos: list[dict], workers: int) -> pd.DataFrame:
    jobs = [{**base, **c} for c in combos]
    if "fork" in mp.get_all_start_methods() and workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            rows = pool.map(_score_combo, jobs)
    else:  # Windows has no fork: score the same combinations serially (identical results, slower)
        rows = []
        for i, j in enumerate(jobs, 1):
            rows.append(_score_combo(j))
            print(f"  combo {i}/{len(jobs)}: train acc {rows[-1]['train_pointwise_accuracy']:.4f}", flush=True)
    return pd.DataFrame(rows).sort_values("train_pointwise_accuracy", ascending=False)


def evaluate(trips: dict, partition: str, decoder: ViterbiRouteDecoder) -> tuple[pd.DataFrame, dict]:
    """Classifier-only vs HMM (distance only) vs HMM (classifier emissions), per trip and pooled."""
    rows, outcome = [], {"fixed_by_hmm": 0, "broken_by_hmm": 0, "both_right": 0, "both_wrong": 0,
                         "both_wrong_true_arc_not_a_candidate": 0, "broken_within_3_obs_of_segment_start": 0}
    timing = {"prepare_s": [], "viterbi_s": [], "n_obs": []}
    for tid, trip in trips.items():
        if trip["partition"] != partition:
            continue
        only = np.asarray(classifier_only_arcs(trip["candidates"], trip["n_obs"]))
        t0 = time.perf_counter()
        hmm = hmm_arcs(trip, decoder)
        timing["viterbi_s"].append(time.perf_counter() - t0)
        timing["prepare_s"].append(trip["prepare_s"])
        timing["n_obs"].append(trip["n_obs"])
        dist_only = hmm_arcs(trip, decoder, use_proba=False)
        for method, arcs in (("classifier_only", only), ("hmm_distance_only", dist_only), ("hmm_classifier", hmm)):
            rows.append({"trip_id": tid, "method": method, "n_obs": trip["n_obs"],
                         **compute_route_metrics(arcs.tolist(), trip["gt"].tolist(), trip["arcs_df"])})

        gt = trip["gt"]
        only_ok, hmm_ok = only == gt, hmm == gt
        cand = trip["candidates"]
        in_cands = set(zip(cand["obs_idx"], cand["arc_id"]))
        gt_missing = np.array([(i, g) not in in_cands for i, g in enumerate(gt)])
        seg_start = np.zeros(trip["n_obs"], dtype=bool)
        for seg in trip["prepared"].segments:
            for step in seg[:3]:
                seg_start[step.obs_idx] = True
        broken = only_ok & ~hmm_ok
        outcome["fixed_by_hmm"] += int((~only_ok & hmm_ok).sum())
        outcome["broken_by_hmm"] += int(broken.sum())
        outcome["both_right"] += int((only_ok & hmm_ok).sum())
        outcome["both_wrong"] += int((~only_ok & ~hmm_ok).sum())
        outcome["both_wrong_true_arc_not_a_candidate"] += int((~only_ok & ~hmm_ok & gt_missing).sum())
        outcome["broken_within_3_obs_of_segment_start"] += int((broken & seg_start).sum())

    per_trip = pd.DataFrame(rows)
    pooled = {m: pool_route_metrics(g) for m, g in per_trip.groupby("method")}
    n = np.asarray(timing["n_obs"], dtype=float)
    pooled["outcomes_vs_classifier_only"] = outcome
    pooled["decode_timing"] = {
        "prepare_s_mean_per_trip": float(np.mean(timing["prepare_s"])),
        "viterbi_s_mean_per_trip": float(np.mean(timing["viterbi_s"])),
        "ms_per_1k_obs_total": float(1000 * (np.sum(timing["prepare_s"]) + np.sum(timing["viterbi_s"])) / n.sum() * 1000),
    }
    return per_trip, pooled


def _reproduction_check(proba_dir: Path, reference_dir: Path | None) -> dict | None:
    """The probability files are large and kept outside git; record which committed run they reproduce."""
    if reference_dir is None:
        return None
    mine, ref = (json.loads((d / "metrics.json").read_text()) for d in (proba_dir, reference_dir))
    keys = ("top1_acc", "macro_f1", "log_loss", "ece")
    return {"reference_run_id": reference_dir.name,
            "cv_macro_f1_mean": [mine["cv_macro_f1_mean"], ref["cv_macro_f1_mean"]],
            **{f"student_test_{k}": [mine["student_test"][k], ref["student_test"][k]] for k in keys},
            "note": "[reproduced, logged]; same table, params and seed; differences come from multi-threaded XGBoost"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=REPO_ROOT / "person2/experiments/team_table/candidates_features.parquet")
    ap.add_argument("--proba-dir", type=Path, required=True, help="harness run folder with oof_proba_train / proba_student_test parquet")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "hmm" / "tuning")
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "experiments" / "results")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--grid-json", type=str, default=None,
                    help="override GRID with a JSON dict of lists, e.g. emission_weight, distance_scale_m, transition_scale_m")
    ap.add_argument("--write-config", action="store_true", help="write the chosen weights to hmm/config.yaml")
    ap.add_argument("--reference-run-dir", type=Path, default=None,
                    help="committed harness run that --proba-dir reproduces (its metrics are compared and recorded)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(a.table, a.proba_dir)
    run_cfg = json.loads((a.proba_dir / "config.json").read_text())
    print(f"{df.trip_id.nunique()} trips, {len(df):,} candidates; probabilities from {run_cfg['run_id']}", flush=True)

    base = dict(disconnect_penalty=25.0, gap_threshold_s=10.0, dijkstra_cutoff_m=1000.0,
                backtrack_tolerance_m=25.0, emission_blend="log_linear")
    global _TRIPS
    t0 = time.perf_counter()
    _TRIPS = prepare_trips(df, ViterbiRouteDecoder(**base))
    print(f"prepared in {time.perf_counter() - t0:.1f}s", flush=True)

    # 1) emission weight x GPS-noise scale x transition scale
    grid_def = json.loads(a.grid_json) if a.grid_json else GRID
    combos = [dict(zip(grid_def, v)) for v in itertools.product(*grid_def.values())]
    grid = grid_search(base, combos, a.workers)
    best = grid.iloc[0][list(grid_def)].to_dict()
    print("best emission/transition weights:", best, f"train acc {grid.iloc[0].train_pointwise_accuracy:.4f}", flush=True)

    # 2) unreachable-transition penalty, at those weights
    pen = grid_search({**base, **best}, [{"disconnect_penalty": p} for p in PENALTIES], a.workers)
    best["disconnect_penalty"] = float(pen.iloc[0].disconnect_penalty)
    grid = pd.concat([grid, pen], ignore_index=True)

    # 3) gap threshold and backtrack tolerance change the prepared trips, so re-prepare for each pair
    prep_rows = []
    for gap, back in itertools.product(GAP_THRESHOLDS, BACKTRACK_TOLERANCES):
        params = {**base, **best, "gap_threshold_s": gap, "backtrack_tolerance_m": back}
        _TRIPS = prepare_trips(df, ViterbiRouteDecoder(**params))
        prep_rows.append(_score_combo(params))
    preps = pd.DataFrame(prep_rows).sort_values("train_pointwise_accuracy", ascending=False)
    best["gap_threshold_s"] = float(preps.iloc[0].gap_threshold_s)
    best["backtrack_tolerance_m"] = float(preps.iloc[0].backtrack_tolerance_m)
    grid = pd.concat([grid, preps], ignore_index=True)
    grid.to_csv(a.out_dir / "grid_search.csv", index=False)

    final = {**base, **{k: float(v) for k, v in best.items()}}
    decoder = ViterbiRouteDecoder(**final)
    _TRIPS = prepare_trips(df, decoder)

    per_trip_tr, pooled_tr = evaluate(_TRIPS, "train", decoder)
    per_trip_te, pooled_te = evaluate(_TRIPS, "student_test", decoder)
    per_trip = pd.concat([per_trip_tr.assign(partition="train"), per_trip_te.assign(partition="student_test")])
    a.results_dir.mkdir(parents=True, exist_ok=True)
    per_trip.to_csv(a.results_dir / "person3_hmm_comparison_per_trip.csv", index=False)
    summary = {
        "decoder_params": final,
        "probabilities": {"run_id": run_cfg["run_id"], "model": run_cfg["model"], "params": run_cfg["params"],
                          "calibration": run_cfg["calibration"], "train": "out-of-fold (trip-grouped 5-fold CV)",
                          "student_test": "final model refit on the 80 training trips"},
        "tuning": {"objective": "pooled pointwise accuracy on the 80 training trips (out-of-fold probabilities)",
                   "combinations_tried": int(len(grid))},
        "reproduces": _reproduction_check(a.proba_dir, a.reference_run_dir),
        "train_out_of_fold": pooled_tr,
        "student_test": pooled_te,
    }
    (a.results_dir / "person3_hmm_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    (a.out_dir / "best_params.json").write_text(json.dumps({**summary["tuning"], "params": final,
                                                           "probabilities_run_id": run_cfg["run_id"]}, indent=2))
    if a.write_config:
        cfg_path = REPO_ROOT / "hmm" / "config.yaml"
        header = ("# HMM/Viterbi decoder weights (Person 3), chosen by hmm/tune_decoder.py on the 80 training\n"
                  "# trips with out-of-fold probabilities. See hmm/tuning/best_params.json and grid_search.csv.\n")
        cfg_path.write_text(header + yaml.safe_dump(final, sort_keys=False))

    for part, pooled in (("train (OOF)", pooled_tr), ("student_test", pooled_te)):
        print(f"\n== {part}")
        for m in ("classifier_only", "hmm_distance_only", "hmm_classifier"):
            p = pooled[m]
            print(f"  {m:18s} acc {p['pointwise_accuracy']:.4f}  continuity {p['route_continuity']:.4f}  "
                  f"arc-change continuity {p['arc_change_continuity']:.4f}  sequence {p['sequence_score']:.4f}")
        print("  outcomes:", pooled["outcomes_vs_classifier_only"])
        print("  timing:", pooled["decode_timing"])


if __name__ == "__main__":
    main()
