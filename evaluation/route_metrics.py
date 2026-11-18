"""
Shared route-level evaluation metrics for PathSnap (Person 3).

Used identically by the report tables (Person 2's harness, through
`route_metrics_for_predictions`), the HMM experiments (hmm/tune_decoder.py)
and the live API (api/pipeline.py), per docs/INTERFACES.md Interface 4.1, so
every route number in the report and the demo is computed one way.

Ground truth per observation is the reference-route arc assigned to that GPS
point by features.candidate_generator.match_gps_to_route_arcs -- the same
assignment that produced the training labels. (`.route` itself lists each
traversed arc once, so it cannot be compared position by position with one
prediction per GPS point.)
"""
from __future__ import annotations

import os
from difflib import SequenceMatcher
from itertools import groupby
from pathlib import Path
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

NO_CANDIDATE_ARC = -1
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", Path(__file__).resolve().parents[1] / "data" / "raw"))


def _arc_endpoints(road: Union[nx.MultiDiGraph, pd.DataFrame]) -> tuple:
    """(from_node, to_node) arrays indexed by arc_id."""
    if isinstance(road, pd.DataFrame):
        return road["from_node"].to_numpy(), road["to_node"].to_numpy()
    edges = [(d["arc_id"], u, v) for u, v, d in road.edges(data=True)]
    n = max(e[0] for e in edges) + 1
    frm, to = np.full(n, -1), np.full(n, -2)
    for arc_id, u, v in edges:
        frm[arc_id], to[arc_id] = u, v
    return frm, to


def _collapse(arcs) -> list:
    """Ordered list of distinct consecutive arcs, i.e. the route as a road sequence."""
    return [a for a, _ in groupby(arcs) if a != NO_CANDIDATE_ARC]


def compute_route_metrics(
    predicted_arcs: List[int],
    ground_truth_arcs: List[int],
    road_graph: Union[nx.MultiDiGraph, pd.DataFrame],
) -> dict:
    """
    predicted_arcs / ground_truth_arcs: one arc_id per GPS observation, in
    time order, same length (NO_CANDIDATE_ARC = -1 where there is none).
    road_graph: hmm.road_graph.build_road_graph() output, or the raw arcs_df
    (arc_id = row index), which is what api/pipeline.py has.

    Returns:
      pointwise_accuracy (also as pointwise_top1_accuracy, the API field):
          fraction of observations whose predicted arc is the ground-truth arc.
      route_continuity: fraction of consecutive predicted arcs (u, v),
          skipping observations without a prediction, where v can directly
          follow u: u == v (still on the same road) or end(u) == start(v).
      arc_change_continuity: the same, counted only over pairs with u != v.
          At 1 Hz most pairs are "same arc", which makes route_continuity
          high for any matcher; this isolates the moments the route moves to
          a new road, where an independent classifier tends to jump.
      sequence_score: difflib similarity ratio (0..1) between the predicted
          and ground-truth routes as ordered lists of distinct arcs.
      sequence_coverage / sequence_precision: share of ground-truth route
          arcs that were predicted / share of predicted arcs on the route.
      plus the raw counts behind them, so results can be pooled over trips.
    """
    n = min(len(predicted_arcs), len(ground_truth_arcs))
    pred = np.asarray(predicted_arcs[:n], dtype=np.int64)
    gt = np.asarray(ground_truth_arcs[:n], dtype=np.int64)
    matches = int(((pred == gt) & (pred != NO_CANDIDATE_ARC)).sum())

    frm, to = _arc_endpoints(road_graph)
    valid = np.asarray([a for a in predicted_arcs if 0 <= a < len(frm)], dtype=np.int64)
    a, b = valid[:-1], valid[1:]
    connected = (a == b) | (to[a] == frm[b])
    changes = a != b
    pairs, n_connected = int(len(a)), int(connected.sum())
    n_changes, n_changes_connected = int(changes.sum()), int((connected & changes).sum())

    pred_route, gt_route = _collapse(pred.tolist()), _collapse(gt.tolist())
    sequence_score = SequenceMatcher(None, pred_route, gt_route, autojunk=False).ratio() if (pred_route or gt_route) else 0.0
    gt_set, pred_set = set(gt_route), set(pred_route)
    overlap = len(gt_set & pred_set)

    pointwise = matches / n if n else 0.0
    return {
        "pointwise_accuracy": round(pointwise, 4),
        "pointwise_top1_accuracy": round(pointwise, 4),
        "route_continuity": round(n_connected / pairs, 4) if pairs else 1.0,
        "arc_change_continuity": round(n_changes_connected / n_changes, 4) if n_changes else 1.0,
        "sequence_score": round(sequence_score, 4),
        "sequence_coverage": round(overlap / len(gt_set), 4) if gt_set else 0.0,
        "sequence_precision": round(overlap / len(pred_set), 4) if pred_set else 0.0,
        "total_observations": n,
        "matched_observations": matches,
        "continuity_pairs": pairs,
        "continuity_connected": n_connected,
        "arc_changes": n_changes,
        "arc_changes_connected": n_changes_connected,
    }


def pool_route_metrics(per_trip: pd.DataFrame) -> dict:
    """Pools per-trip compute_route_metrics rows: counts are summed, sequence scores averaged over trips."""
    s = per_trip[["total_observations", "matched_observations", "continuity_pairs", "continuity_connected",
                  "arc_changes", "arc_changes_connected"]].sum()
    return {
        "pointwise_accuracy": float(s.matched_observations / max(s.total_observations, 1)),
        "route_continuity": float(s.continuity_connected / max(s.continuity_pairs, 1)),
        "arc_change_continuity": float(s.arc_changes_connected / max(s.arc_changes, 1)),
        "sequence_score": float(per_trip["sequence_score"].mean()),
        "sequence_coverage": float(per_trip["sequence_coverage"].mean()),
        "sequence_precision": float(per_trip["sequence_precision"].mean()),
        "n_trips": int(len(per_trip)),
        "n_observations": int(s.total_observations),
    }


def align_route_to_observations(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route: List[int],
) -> List[int]:
    """One ground-truth arc per GPS observation, assigned exactly as the training labels were."""
    if not route:
        return [NO_CANDIDATE_ARC] * len(track_df)
    from features.candidate_generator import match_gps_to_route_arcs
    matched, _ = match_gps_to_route_arcs(track_df, nodes_df, arcs_df, route)
    return [int(a) for a in matched]


def trip_dir(trip_id: str, data_root: Path = DATA_ROOT) -> Path:
    for part in ("student_test", "train"):
        d = Path(data_root) / part / trip_id
        if d.is_dir():
            return d
    raise FileNotFoundError(f"trip {trip_id} not found under {data_root}/train or {data_root}/student_test")


def load_trip_reference(trip_id: str, data_root: Path = DATA_ROOT) -> dict:
    """Raw files plus the per-observation ground truth for one train/student_test trip."""
    from data.loader import load_arcs, load_nodes, load_route, load_track
    d = trip_dir(trip_id, data_root)
    track, nodes, arcs = load_track(d / f"{trip_id}.track"), load_nodes(d / f"{trip_id}.nodes"), load_arcs(d / f"{trip_id}.arcs")
    route = load_route(d / f"{trip_id}.route")
    return {"track": track, "nodes": nodes, "arcs": arcs, "route": route,
            "gt": align_route_to_observations(track, nodes, arcs, route)}


def classifier_only_arcs(trip_pred: pd.DataFrame, n_obs: int, score_col: str = "proba") -> List[int]:
    """Highest-scoring candidate per observation, each chosen independently (no HMM)."""
    best = trip_pred.loc[trip_pred.groupby("obs_idx")[score_col].idxmax(), ["obs_idx", "arc_id"]]
    out = np.full(n_obs, NO_CANDIDATE_ARC, dtype=np.int64)
    out[best["obs_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return out.tolist()


def route_metrics_for_predictions(pred_df: pd.DataFrame, data_root: Path = DATA_ROOT,
                                  per_trip_out: Optional[Path] = None) -> dict:
    """
    Route metrics of classifier-only predictions, pooled over the trips in
    pred_df (columns trip_id, obs_idx, arc_id, proba). Plugs into Person 2's
    harness as run_experiment(route_metrics_fn=...), or can be applied to an
    existing run's proba parquet with evaluation/route_metrics_cli.py.
    """
    rows = []
    for tid, g in pred_df.groupby("trip_id"):
        ref = load_trip_reference(str(tid), data_root)
        arcs = classifier_only_arcs(g, len(ref["track"]))
        rows.append({"trip_id": tid, **compute_route_metrics(arcs, ref["gt"], ref["arcs"])})
    per_trip = pd.DataFrame(rows)
    if per_trip_out is not None:
        per_trip.to_csv(per_trip_out, index=False)
    return pool_route_metrics(per_trip)
