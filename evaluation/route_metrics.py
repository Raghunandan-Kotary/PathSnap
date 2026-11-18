"""
Shared route-level evaluation metrics for PathSnap (Person 3).

Used identically by Person 2's report tables and Person 4's API
(`api/pipeline.py`), per docs/INTERFACES.md Interface 4.1, so the numbers
the report quotes and the numbers the live demo shows always agree.
"""
import itertools
from difflib import SequenceMatcher
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from data.preprocessor import R_EARTH

NO_CANDIDATE_ARC = -1


def _haversine_distance_vec(lon1, lat1, lon2_arr: np.ndarray, lat2_arr: np.ndarray) -> np.ndarray:
    """Vectorized haversine from one point to an array of points, in meters."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * R_EARTH * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _arc_edge_lookup_from_arcs_df(arcs_df: pd.DataFrame) -> dict:
    """arc_id (row index, per data/loader.load_arcs) -> (from_node, to_node)."""
    return {int(i): (int(row["from_node"]), int(row["to_node"])) for i, row in arcs_df.iterrows()}


def _arc_edge_lookup(road_graph_or_arcs_df: Union[nx.MultiDiGraph, pd.DataFrame]) -> dict:
    if isinstance(road_graph_or_arcs_df, pd.DataFrame):
        return _arc_edge_lookup_from_arcs_df(road_graph_or_arcs_df)
    return {data["arc_id"]: (u, v) for u, v, data in road_graph_or_arcs_df.edges(data=True)}


def _arcs_connected(arc_u: int, arc_v: int, arc_edge_lookup: dict) -> bool:
    """True if arc_v can immediately follow arc_u: same arc, or u's end node is v's start node."""
    if arc_u == arc_v:
        return True
    if arc_u not in arc_edge_lookup or arc_v not in arc_edge_lookup:
        return False
    _, u_end = arc_edge_lookup[arc_u]
    v_start, _ = arc_edge_lookup[arc_v]
    return u_end == v_start


def compute_route_metrics(
    predicted_arcs: List[int],
    ground_truth_arcs: List[int],
    road_graph: Union[nx.MultiDiGraph, pd.DataFrame],
) -> dict:
    """
    predicted_arcs / ground_truth_arcs: chronological, observation-aligned
    arc_id sequences (same length and index meaning -- element i of each is
    "the arc at the i-th relevant GPS observation"). Observations with no
    candidate are represented by NO_CANDIDATE_ARC (-1) and never count as a
    match.

    road_graph: a prebuilt hmm.road_graph.build_road_graph() MultiDiGraph,
    OR the raw `arcs_df` DataFrame (arc_id = row index, columns from_node/
    to_node) -- the latter is accepted directly for compatibility with
    api/pipeline.py, which only has arcs_df on hand, not a prebuilt graph.

    Returns a superset of keys covering both the documented interface and
    api/pipeline.py's RouteMetricsResponse fields:
      - "pointwise_accuracy" / "pointwise_top1_accuracy": fraction of
        observations where predicted_arc == gt_arc (same value, two names
        -- the schema in api/schemas.py uses the latter).
      - "route_continuity": fraction of consecutive predicted arcs (u, v)
        that are topologically connected (same arc, or u's end node ==
        v's start node).
      - "sequence_score": normalized sequence similarity (difflib ratio,
        in [0, 1]) between the full predicted and ground-truth arc
        sequences -- robust to small shifts/insertions that pointwise
        accuracy punishes harshly.
      - "sequence_coverage" / "sequence_precision": recall/precision of
        the *set* of distinct arcs predicted (consecutive repeats
        collapsed) against the set of distinct ground-truth route arcs.
      - "total_observations", "matched_observations": raw counts behind
        pointwise_accuracy.
    """
    n = min(len(predicted_arcs), len(ground_truth_arcs))
    if n == 0:
        return {
            "pointwise_accuracy": 0.0,
            "pointwise_top1_accuracy": 0.0,
            "route_continuity": 1.0,
            "sequence_score": 0.0,
            "sequence_coverage": 0.0,
            "sequence_precision": 0.0,
            "total_observations": 0,
            "matched_observations": 0,
        }

    pred, gt = predicted_arcs[:n], ground_truth_arcs[:n]
    matches = sum(1 for p, g in zip(pred, gt) if p == g and p != NO_CANDIDATE_ARC)
    pointwise_accuracy = matches / n

    arc_edge_lookup = _arc_edge_lookup(road_graph)
    valid_pred = [a for a in predicted_arcs if a != NO_CANDIDATE_ARC]
    if len(valid_pred) < 2:
        route_continuity = 1.0
    else:
        connected = sum(
            1 for a, b in zip(valid_pred[:-1], valid_pred[1:])
            if _arcs_connected(a, b, arc_edge_lookup)
        )
        route_continuity = connected / (len(valid_pred) - 1)

    sequence_score = SequenceMatcher(None, pred, gt).ratio()

    gt_set = set(ground_truth_arcs) if ground_truth_arcs else set()
    collapsed_pred_set = {k for k, _ in itertools.groupby(predicted_arcs) if k != NO_CANDIDATE_ARC}
    if gt_set:
        overlap = len(collapsed_pred_set & gt_set)
        sequence_coverage = overlap / len(gt_set)
        sequence_precision = overlap / max(len(collapsed_pred_set), 1)
    else:
        sequence_coverage = 0.0
        sequence_precision = 0.0

    return {
        "pointwise_accuracy": round(pointwise_accuracy, 4),
        "pointwise_top1_accuracy": round(pointwise_accuracy, 4),
        "route_continuity": round(route_continuity, 4),
        "sequence_score": round(sequence_score, 4),
        "sequence_coverage": round(sequence_coverage, 4),
        "sequence_precision": round(sequence_precision, 4),
        "total_observations": n,
        "matched_observations": matches,
    }


def align_route_to_observations(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route: List[int],
    radius_m: float = 60.0,
) -> List[int]:
    """
    Converts the raw `.route` file (the ordered list of DISTINCT arcs
    traversed -- routinely 2.5x-28x shorter than `.track`, since one arc
    spans many consecutive GPS fixes) into an observation-aligned sequence
    the same length as predicted_arcs, for a fair pointwise comparison in
    compute_route_metrics.

    For each GPS observation, picks whichever route arc is geometrically
    nearest (by midpoint distance), among those within radius_m;
    NO_CANDIDATE_ARC (-1) if none are that close (GPS noise/dropout). Same
    nearest-route-member heuristic as features/candidate_generator.py uses
    to build training labels -- see its docstring for why positional
    indexing into `.route` is wrong.
    """
    if not route:
        return [NO_CANDIDATE_ARC] * len(track_df)

    route_arcs = arcs_df.iloc[route]
    mid_lon = ((nodes_df["lon"].values[route_arcs["from_node"].values]
                + nodes_df["lon"].values[route_arcs["to_node"].values]) / 2.0)
    mid_lat = ((nodes_df["lat"].values[route_arcs["from_node"].values]
                + nodes_df["lat"].values[route_arcs["to_node"].values]) / 2.0)

    aligned = []
    for _, obs in track_df.iterrows():
        dists = _haversine_distance_vec(obs["lon"], obs["lat"], mid_lon, mid_lat)
        best_i = int(np.argmin(dists))
        aligned.append(route[best_i] if dists[best_i] <= radius_m else NO_CANDIDATE_ARC)
    return aligned


def candidate_level_ground_truth(candidates_df, obs_range: Optional[range] = None) -> List[int]:
    """
    Derives the observation-aligned ground-truth arc sequence from a
    *training-time* candidates_df that still has the `label` column (label
    == 1 marks the candidate matching the reference route at that step).
    Used to build `ground_truth_arcs` for compute_route_metrics during
    offline evaluation; at serving time, Person 4 uses
    `align_route_to_observations` instead, since only the raw `.route`
    file (not a labeled candidate table) is available there.
    """
    if "label" not in candidates_df.columns:
        raise ValueError("candidates_df has no 'label' column -- pass the training-time candidate table")

    positives = candidates_df[candidates_df["label"] == 1]
    gt_by_obs = dict(zip(positives["obs_idx"], positives["arc_id"]))

    if obs_range is None:
        obs_range = range(int(candidates_df["obs_idx"].min()), int(candidates_df["obs_idx"].max()) + 1)

    return [gt_by_obs.get(obs_idx, NO_CANDIDATE_ARC) for obs_idx in obs_range]
