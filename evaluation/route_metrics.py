"""
Shared route-level evaluation metrics for PathSnap (Person 3).

Used identically by Person 2's report tables and Person 4's API
(`api/pipeline.py`), per docs/INTERFACES.md Interface 4.1, so the numbers
the report quotes and the numbers the live demo shows always agree.
"""
from difflib import SequenceMatcher
from typing import List, Optional

import networkx as nx

NO_CANDIDATE_ARC = -1


def _arcs_connected(arc_u: int, arc_v: int, road_graph: nx.MultiDiGraph, arc_edge_lookup: dict) -> bool:
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
    road_graph: nx.MultiDiGraph,
) -> dict:
    """
    predicted_arcs / ground_truth_arcs: chronological, observation-aligned
    arc_id sequences (same length and index meaning -- element i of each is
    "the arc at the i-th relevant GPS observation"). Observations with no
    candidate are represented by NO_CANDIDATE_ARC (-1) and never count as a
    match.

    Returns:
      - "pointwise_accuracy": fraction of observations where predicted_arc
        == gt_arc.
      - "route_continuity": fraction of consecutive predicted arcs (u, v)
        that are topologically connected in road_graph (same arc, or u's
        end node equals v's start node).
      - "sequence_score": normalized sequence similarity (difflib ratio,
        in [0, 1]) between the predicted and ground-truth arc sequences --
        robust to small shifts/insertions that pointwise accuracy punishes
        harshly.
    """
    n = min(len(predicted_arcs), len(ground_truth_arcs))
    if n == 0:
        return {"pointwise_accuracy": 0.0, "route_continuity": 1.0, "sequence_score": 0.0}

    pred, gt = predicted_arcs[:n], ground_truth_arcs[:n]
    matches = sum(1 for p, g in zip(pred, gt) if p == g and p != NO_CANDIDATE_ARC)
    pointwise_accuracy = matches / n

    arc_edge_lookup = {data["arc_id"]: (u, v) for u, v, data in road_graph.edges(data=True)}
    valid_pred = [a for a in predicted_arcs if a != NO_CANDIDATE_ARC]
    if len(valid_pred) < 2:
        route_continuity = 1.0
    else:
        connected = sum(
            1 for a, b in zip(valid_pred[:-1], valid_pred[1:])
            if _arcs_connected(a, b, road_graph, arc_edge_lookup)
        )
        route_continuity = connected / (len(valid_pred) - 1)

    sequence_score = SequenceMatcher(None, pred, gt).ratio()

    return {
        "pointwise_accuracy": pointwise_accuracy,
        "route_continuity": route_continuity,
        "sequence_score": sequence_score,
    }


def candidate_level_ground_truth(candidates_df, obs_range: Optional[range] = None) -> List[int]:
    """
    Derives the observation-aligned ground-truth arc sequence from a
    *training-time* candidates_df that still has the `label` column (label
    == 1 marks the candidate matching the reference route at that step).
    Used to build `ground_truth_arcs` for compute_route_metrics during
    offline evaluation; at serving time, Person 4 instead compares directly
    against the raw `.route` file server-side.
    """
    if "label" not in candidates_df.columns:
        raise ValueError("candidates_df has no 'label' column -- pass the training-time candidate table")

    positives = candidates_df[candidates_df["label"] == 1]
    gt_by_obs = dict(zip(positives["obs_idx"], positives["arc_id"]))

    if obs_range is None:
        obs_range = range(int(candidates_df["obs_idx"].min()), int(candidates_df["obs_idx"].max()) + 1)

    return [gt_by_obs.get(obs_idx, NO_CANDIDATE_ARC) for obs_idx in obs_range]
