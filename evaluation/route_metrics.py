"""
Route-level and observation-level evaluation metrics for PathSnap.
Shared across offline model experimentation and online API serving.
"""
from typing import List, Dict, Set
import itertools
import pandas as pd

def compute_route_metrics(
    predicted_arcs: List[int],
    ground_truth_route: List[int],
    arcs_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Computes rigorous evaluation metrics:
    1. observation_route_match: Fraction of predicted observations on ground-truth route arcs
    2. sequence_coverage (Recall): Fraction of ground-truth route arcs recovered in prediction
    3. sequence_precision: Fraction of unique predicted arcs that exist on ground-truth route
    4. route_continuity: Fraction of consecutive arc transitions that maintain topological connectivity
    """
    total_obs = len(predicted_arcs)
    gt_set = set(ground_truth_route) if ground_truth_route else set()
    
    # 1. Observation-level match against ground-truth route
    if gt_set and total_obs > 0:
        obs_matches = sum(1 for a in predicted_arcs if a in gt_set)
        pointwise_top1 = obs_matches / total_obs
    else:
        pointwise_top1 = 0.0

    # Deduplicated consecutive predicted arcs
    collapsed_pred = [k for k, _ in itertools.groupby(predicted_arcs)]
    pred_set = set(collapsed_pred)

    # 2. Sequence Coverage (Recall)
    if gt_set:
        coverage = len(pred_set.intersection(gt_set)) / len(gt_set)
        precision = len(pred_set.intersection(gt_set)) / max(len(pred_set), 1)
    else:
        coverage = 0.0
        precision = 0.0

    # 3. Route Continuity
    continuity_matches = 0
    if len(predicted_arcs) > 1:
        for i in range(len(predicted_arcs) - 1):
            u = predicted_arcs[i]
            v = predicted_arcs[i + 1]
            if u == v:
                continuity_matches += 1
            elif u < len(arcs_df) and v < len(arcs_df):
                if arcs_df.iloc[u]["to_node"] == arcs_df.iloc[v]["from_node"]:
                    continuity_matches += 1
        continuity = continuity_matches / (len(predicted_arcs) - 1)
    else:
        continuity = 1.0

    return {
        "pointwise_top1_accuracy": round(pointwise_top1, 4),
        "route_continuity": round(continuity, 4),
        "sequence_coverage": round(coverage, 4),
        "sequence_precision": round(precision, 4),
        "total_observations": total_obs,
        "matched_observations": int(pointwise_top1 * total_obs)
    }
