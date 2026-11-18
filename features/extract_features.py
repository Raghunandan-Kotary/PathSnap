"""
Frozen inference-time feature extraction pipeline for PathSnap.
Shared across offline model training, evaluation harness, and online FastAPI serving.
"""
from typing import Optional, List
import pandas as pd
from features.candidate_generator import generate_candidates
from features.feature_engineering import engineer_candidate_features

def generate_trip_candidates_and_features(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_arcs: Optional[List[int]] = None,
    search_radius_m: float = 60.0
) -> pd.DataFrame:
    """
    Given raw trip dataframes, generates spatial candidate arcs for each GPS observation
    and returns a DataFrame containing all engineered features.
    If route_arcs is provided, assigns ground-truth binary labels (1 = match, 0 = non-match).
    """
    cand_result = generate_candidates(
        track_df=track_df,
        nodes_df=nodes_df,
        arcs_df=arcs_df,
        route_arcs=route_arcs,
        radius_m=search_radius_m
    )
    candidates_df = cand_result["candidates_df"]
    features_df = engineer_candidate_features(
        candidates_df=candidates_df,
        nodes_df=nodes_df,
        arcs_df=arcs_df,
        track_df=track_df
    )
    return features_df
