"""
Candidate arc generation per GPS point with candidate miss rate tracking.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from features.spatial_index import RoadNetworkSpatialIndex

def generate_candidates(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_arcs: Optional[List[int]] = None,
    radius_m: float = 60.0
) -> Dict[str, Any]:
    """
    Finds road arc candidates for each GPS point in track_df.
    Computes candidate miss rate if route_arcs is provided.
    """
    spatial_idx = RoadNetworkSpatialIndex(nodes_df, arcs_df)
    
    rows = []
    total_obs = len(track_df)
    miss_count = 0
    
    for obs_idx, row in track_df.iterrows():
        lon, lat, t = row["lon"], row["lat"], row["time"]
        candidate_arc_ids = spatial_idx.query_radius(lon, lat, radius_m=radius_m)
        
        gt_arc = route_arcs[obs_idx] if route_arcs and obs_idx < len(route_arcs) else None
        
        if gt_arc is not None and gt_arc not in candidate_arc_ids:
            miss_count += 1
            
        for arc_id in candidate_arc_ids:
            is_positive = 1 if (gt_arc is not None and arc_id == gt_arc) else 0
            rows.append({
                "obs_idx": obs_idx,
                "gps_lon": lon,
                "gps_lat": lat,
                "gps_time": t,
                "arc_id": arc_id,
                "from_node": arcs_df.iloc[arc_id]["from_node"],
                "to_node": arcs_df.iloc[arc_id]["to_node"],
                "label": is_positive
            })
            
    candidates_df = pd.DataFrame(rows)
    miss_rate = (miss_count / max(total_obs, 1)) if route_arcs else 0.0
    
    return {
        "candidates_df": candidates_df,
        "candidate_miss_rate": miss_rate,
        "miss_count": miss_count,
        "total_observations": total_obs
    }
