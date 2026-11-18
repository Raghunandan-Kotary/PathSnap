"""
Candidate arc generation per GPS point with candidate miss rate tracking.
Assigns ground-truth labels by projecting GPS observations onto the traversed route sequence.
"""
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from data.preprocessor import project_coords
from features.spatial_index import RoadNetworkSpatialIndex

def match_gps_to_route_arcs(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_arcs: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assigns each GPS observation to its ground-truth arc in the route sequence.
    Uses windowed sequential projection onto route segments to respect temporal order.
    Returns (matched_arc_ids, perpendicular_distances).
    """
    ref_lon = float(nodes_df["lon"].median())
    ref_lat = float(nodes_df["lat"].median())
    
    nx, ny = project_coords(nodes_df["lon"].values, nodes_df["lat"].values, ref_lon, ref_lat)
    gx, gy = project_coords(track_df["lon"].values, track_df["lat"].values, ref_lon, ref_lat)
    
    fn = arcs_df.iloc[route_arcs]["from_node"].values
    tn = arcs_df.iloc[route_arcs]["to_node"].values
    
    x1, y1 = nx[fn], ny[fn]
    x2, y2 = nx[tn], ny[tn]
    dx, dy = x2 - x1, y2 - y1
    seg_lens_sq = np.where(dx*dx + dy*dy < 1e-10, 1e-10, dx*dx + dy*dy)
    
    M = len(route_arcs)
    N = len(track_df)
    
    matched_arcs = np.empty(N, dtype=np.int32)
    matched_dists = np.empty(N, dtype=np.float64)
    
    curr_route_idx = 0
    window_ahead = 20
    window_behind = 5
    
    for i in range(N):
        px, py = gx[i], gy[i]
        
        w_start = max(0, curr_route_idx - window_behind)
        w_end = min(M, curr_route_idx + window_ahead)
        
        sub_dx = dx[w_start:w_end]
        sub_dy = dy[w_start:w_end]
        sub_x1 = x1[w_start:w_end]
        sub_y1 = y1[w_start:w_end]
        sub_lens = seg_lens_sq[w_start:w_end]
        
        t = np.clip(((px - sub_x1)*sub_dx + (py - sub_y1)*sub_dy) / sub_lens, 0.0, 1.0)
        proj_x = sub_x1 + t * sub_dx
        proj_y = sub_y1 + t * sub_dy
        dists = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        
        best_local_idx = int(np.argmin(dists))
        best_route_idx = w_start + best_local_idx
        
        if best_route_idx > curr_route_idx:
            curr_route_idx = best_route_idx
            
        matched_arcs[i] = route_arcs[best_route_idx]
        matched_dists[i] = dists[best_local_idx]
        
    return matched_arcs, matched_dists

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
    
    gt_arc_per_obs = None
    if route_arcs is not None and len(route_arcs) > 0:
        gt_arc_per_obs, _ = match_gps_to_route_arcs(track_df, nodes_df, arcs_df, route_arcs)
        
    rows = []
    total_obs = len(track_df)
    miss_count = 0
    
    from_nodes = arcs_df["from_node"].values
    to_nodes = arcs_df["to_node"].values
    
    for obs_idx, row in track_df.iterrows():
        lon, lat, t = row["lon"], row["lat"], row["time"]
        candidate_arc_ids = spatial_idx.query_radius(lon, lat, radius_m=radius_m)
        
        gt_arc = int(gt_arc_per_obs[obs_idx]) if gt_arc_per_obs is not None else None
        
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
                "from_node": from_nodes[arc_id],
                "to_node": to_nodes[arc_id],
                "label": is_positive
            })
            
    candidates_df = pd.DataFrame(rows)
    miss_rate = (miss_count / max(total_obs, 1)) if gt_arc_per_obs is not None else 0.0
    
    return {
        "candidates_df": candidates_df,
        "candidate_miss_rate": miss_rate,
        "miss_count": miss_count,
        "total_observations": total_obs
    }
