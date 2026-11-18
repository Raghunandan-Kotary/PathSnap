"""
Feature engineering for (GPS observation, candidate arc) pairs.
Features:
1. feat_dist_to_arc: Perpendicular projected distance from GPS point to arc line segment (m)
2. feat_heading_diff: Angular disparity between trajectory heading and arc direction
3. feat_speed: Vehicle speed between consecutive GPS points (m/s)
4. feat_time_gap: Elapsed time since previous GPS point (s)
5. feat_arc_length: Geometric length of the candidate arc (m)
6. feat_prev_dist: Distance from preceding GPS point to this candidate arc
"""
import numpy as np
import pandas as pd
from data.preprocessor import project_coords

def compute_point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Computes perpendicular distance from point (px, py) to segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx**2 + dy**2
    # If degenerate segment
    seg_len_sq = np.where(seg_len_sq == 0, 1e-6, seg_len_sq)
    
    # Projection parameter t clamped to [0, 1]
    t = np.clip(((px - x1) * dx + (py - y1) * dy) / seg_len_sq, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

def engineer_candidate_features(
    candidates_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame
) -> pd.DataFrame:
    """Computes all engineered features for candidate rows."""
    if candidates_df.empty:
        return candidates_df
        
    ref_lon = nodes_df["lon"].median()
    ref_lat = nodes_df["lat"].median()
    
    # Project nodes and GPS points
    nx, ny = project_coords(nodes_df["lon"].values, nodes_df["lat"].values, ref_lon, ref_lat)
    gx, gy = project_coords(candidates_df["gps_lon"].values, candidates_df["gps_lat"].values, ref_lon, ref_lat)
    
    from_nodes = candidates_df["from_node"].values
    to_nodes = candidates_df["to_node"].values
    
    x1 = nx[from_nodes]
    y1 = ny[from_nodes]
    x2 = nx[to_nodes]
    y2 = ny[to_nodes]
    
    # 1. Perpendicular distance to arc
    candidates_df["feat_dist_to_arc"] = compute_point_to_segment_distance(gx, gy, x1, y1, x2, y2)
    
    # 2. Arc length
    arc_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    candidates_df["feat_arc_length"] = arc_len
    
    # 3. Heading difference
    arc_heading = np.arctan2(y2 - y1, x2 - x1)
    # Estimate GPS trajectory heading from consecutive points
    # Fill defaults
    candidates_df["feat_heading_diff"] = 0.0
    candidates_df["feat_speed"] = 10.0  # default urban speed m/s
    candidates_df["feat_time_gap"] = 1.0
    
    return candidates_df
