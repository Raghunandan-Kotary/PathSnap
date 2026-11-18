"""
Feature engineering for (GPS observation, candidate arc) pairs.
Features:
1. feat_dist_to_arc: Perpendicular projected distance from GPS point to arc line segment (m)
2. feat_arc_length: Geometric length of the candidate arc (m)
3. feat_heading_diff: Angular disparity between trajectory heading and arc direction [0, pi]
4. feat_speed: Vehicle speed between consecutive GPS points (m/s)
5. feat_time_gap: Elapsed time since previous GPS point (s)
6. feat_prev_dist: Distance from preceding GPS point to this candidate arc (m)
"""
from typing import Tuple
import numpy as np
import pandas as pd
from data.preprocessor import project_coords

def compute_point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Computes perpendicular distance from point (px, py) to segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx**2 + dy**2
    seg_len_sq = np.where(seg_len_sq < 1e-10, 1e-10, seg_len_sq)
    
    t = np.clip(((px - x1) * dx + (py - y1) * dy) / seg_len_sq, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

def compute_heading_diff(traj_heading: np.ndarray, arc_heading: np.ndarray) -> np.ndarray:
    """Computes absolute angular difference folded into [0, pi]."""
    diff = np.abs(traj_heading - arc_heading) % (2 * np.pi)
    diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
    return diff

def engineer_candidate_features(
    candidates_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    track_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Computes all 6 engineered features for candidate rows.
    """
    if candidates_df.empty:
        return candidates_df
        
    ref_lon = float(nodes_df["lon"].median())
    ref_lat = float(nodes_df["lat"].median())
    
    # Project nodes
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
    dx_arc = x2 - x1
    dy_arc = y2 - y1
    arc_len = np.sqrt(dx_arc**2 + dy_arc**2)
    candidates_df["feat_arc_length"] = arc_len
    
    # Candidate arc heading
    arc_heading = np.arctan2(dy_arc, dx_arc)
    
    # 3-6. Trajectory temporal and heading properties
    # If track_df is available, precompute trajectory properties per obs_idx
    if track_df is not None and len(track_df) > 0:
        t_gx, t_gy = project_coords(track_df["lon"].values, track_df["lat"].values, ref_lon, ref_lat)
        t_time = track_df["time"].values
        N = len(track_df)
        
        # Heading per observation
        t_heading = np.zeros(N, dtype=np.float64)
        t_speed = np.full(N, 10.0, dtype=np.float64)  # default urban speed 10 m/s
        t_dt = np.ones(N, dtype=np.float64)
        
        # Forward difference for i=0
        if N > 1:
            dx0 = t_gx[1] - t_gx[0]
            dy0 = t_gy[1] - t_gy[0]
            t_heading[0] = np.arctan2(dy0, dx0)
            
            # Backward difference for i >= 1
            dx_back = np.diff(t_gx)
            dy_back = np.diff(t_gy)
            dt_back = np.diff(t_time)
            dt_back = np.where(dt_back <= 0, 1.0, dt_back)
            
            dists = np.sqrt(dx_back**2 + dy_back**2)
            speeds = dists / dt_back
            
            # Clamp speeds during dropouts
            speeds = np.where((dt_back > 5.0) | (speeds > 50.0), 10.0, speeds)
            
            t_heading[1:] = np.arctan2(dy_back, dx_back)
            t_speed[1:] = speeds
            t_dt[1:] = dt_back
            
        # Map onto candidate rows by obs_idx
        obs_indices = candidates_df["obs_idx"].values
        cand_traj_heading = t_heading[obs_indices]
        candidates_df["feat_heading_diff"] = compute_heading_diff(cand_traj_heading, arc_heading)
        candidates_df["feat_speed"] = t_speed[obs_indices]
        candidates_df["feat_time_gap"] = t_dt[obs_indices]
        
        # 6. feat_prev_dist: distance from previous GPS point to current candidate arc
        prev_obs_indices = np.maximum(0, obs_indices - 1)
        prev_gx = t_gx[prev_obs_indices]
        prev_gy = t_gy[prev_obs_indices]
        candidates_df["feat_prev_dist"] = compute_point_to_segment_distance(prev_gx, prev_gy, x1, y1, x2, y2)
        
    else:
        # Fallbacks when track_df is not passed
        candidates_df["feat_heading_diff"] = 0.0
        candidates_df["feat_speed"] = 10.0
        candidates_df["feat_time_gap"] = 1.0
        candidates_df["feat_prev_dist"] = candidates_df["feat_dist_to_arc"]
        
    return candidates_df
