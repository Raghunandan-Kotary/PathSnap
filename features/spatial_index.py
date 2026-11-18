"""
Spatial indexing for fast road-network candidate querying.
Uses SciPy KDTree over projected road-arc line segments.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import List, Tuple
from data.preprocessor import project_coords

class RoadNetworkSpatialIndex:
    def __init__(self, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame):
        self.nodes_df = nodes_df
        self.arcs_df = arcs_df
        
        # Reference coordinates for local projection
        self.ref_lon = nodes_df["lon"].median()
        self.ref_lat = nodes_df["lat"].median()
        
        # Project all nodes
        nx, ny = project_coords(nodes_df["lon"].values, nodes_df["lat"].values, self.ref_lon, self.ref_lat)
        self.nodes_x = nx
        self.nodes_y = ny
        
        # Arc segment endpoints
        from_idx = arcs_df["from_node"].values
        to_idx = arcs_df["to_node"].values
        
        self.arc_x1 = nx[from_idx]
        self.arc_y1 = ny[from_idx]
        self.arc_x2 = nx[to_idx]
        self.arc_y2 = ny[to_idx]
        
        # Arc midpoints for KDTree radius querying
        mid_x = (self.arc_x1 + self.arc_x2) / 2.0
        mid_y = (self.arc_y1 + self.arc_y2) / 2.0
        
        self.kdtree = cKDTree(np.column_stack([mid_x, mid_y]))

    def query_radius(self, lon: float, lat: float, radius_m: float = 60.0, k_fallback: int = 5) -> List[int]:
        """
        Queries candidate arcs within radius_m. If empty, falls back to top k nearest.
        """
        qx, qy = project_coords(np.array([lon]), np.array([lat]), self.ref_lon, self.ref_lat)
        point = np.array([qx[0], qy[0]])
        
        indices = self.kdtree.query_ball_point(point, r=radius_m)
        if not indices:
            _, indices = self.kdtree.query(point, k=k_fallback)
            indices = list(np.atleast_1d(indices))
        return indices
