"""
Spatial indexing for fast road-network candidate querying.
Uses SciPy KDTree over projected road-arc line segments (sampled at start, midpoint, and end)
to guarantee minimal candidate miss rate even on long road segments.
"""
from typing import List, Set
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from data.preprocessor import project_coords

class RoadNetworkSpatialIndex:
    def __init__(self, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame):
        self.nodes_df = nodes_df
        self.arcs_df = arcs_df
        
        # Reference coordinates for local projection
        self.ref_lon = float(nodes_df["lon"].median())
        self.ref_lat = float(nodes_df["lat"].median())
        
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
        
        # Multi-point arc sampling: from_node, midpoint, to_node
        mid_x = (self.arc_x1 + self.arc_x2) / 2.0
        mid_y = (self.arc_y1 + self.arc_y2) / 2.0
        
        n_arcs = len(arcs_df)
        pts = np.empty((n_arcs * 3, 2), dtype=np.float64)
        pts[0::3, 0] = self.arc_x1
        pts[0::3, 1] = self.arc_y1
        pts[1::3, 0] = mid_x
        pts[1::3, 1] = mid_y
        pts[2::3, 0] = self.arc_x2
        pts[2::3, 1] = self.arc_y2
        
        self.kdtree = cKDTree(pts)
        self.point_to_arc = np.repeat(np.arange(n_arcs, dtype=np.int32), 3)

    def query_radius(self, lon: float, lat: float, radius_m: float = 60.0, k_fallback: int = 5) -> List[int]:
        """
        Queries candidate arcs within radius_m. If empty, falls back to top k nearest arcs.
        Returns unique arc IDs.
        """
        qx, qy = project_coords(np.array([lon]), np.array([lat]), self.ref_lon, self.ref_lat)
        point = np.array([qx[0], qy[0]])
        
        indices = self.kdtree.query_ball_point(point, r=radius_m)
        if not indices:
            _, indices = self.kdtree.query(point, k=k_fallback * 3)
            indices = list(np.atleast_1d(indices))
            
        candidate_arcs = list(dict.fromkeys(self.point_to_arc[indices]))
        return candidate_arcs
