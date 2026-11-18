"""
Coordinate transformations and data preprocessing for PathSnap.
Applies Transverse Mercator / local UTM projection or geodesic distance calculations.
Degree-based Euclidean distance is strictly prohibited.
"""
import numpy as np
import pandas as pd
from typing import Tuple

# WGS84 Ellipsoid constants for local flat-earth / UTM projection
R_EARTH = 6371000.0  # meters

def project_coords(lon: np.ndarray, lat: np.ndarray, ref_lon: float, ref_lat: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects (lon, lat) degrees to local Cartesian coordinates (x, y) in meters
    using an Equirectangular / Transverse Mercator approximation centered at reference origin.
    Prevents latitude degree distortion.
    """
    ref_lat_rad = np.radians(ref_lat)
    x = np.radians(lon - ref_lon) * R_EARTH * np.cos(ref_lat_rad)
    y = np.radians(lat - ref_lat) * R_EARTH
    return x, y

def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes great-circle distance between two points in meters."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return float(2 * R_EARTH * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def check_timestamp_continuity(track_df: pd.DataFrame, max_gap_s: float = 5.0) -> pd.DataFrame:
    """Identifies temporal gaps and sampling intervals in GPS track."""
    dt = track_df["time"].diff().fillna(1.0)
    track_df["delta_time"] = dt
    track_df["is_large_gap"] = dt > max_gap_s
    return track_df
