"""
Data loading utilities for PathSnap trajectory and road-network files.
Handles .track, .nodes, .arcs, and .route formats.
"""
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import pandas as pd

def load_track(track_path: Path) -> pd.DataFrame:
    """
    Loads a .track file.
    Columns: lon (deg), lat (deg), time (seconds from start / timestamp)
    """
    df = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "time"],
        dtype={"lon": float, "lat": float, "time": float}
    )
    df["obs_idx"] = range(len(df))
    return df

def load_nodes(nodes_path: Path) -> pd.DataFrame:
    """
    Loads a .nodes file.
    Row index (0-based) represents node_id.
    Columns: lon (deg), lat (deg)
    """
    df = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": float, "lat": float}
    )
    df["node_id"] = range(len(df))
    return df

def load_arcs(arcs_path: Path) -> pd.DataFrame:
    """
    Loads an .arcs file.
    Row index (0-based) represents arc_id.
    Columns: from_node, to_node (integer indices into .nodes)
    """
    df = pd.read_csv(
        arcs_path,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={"from_node": int, "to_node": int}
    )
    df["arc_id"] = range(len(df))
    return df

def load_route(route_path: Path) -> List[int]:
    """
    Loads a .route reference file.
    Single column: chronological ground-truth sequence of traversed arc_ids.
    """
    if not Path(route_path).exists():
        return []
    with open(route_path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]

def load_trip(trip_dir: Path, trip_id: str) -> Dict[str, Any]:
    """Loads all 4 files for a trip into a dictionary."""
    trip_path = Path(trip_dir)
    return {
        "trip_id": trip_id,
        "track": load_track(trip_path / f"{trip_id}.track"),
        "nodes": load_nodes(trip_path / f"{trip_id}.nodes"),
        "arcs": load_arcs(trip_path / f"{trip_id}.arcs"),
        "route": load_route(trip_path / f"{trip_id}.route")
    }
