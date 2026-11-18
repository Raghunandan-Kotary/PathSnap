"""
Candidate arc generation per GPS point with candidate miss rate tracking.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from data.preprocessor import haversine_distance
from features.spatial_index import RoadNetworkSpatialIndex


def _arc_midpoint_lonlat(arc_row: pd.Series, nodes_df: pd.DataFrame) -> tuple:
    frm = nodes_df.iloc[int(arc_row["from_node"])]
    to = nodes_df.iloc[int(arc_row["to_node"])]
    return (frm["lon"] + to["lon"]) / 2.0, (frm["lat"] + to["lat"]) / 2.0


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

    # IMPORTANT: .route is the ordered list of DISTINCT arcs the vehicle
    # traversed, not one entry per GPS fix -- it is routinely 3-30x shorter
    # than .track (one arc can span many consecutive GPS points). Indexing
    # it by obs_idx (as an earlier version of this function did) compares
    # two unrelated sequences and silently mislabels almost everything.
    # Without a full temporal alignment between the GPS trace and the route
    # sequence, the correct, well-defined proxy is set membership: a
    # candidate is the positive match if its arc_id is one of the arcs the
    # vehicle is known to have driven on, full stop.
    route_arc_set = set(route_arcs) if route_arcs else None

    rows = []
    total_obs = len(track_df)
    miss_count = 0

    for obs_idx, row in track_df.iterrows():
        lon, lat, t = row["lon"], row["lat"], row["time"]
        candidate_arc_ids = spatial_idx.query_radius(lon, lat, radius_m=radius_m)

        member_arc_ids = [a for a in candidate_arc_ids if route_arc_set is not None and a in route_arc_set]
        if route_arc_set is not None and not member_arc_ids:
            miss_count += 1

        # Several route arcs (visited at other times in the trip, e.g. the
        # arc just entered and the one about to be left at a junction) can
        # legitimately fall within radius_m of a single GPS point. Only the
        # geometrically nearest one is the actual match for *this*
        # observation -- picking that single nearest arc, rather than
        # labeling every route-member candidate positive, keeps exactly one
        # positive per observation and avoids teaching the classifier to
        # score multiple candidates high at once.
        true_arc_id = None
        if member_arc_ids:
            dists = [
                haversine_distance(
                    lon, lat,
                    *_arc_midpoint_lonlat(arcs_df.iloc[a], nodes_df)
                )
                for a in member_arc_ids
            ]
            true_arc_id = member_arc_ids[int(np.argmin(dists))]

        for arc_id in candidate_arc_ids:
            is_positive = 1 if (true_arc_id is not None and arc_id == true_arc_id) else 0
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
