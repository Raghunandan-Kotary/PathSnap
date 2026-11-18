"""
Folium map generation for PathSnap (Person 3).

Builds one interactive map per trip with four toggleable layers: raw GPS
points, the road network, the reference route (server-side only) and the
predicted route, per docs/INTERFACES.md Interface 4.2.
"""
from pathlib import Path
from typing import List, Optional

import folium
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data.preprocessor import project_coords

GPS_COLOR = "#1E88E5"       # blue dots: raw GPS
ROAD_COLOR = "#9E9E9E"      # gray lines: road network
REFERENCE_COLOR = "#2E7D32"  # green line: reference route
PREDICTED_COLOR = "#D32F2F"  # red line: predicted route


def _arc_coords(arc_id: int, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> Optional[tuple]:
    if arc_id < 0 or arc_id >= len(arcs_df):
        return None
    arc = arcs_df.iloc[arc_id]
    frm, to = nodes_df.iloc[int(arc["from_node"])], nodes_df.iloc[int(arc["to_node"])]
    return (frm["lat"], frm["lon"]), (to["lat"], to["lon"])


def _arcs_near_track(
    track_df: pd.DataFrame, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame, radius_m: float
) -> "np.ndarray":
    """
    Boolean mask over arcs_df of arcs whose midpoint lies within radius_m of
    *any* GPS point. Needed because a long trip's bounding box can cover
    almost the whole region-wide .nodes/.arcs extract (tens of thousands of
    arcs), while the road actually touched by the trip is a thin corridor
    around the GPS trace.
    """
    ref_lon, ref_lat = nodes_df["lon"].median(), nodes_df["lat"].median()
    node_x, node_y = project_coords(nodes_df["lon"].values, nodes_df["lat"].values, ref_lon, ref_lat)

    from_idx, to_idx = arcs_df["from_node"].values, arcs_df["to_node"].values
    mid_x = (node_x[from_idx] + node_x[to_idx]) / 2.0
    mid_y = (node_y[from_idx] + node_y[to_idx]) / 2.0
    arc_tree = cKDTree(np.column_stack([mid_x, mid_y]))

    track_x, track_y = project_coords(track_df["lon"].values, track_df["lat"].values, ref_lon, ref_lat)
    nearby_lists = arc_tree.query_ball_point(np.column_stack([track_x, track_y]), r=radius_m)

    mask = np.zeros(len(arcs_df), dtype=bool)
    for idx_list in nearby_lists:
        mask[idx_list] = True
    return mask


def generate_trip_map(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    predicted_route: List[int],
    ground_truth_route: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
    road_corridor_radius_m: float = 150.0,
) -> str:
    """
    Returns the map as an HTML string (suitable for embedding directly in
    the Streamlit UI via `st.components.v1.html`, per `map_html` in
    api/schemas.py). If `save_path` is given, also writes the same HTML
    there (e.g. maps/html/<trip_id>.html) for static serving.

    Layers (toggleable via LayerControl):
      - Raw GPS points (blue dots)
      - Road network (gray lines)
      - Reference route (green line) -- only if ground_truth_route is given;
        omit when calling from anywhere the UI can see the response (ground
        truth must never leave the server, per docs/SYSTEM_ARCHITECTURE.md).
      - Predicted route (red line)
    """
    center_lat, center_lon = track_df["lat"].mean(), track_df["lon"].mean()
    fmap = folium.Map(
        location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap", prefer_canvas=True
    )

    gps_layer = folium.FeatureGroup(name="Raw GPS points", show=True)
    for _, row in track_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2,
            color=GPS_COLOR,
            fill=True,
            fill_opacity=0.8,
            weight=1,
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    # The .nodes/.arcs files are a region-wide network extract -- often
    # 10,000-150,000+ arcs -- while even a long trip only ever touches a
    # thin corridor of it (a bounding-box filter doesn't help: a cross-city
    # trip's bbox can cover almost the entire extract). Drawing every arc
    # would make the map unusably slow and the HTML file enormous, so only
    # arcs within road_corridor_radius_m of some GPS point are rendered.
    in_view = _arcs_near_track(track_df, nodes_df, arcs_df, road_corridor_radius_m)
    from_lat = nodes_df["lat"].values[arcs_df["from_node"].values]
    from_lon = nodes_df["lon"].values[arcs_df["from_node"].values]
    to_lat = nodes_df["lat"].values[arcs_df["to_node"].values]
    to_lon = nodes_df["lon"].values[arcs_df["to_node"].values]

    road_layer = folium.FeatureGroup(name="Road network", show=True)
    for flat, flon, tlat, tlon in zip(from_lat[in_view], from_lon[in_view], to_lat[in_view], to_lon[in_view]):
        folium.PolyLine(
            locations=[(flat, flon), (tlat, tlon)],
            color=ROAD_COLOR,
            weight=1,
            opacity=0.5,
        ).add_to(road_layer)
    road_layer.add_to(fmap)

    if ground_truth_route:
        gt_layer = folium.FeatureGroup(name="Reference route", show=True)
        for arc_id in ground_truth_route:
            coords = _arc_coords(arc_id, nodes_df, arcs_df)
            if coords:
                folium.PolyLine(coords, color=REFERENCE_COLOR, weight=4, opacity=0.9).add_to(gt_layer)
        gt_layer.add_to(fmap)

    pred_layer = folium.FeatureGroup(name="Predicted route", show=True)
    for arc_id in predicted_route:
        coords = _arc_coords(arc_id, nodes_df, arcs_df)
        if coords:
            folium.PolyLine(coords, color=PREDICTED_COLOR, weight=4, opacity=0.9, dash_array="6,4").add_to(pred_layer)
    pred_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    _add_legend(fmap, show_reference=bool(ground_truth_route))

    # Bound on the GPS track, not the full road-network node table: the
    # .nodes file is a region-wide extract that typically covers a much
    # larger area than any single trip, so including it here would zoom
    # out far past the actual route.
    fmap.fit_bounds([
        [track_df["lat"].min(), track_df["lon"].min()],
        [track_df["lat"].max(), track_df["lon"].max()],
    ])

    html = fmap.get_root().render()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(html, encoding="utf-8")
    return html


def _add_legend(fmap: folium.Map, show_reference: bool) -> None:
    reference_row = (
        f'<div><span style="background:{REFERENCE_COLOR};width:14px;height:4px;'
        'display:inline-block;margin-right:6px;"></span>Reference route</div>'
        if show_reference else ""
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: white; padding: 10px 12px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 13px;">
      <div style="font-weight:600; margin-bottom:4px;">PathSnap map legend</div>
      <div><span style="background:{GPS_COLOR};width:10px;height:10px;border-radius:50%;
           display:inline-block;margin-right:6px;"></span>Raw GPS points</div>
      <div><span style="background:{ROAD_COLOR};width:14px;height:4px;
           display:inline-block;margin-right:6px;"></span>Road network</div>
      {reference_row}
      <div><span style="background:{PREDICTED_COLOR};width:14px;height:4px;
           display:inline-block;margin-right:6px;"></span>Predicted route</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
