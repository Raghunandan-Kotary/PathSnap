"""
Folium Geographical Map Generator for PathSnap.
Renders interactive HTML maps with layer controls:
- Layer 1: Raw GPS Observations (Blue CircleMarkers)
- Layer 2: Road Network Geometry (Subtle Gray lines)
- Layer 3: Reference Route (Green PolyLine - Ground Truth)
- Layer 4: Predicted Reconstructed Route (Red PolyLine)
"""
from typing import List, Optional
import folium
from folium import plugins
import pandas as pd
import numpy as np

def generate_folium_map(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    predicted_arcs: List[int],
    ground_truth_arcs: Optional[List[int]] = None,
    trip_id: str = "Trip",
    max_network_arcs: int = 600
) -> folium.Map:
    """
    Generates an interactive Folium map centered on the trajectory.
    """
    # Calculate bounding box and center
    center_lat = float(track_df["lat"].median())
    center_lon = float(track_df["lon"].median())
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )
    
    # 1. Road Network Layer (Background context)
    fg_network = folium.FeatureGroup(name="Road Network (Context)", show=False)
    # Filter road arcs near the trip bounding box
    min_lat, max_lat = track_df["lat"].min() - 0.005, track_df["lat"].max() + 0.005
    min_lon, max_lon = track_df["lon"].min() - 0.005, track_df["lon"].max() + 0.005
    
    sampled_count = 0
    for arc_id in range(len(arcs_df)):
        u = int(arcs_df.iloc[arc_id]["from_node"])
        v = int(arcs_df.iloc[arc_id]["to_node"])
        if u < len(nodes_df) and v < len(nodes_df):
            u_lat, u_lon = nodes_df.iloc[u]["lat"], nodes_df.iloc[u]["lon"]
            v_lat, v_lon = nodes_df.iloc[v]["lat"], nodes_df.iloc[v]["lon"]
            if (min_lat <= u_lat <= max_lat and min_lon <= u_lon <= max_lon):
                folium.PolyLine(
                    locations=[(u_lat, u_lon), (v_lat, v_lon)],
                    color="#9CA3AF",
                    weight=1.5,
                    opacity=0.6,
                    tooltip=f"Arc {arc_id}"
                ).add_to(fg_network)
                sampled_count += 1
                if sampled_count >= max_network_arcs:
                    break
    fg_network.add_to(m)

    # 2. Reference Route (Ground Truth) Layer
    if ground_truth_arcs and len(ground_truth_arcs) > 0:
        fg_gt = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        gt_coords = []
        for arc_id in ground_truth_arcs:
            if arc_id < len(arcs_df):
                u = int(arcs_df.iloc[arc_id]["from_node"])
                v = int(arcs_df.iloc[arc_id]["to_node"])
                if u < len(nodes_df) and v < len(nodes_df):
                    gt_coords.append((nodes_df.iloc[u]["lat"], nodes_df.iloc[u]["lon"]))
                    gt_coords.append((nodes_df.iloc[v]["lat"], nodes_df.iloc[v]["lon"]))
        if gt_coords:
            folium.PolyLine(
                locations=gt_coords,
                color="#10B981",  # Emerald Green
                weight=5,
                opacity=0.85,
                tooltip=f"Ground Truth Route ({len(ground_truth_arcs)} arcs)"
            ).add_to(fg_gt)
        fg_gt.add_to(m)

    # 3. Predicted Route Layer
    if predicted_arcs and len(predicted_arcs) > 0:
        fg_pred = folium.FeatureGroup(name="Predicted Route (PathSnap)", show=True)
        pred_coords = []
        # Collapse consecutive duplicate arcs for rendering
        import itertools
        collapsed_arcs = [k for k, _ in itertools.groupby(predicted_arcs)]
        for arc_id in collapsed_arcs:
            if arc_id < len(arcs_df):
                u = int(arcs_df.iloc[arc_id]["from_node"])
                v = int(arcs_df.iloc[arc_id]["to_node"])
                if u < len(nodes_df) and v < len(nodes_df):
                    pred_coords.append((nodes_df.iloc[u]["lat"], nodes_df.iloc[u]["lon"]))
                    pred_coords.append((nodes_df.iloc[v]["lat"], nodes_df.iloc[v]["lon"]))
        if pred_coords:
            folium.PolyLine(
                locations=pred_coords,
                color="#EF4444",  # Crimson Red
                weight=3.5,
                opacity=0.9,
                dash_array="6, 4",
                tooltip=f"Predicted Reconstructed Route ({len(collapsed_arcs)} arcs)"
            ).add_to(fg_pred)
        fg_pred.add_to(m)

    # 4. Raw GPS Observations Layer
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (Noisy)", show=True)
    # Downsample points slightly if trajectory is huge for responsive browser rendering
    step = max(1, len(track_df) // 400)
    for idx in range(0, len(track_df), step):
        row = track_df.iloc[idx]
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3,
            color="#2563EB",  # Royal Blue
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=0.75,
            tooltip=f"GPS Obs #{idx} | t={row['time']}s"
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # Add Layer Control and Title Legend
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Fit bounds to trajectory
    m.fit_bounds([
        [track_df["lat"].min(), track_df["lon"].min()],
        [track_df["lat"].max(), track_df["lon"].max()]
    ])
    
    return m
