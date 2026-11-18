"""
Person 1: Comprehensive Exploratory Data Analysis & Visualizations.
Analyzes:
1. Trajectory quality & sampling intervals (Person 1 Lead Focus)
2. Missing values, spatial extent, and CRS projection justification
3. Road network topology and connectivity
4. Route-to-track mapping and GPS noise analysis
5. Generates publication-ready figures in docs/figures/ and Folium maps in maps/html/
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins

# Setup aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
    "figure.dpi": 150
})

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "experiments" / "results"
FIG_DIR = BASE_DIR / "docs" / "figures"
MAPS_DIR = BASE_DIR / "maps" / "html"

FIG_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

R_EARTH = 6371000.0

def project_coords(lon, lat, ref_lon, ref_lat):
    ref_lat_rad = np.radians(ref_lat)
    x = np.radians(lon - ref_lon) * R_EARTH * np.cos(ref_lat_rad)
    y = np.radians(lat - ref_lat) * R_EARTH
    return x, y

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx**2 + dy**2
    seg_len_sq = np.where(seg_len_sq < 1e-10, 1e-10, seg_len_sq)
    t = np.clip(((px - x1) * dx + (py - y1) * dy) / seg_len_sq, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

def main():
    print("=" * 80)
    print("Starting Person 1 Comprehensive EDA & Visualization Pipeline")
    print("=" * 80)

    # 1. Load Dataset Inventory
    inv_path = RESULTS_DIR / "dataset_inventory.csv"
    if not inv_path.exists():
        from data.inventory import build_inventory
        inv_df = build_inventory(RAW_DIR, inv_path)
    else:
        inv_df = pd.read_csv(inv_path, dtype={"trip_id": str})
    print(f"Loaded inventory: {len(inv_df)} trips ({len(inv_df[inv_df['split']=='train'])} train, {len(inv_df[inv_df['split']=='student_test'])} student_test)")

    # 2. Detailed Sampling Intervals & GPS Gaps across all trips
    all_dts = []
    gap_summary = []
    
    print("\n[Step 1] Analyzing GPS sampling intervals and dropout patterns...")
    for _, row in inv_df.iterrows():
        split, tid = str(row["split"]), str(row["trip_id"]).zfill(8)
        track_path = RAW_DIR / split / tid / f"{tid}.track"
        if not track_path.exists():
            continue
        track = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "time"])
        dt = track["time"].diff().dropna().values
        all_dts.extend(dt)
        gaps_gt5 = dt[dt > 5.0]
        if len(gaps_gt5) > 0:
            gap_summary.append({
                "trip_id": tid,
                "split": split,
                "n_gaps": len(gaps_gt5),
                "max_gap_s": float(np.max(gaps_gt5)),
                "p95_gap_s": float(np.percentile(gaps_gt5, 95))
            })

    all_dts = np.array(all_dts)
    print(f"Total consecutive observation transitions: {len(all_dts):,}")
    print(f"Exact 1.0s interval: {(all_dts == 1.0).sum():,} ({((all_dts == 1.0).mean()*100):.2f}%)")
    print(f"Gaps > 5.0s: {(all_dts > 5.0).sum():,} ({((all_dts > 5.0).mean()*100):.3f}%)")
    print(f"Gaps > 10.0s: {(all_dts > 10.0).sum():,} ({((all_dts > 10.0).mean()*100):.3f}%)")

    # Figure 1: Sampling Intervals and Dropout Gaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram around 1s
    dts_zoom = all_dts[all_dts <= 10.0]
    sns.histplot(dts_zoom, bins=20, ax=axes[0], color="#2b5c8f", edgecolor="black", kde=False)
    axes[0].set_title(r"GPS Sampling Intervals ($\Delta t \leq 10$s)")
    axes[0].set_xlabel(r"Time Delta $\Delta t$ (seconds)")
    axes[0].set_ylabel("Count of Transitions")
    axes[0].axvline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Nominal 1 Hz (94.3%)")
    axes[0].legend()

    # Right: Boxplot of Gaps > 5s across trips with gaps
    gap_df = pd.DataFrame(gap_summary)
    sns.boxplot(y="max_gap_s", data=gap_df, ax=axes[1], color="#e27c38", width=0.4)
    axes[1].set_yscale("log")
    axes[1].set_title(r"Distribution of Maximum GPS Gaps per Trip (Log Scale)")
    axes[1].set_ylabel(r"Maximum Gap $\Delta t_{max}$ (seconds, log scale)")
    axes[1].set_xlabel(r"Trips with Dropout ($\Delta t > 5$s)")

    plt.tight_layout()
    fig1_path = FIG_DIR / "eda_sampling_intervals.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {fig1_path}")

    # Figure 2: Trajectory Sizes & Track-to-Route Ratios
    print("\n[Step 2] Analyzing trajectory lengths and track/route ratios...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(inv_df["gps_points"], bins=25, ax=axes[0], color="#1d7874", kde=True)
    axes[0].set_title("Distribution of GPS Points per Trajectory")
    axes[0].set_xlabel("Number of GPS Observations")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(inv_df["gps_points"].median(), color="darkred", linestyle="--", label=f"Median: {inv_df['gps_points'].median():.0f}")
    axes[0].legend()

    ratio = inv_df["gps_points"] / inv_df["route_arcs_count"]
    sns.histplot(ratio, bins=25, ax=axes[1], color="#679436", kde=True)
    axes[1].set_title("Track-to-Route Ratio (GPS Observations per Ground-Truth Arc)")
    axes[1].set_xlabel("Ratio: len(.track) / len(.route)")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(ratio.median(), color="darkred", linestyle="--", label=f"Median: {ratio.median():.1f}x")
    axes[1].legend()

    plt.tight_layout()
    fig2_path = FIG_DIR / "eda_trajectory_sizes.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {fig2_path}")

    # Figure 3: Geographic Spatial Extent & Multicity Analysis
    print("\n[Step 3] Analyzing geographical extents...")
    plt.figure(figsize=(10, 6))
    plt.scatter(inv_df["min_lon"], inv_df["min_lat"], c=np.where(inv_df["split"]=="train", "#2b5c8f", "#e27c38"), 
                alpha=0.7, s=50, edgecolors="none")
    plt.scatter([], [], c="#2b5c8f", label="Train (80 trips)")
    plt.scatter([], [], c="#e27c38", label="Student Test (10 trips)")
    plt.title("Geographic Spread of Trips across Global Coordinate Space")
    plt.xlabel("Longitude (Degrees WGS84)")
    plt.ylabel("Latitude (Degrees WGS84)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    fig3_path = FIG_DIR / "eda_geographic_spread.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {fig3_path}")

    # Figure 4: Road Network Properties (Arc Lengths & Densities across Sample Trips)
    print("\n[Step 4] Analyzing road network geometry and arc lengths...")
    sample_trip_ids = ["00000000", "00000001", "00000005", "00000040"]
    sample_arc_lengths = []
    
    for tid in sample_trip_ids:
        split = "train" if tid in inv_df[inv_df["split"]=="train"]["trip_id"].values else "student_test"
        tdir = RAW_DIR / split / tid
        nodes = pd.read_csv(tdir / f"{tid}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
        arcs = pd.read_csv(tdir / f"{tid}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
        
        ref_lon, ref_lat = nodes["lon"].median(), nodes["lat"].median()
        nx, ny = project_coords(nodes["lon"].values, nodes["lat"].values, ref_lon, ref_lat)
        
        fn = arcs["from_node"].values
        tn = arcs["to_node"].values
        lengths = np.sqrt((nx[tn] - nx[fn])**2 + (ny[tn] - ny[fn])**2)
        sample_arc_lengths.extend(lengths)

    sample_arc_lengths = np.array(sample_arc_lengths)
    plt.figure(figsize=(9, 5))
    sns.histplot(sample_arc_lengths[sample_arc_lengths <= 300], bins=30, color="#8b5cf6", edgecolor="black")
    plt.title(r"Road Segment Arc Length Distribution ($\leq 300$ meters)")
    plt.xlabel("Arc Length (meters)")
    plt.ylabel("Number of Road Segments")
    plt.axvline(np.median(sample_arc_lengths), color="red", linestyle="--", label=f"Median Length: {np.median(sample_arc_lengths):.1f}m")
    plt.legend()
    
    fig4_path = FIG_DIR / "eda_road_network_properties.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {fig4_path}")

    # Figure 5: GPS-to-Route Distance Distribution (Noise and Ground-Truth Alignment)
    print("\n[Step 5] Analyzing GPS-to-route perpendicular distances for ground truth alignment...")
    gt_distances = []
    for tid in ["00000001", "00000005", "00000000"]:
        split = "train" if tid in inv_df[inv_df["split"]=="train"]["trip_id"].values else "student_test"
        tdir = RAW_DIR / split / tid
        track = pd.read_csv(tdir / f"{tid}.track", sep=r"\s+", header=None, names=["lon", "lat", "time"])
        nodes = pd.read_csv(tdir / f"{tid}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
        arcs = pd.read_csv(tdir / f"{tid}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
        with open(tdir / f"{tid}.route") as f:
            route = [int(l.strip()) for l in f if l.strip().isdigit()]

        ref_lon, ref_lat = nodes["lon"].median(), nodes["lat"].median()
        nx, ny = project_coords(nodes["lon"].values, nodes["lat"].values, ref_lon, ref_lat)
        gx, gy = project_coords(track["lon"].values, track["lat"].values, ref_lon, ref_lat)

        # Build segments for route
        fn = arcs.iloc[route]["from_node"].values
        tn = arcs.iloc[route]["to_node"].values
        x1_r, y1_r = nx[fn], ny[fn]
        x2_r, y2_r = nx[tn], ny[tn]

        # For each GPS point, compute distance to nearest route arc segment
        for i in range(len(track)):
            dists = point_to_segment_distance(gx[i], gy[i], x1_r, y1_r, x2_r, y2_r)
            gt_distances.append(np.min(dists))

    gt_distances = np.array(gt_distances)
    plt.figure(figsize=(9, 5))
    sns.histplot(gt_distances[gt_distances <= 50], bins=30, color="#059669", kde=True)
    plt.title(r"GPS Noise Distribution: Perpendicular Distance to Reference Route ($\leq 50$m)")
    plt.xlabel("Perpendicular Distance (meters)")
    plt.ylabel("GPS Observations")
    plt.axvline(np.median(gt_distances), color="crimson", linestyle="--", label=f"Median GPS Error: {np.median(gt_distances):.1f}m")
    plt.axvline(np.percentile(gt_distances, 95), color="darkorange", linestyle=":", label=f"95th Percentile: {np.percentile(gt_distances, 95):.1f}m")
    plt.legend()

    fig5_path = FIG_DIR / "eda_gps_to_route_error.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved Figure 5 to {fig5_path}")

    # Step 6: Generate Mandatory Interactive Folium Map
    print("\n[Step 6] Generating interactive Folium visualization for Trip 00000000...")
    target_trip = "00000000"
    tdir = RAW_DIR / "student_test" / target_trip
    track = pd.read_csv(tdir / f"{target_trip}.track", sep=r"\s+", header=None, names=["lon", "lat", "time"])
    nodes = pd.read_csv(tdir / f"{target_trip}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(tdir / f"{target_trip}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
    with open(tdir / f"{target_trip}.route") as f:
        route = [int(l.strip()) for l in f if l.strip().isdigit()]

    # Center map on median coordinates
    center_lat = track["lat"].median()
    center_lon = track["lon"].median()
    
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

    # Layer 1: Road Network Subgraph (arcs near the trajectory bounding box)
    net_group = folium.FeatureGroup(name="Road Network (Nearby Arcs)", show=True)
    margin = 0.015
    min_lon, max_lon = track["lon"].min() - margin, track["lon"].max() + margin
    min_lat, max_lat = track["lat"].min() - margin, track["lat"].max() + margin
    
    # Filter nodes within bbox
    valid_nodes = (nodes["lon"] >= min_lon) & (nodes["lon"] <= max_lon) & (nodes["lat"] >= min_lat) & (nodes["lat"] <= max_lat)
    valid_node_ids = set(nodes[valid_nodes].index)
    
    # Sample arcs to avoid rendering 18,000 lines (which slows down browser)
    sample_arcs = arcs[arcs["from_node"].isin(valid_node_ids) & arcs["to_node"].isin(valid_node_ids)]
    if len(sample_arcs) > 2000:
        sample_arcs = sample_arcs.sample(2000, random_state=42)
        
    for _, arc in sample_arcs.iterrows():
        fn, tn = int(arc["from_node"]), int(arc["to_node"])
        p1 = [nodes.iloc[fn]["lat"], nodes.iloc[fn]["lon"]]
        p2 = [nodes.iloc[tn]["lat"], nodes.iloc[tn]["lon"]]
        folium.PolyLine([p1, p2], color="#94a3b8", weight=1.5, opacity=0.6).add_to(net_group)
    net_group.add_to(fmap)

    # Layer 2: Reference Route (Ground Truth)
    route_group = folium.FeatureGroup(name="Reference Route (.route)", show=True)
    route_coords = []
    for arc_id in route:
        fn, tn = int(arcs.iloc[arc_id]["from_node"]), int(arcs.iloc[arc_id]["to_node"])
        p1 = [nodes.iloc[fn]["lat"], nodes.iloc[fn]["lon"]]
        p2 = [nodes.iloc[tn]["lat"], nodes.iloc[tn]["lon"]]
        folium.PolyLine([p1, p2], color="#16a34a", weight=4, opacity=0.85, tooltip=f"Arc ID: {arc_id}").add_to(route_group)
    route_group.add_to(fmap)

    # Layer 3: Raw GPS Observations (Downsampled by step of 2 for clean rendering)
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    for i in range(0, len(track), 2):
        row = track.iloc[i]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2.5,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.7,
            popup=f"Obs {i} | Time: {row['time']}s"
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(fmap)

    map_html_path = MAPS_DIR / f"eda_trip_{target_trip}_map.html"
    fmap.save(str(map_html_path))
    print(f"Saved interactive Folium map to {map_html_path}")

    print("\n" + "=" * 80)
    print("Person 1 EDA Data Processing and Visualizations Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
