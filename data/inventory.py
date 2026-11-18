"""
Scans data/raw/ directories (train, student_test) and compiles trip counts,
row counts, file sizes, spatial extents, temporal gaps, and missingness metrics into a summary report.
Outputs CSV to experiments/results/dataset_inventory.csv.
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np

def build_inventory(raw_dir: Path = Path("data/raw"), output_csv: Path = Path("experiments/results/dataset_inventory.csv")):
    inventory = []
    
    for split in ["train", "student_test"]:
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue
        for trip_dir in sorted(split_dir.iterdir()):
            if not trip_dir.is_dir() or trip_dir.name.startswith("."):
                continue
            trip_id = trip_dir.name
            track_file = trip_dir / f"{trip_id}.track"
            nodes_file = trip_dir / f"{trip_id}.nodes"
            arcs_file = trip_dir / f"{trip_id}.arcs"
            route_file = trip_dir / f"{trip_id}.route"
            
            # File existence and sizes
            track_bytes = track_file.stat().st_size if track_file.exists() else 0
            nodes_bytes = nodes_file.stat().st_size if nodes_file.exists() else 0
            arcs_bytes = arcs_file.stat().st_size if arcs_file.exists() else 0
            route_bytes = route_file.stat().st_size if route_file.exists() else 0
            total_kb = (track_bytes + nodes_bytes + arcs_bytes + route_bytes) / 1024.0

            # Inspect track
            gps_points = 0
            min_lon = max_lon = min_lat = max_lat = np.nan
            duration_s = 0.0
            pct_1s = 0.0
            n_gaps_gt5 = 0
            max_gap_s = 0.0

            if track_file.exists() and track_bytes > 0:
                track_df = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "time"])
                gps_points = len(track_df)
                if gps_points > 0:
                    min_lon = track_df["lon"].min()
                    max_lon = track_df["lon"].max()
                    min_lat = track_df["lat"].min()
                    max_lat = track_df["lat"].max()
                    duration_s = track_df["time"].max() - track_df["time"].min()
                    dt = track_df["time"].diff().dropna()
                    if len(dt) > 0:
                        pct_1s = float((dt == 1.0).mean() * 100.0)
                        n_gaps_gt5 = int((dt > 5.0).sum())
                        max_gap_s = float(dt.max())

            # Inspect nodes
            nodes_count = sum(1 for _ in open(nodes_file, "r", encoding="utf-8")) if nodes_file.exists() else 0

            # Inspect arcs
            arcs_count = sum(1 for _ in open(arcs_file, "r", encoding="utf-8")) if arcs_file.exists() else 0

            # Inspect route
            route_count = 0
            unique_route_arcs = 0
            if route_file.exists():
                with open(route_file, "r", encoding="utf-8") as f:
                    route_arcs = [int(line.strip()) for line in f if line.strip().isdigit()]
                route_count = len(route_arcs)
                unique_route_arcs = len(set(route_arcs))

            inventory.append({
                "split": split,
                "trip_id": trip_id,
                "gps_points": gps_points,
                "nodes_count": nodes_count,
                "arcs_count": arcs_count,
                "route_arcs_count": route_count,
                "route_unique_arcs": unique_route_arcs,
                "total_size_kb": round(total_kb, 2),
                "duration_seconds": round(duration_s, 2),
                "sampling_1s_pct": round(pct_1s, 2),
                "gaps_gt_5s": n_gaps_gt5,
                "max_gap_s": round(max_gap_s, 2),
                "min_lon": min_lon,
                "max_lon": max_lon,
                "min_lat": min_lat,
                "max_lat": max_lat,
                "is_complete": all([track_file.exists(), nodes_file.exists(), arcs_file.exists(), route_file.exists()])
            })

    df = pd.DataFrame(inventory)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved dataset inventory ({len(df)} trips) to {output_csv}")
        
    return df

if __name__ == "__main__":
    df = build_inventory()
    print(f"Summary: {len(df)} trips loaded successfully.")
