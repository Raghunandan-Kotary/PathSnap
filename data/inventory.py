"""
Dataset inventory script for Person 1.
Scans data/raw/ directories (train, student_test) and compiles trip counts,
row counts, file sizes, and missingness metrics into a summary report.
"""
from pathlib import Path
import pandas as pd

def build_inventory(raw_dir: Path = Path("data/raw")):
    inventory = []
    for split in ["train", "student_test"]:
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue
        for trip_dir in split_dir.iterdir():
            if not trip_dir.is_dir() or trip_dir.name.startswith("."):
                continue
            trip_id = trip_dir.name
            track_file = trip_dir / f"{trip_id}.track"
            nodes_file = trip_dir / f"{trip_id}.nodes"
            arcs_file = trip_dir / f"{trip_id}.arcs"
            route_file = trip_dir / f"{trip_id}.route"
            
            track_rows = sum(1 for _ in open(track_file)) if track_file.exists() else 0
            nodes_rows = sum(1 for _ in open(nodes_file)) if nodes_file.exists() else 0
            arcs_rows = sum(1 for _ in open(arcs_file)) if arcs_file.exists() else 0
            route_rows = sum(1 for _ in open(route_file)) if route_file.exists() else 0
            
            inventory.append({
                "split": split,
                "trip_id": trip_id,
                "gps_points": track_rows,
                "nodes": nodes_rows,
                "arcs": arcs_rows,
                "reference_route_arcs": route_rows,
                "is_complete": all([track_file.exists(), nodes_file.exists(), arcs_file.exists(), route_file.exists()])
            })
    df = pd.DataFrame(inventory)
    print(f"Total trips inventoried: {len(df)}")
    return df

if __name__ == "__main__":
    df = build_inventory()
    print(df.head(10))
