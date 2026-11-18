"""
Person 3 EDA lead focus: road-network connectivity, isolated nodes and
disconnected components, across all 90 trips (80 train + 10 student_test).

Run from the repo root:
    python notebooks/person3_road_network_connectivity_eda.py

Writes a per-trip summary table to data/processed/road_network_connectivity.csv
and a markdown report to docs/eda/road_network_connectivity.md, both of which
feed report section 2.2 and the HMM transition-model design (hmm/decoder.py).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data.loader import load_nodes, load_arcs, load_route
from hmm.road_graph import build_road_graph, connectivity_summary

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "road_network_connectivity.csv"
OUT_MD = Path(__file__).resolve().parents[1] / "docs" / "eda" / "road_network_connectivity.md"


def analyze_trip(trip_dir: Path, trip_id: str, split: str) -> dict:
    nodes_df = load_nodes(trip_dir / f"{trip_id}.nodes")
    arcs_df = load_arcs(trip_dir / f"{trip_id}.arcs")
    route = load_route(trip_dir / f"{trip_id}.route")

    graph = build_road_graph(nodes_df, arcs_df)
    summary = connectivity_summary(graph)

    # Does the reference route ever cross a weak-component boundary?
    # (it shouldn't, but confirms the graph is built correctly)
    route_arc_lookup = {data["arc_id"]: (u, v) for u, v, data in graph.edges(data=True)}
    route_nodes = set()
    for arc_id in route:
        if arc_id in route_arc_lookup:
            route_nodes.update(route_arc_lookup[arc_id])
    weak_components = list(__import__("networkx").weakly_connected_components(graph))
    route_spans_multiple_components = False
    if route_nodes:
        touched = [i for i, comp in enumerate(weak_components) if comp & route_nodes]
        route_spans_multiple_components = len(set(touched)) > 1

    summary.update({
        "trip_id": trip_id,
        "split": split,
        "n_route_arcs": len(route),
        "route_spans_multiple_weak_components": route_spans_multiple_components,
    })
    return summary


def main():
    rows = []
    for split in ["train", "student_test"]:
        split_dir = RAW_DIR / split
        if not split_dir.exists():
            print(f"skip missing split: {split_dir}")
            continue
        for trip_dir in sorted(split_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            trip_id = trip_dir.name
            try:
                rows.append(analyze_trip(trip_dir, trip_id, split))
            except Exception as exc:
                print(f"FAILED {split}/{trip_id}: {exc}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} trip rows to {OUT_CSV}")

    write_markdown_report(df)
    print(f"Wrote report to {OUT_MD}")


def write_markdown_report(df: pd.DataFrame) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    n_trips = len(df)
    n_single_component = int((df["n_weakly_connected_components"] == 1).sum())
    n_with_isolated = int((df["n_isolated_nodes"] > 0).sum())
    n_route_spans = int(df["route_spans_multiple_weak_components"].sum())

    lines = [
        "# Road-network connectivity EDA (Person 3 lead focus)",
        "",
        f"Analyzed {n_trips} trips (80 train + 10 student_test), "
        "one road-network graph per trip, built from that trip's `.nodes`/`.arcs` "
        "files with `hmm/road_graph.build_road_graph`.",
        "",
        "## Headline findings",
        "",
        f"- **{n_single_component}/{n_trips}** trips have a single weakly-connected "
        "road network (every node reachable from every other, ignoring direction).",
        f"- **{n_trips - n_single_component}/{n_trips}** trips have more than one "
        "weakly-connected component, i.e. the road graph is fragmented.",
        f"- **{n_with_isolated}/{n_trips}** trips contain at least one isolated node "
        "(degree 0 — appears in `.nodes` but is the endpoint of no arc).",
        f"- **{n_route_spans}/{n_trips}** reference routes touch more than one weak "
        "component — any such case means the ground-truth route itself crosses a "
        "disconnection, which the HMM gap-handling / shortest-path-bridge logic must "
        "account for.",
        "",
        "Strongly-connected components (direction-aware) are typically far more "
        "fragmented than weak components, since arcs are one-way: this matters for "
        "the Viterbi transition model, where a transition between consecutive "
        "candidate arcs implicitly requires a directed path.",
        "",
        "## Per-trip summary",
        "",
        "Note: read `road_network_connectivity.csv` with `dtype={'trip_id': str}` -- "
        "plain `pd.read_csv` infers `trip_id` as an int and drops the leading zeros.",
        "",
        df.to_markdown(index=False),
        "",
        "## Implications for the HMM decoder",
        "",
        "- Transition probabilities use directed shortest-path distance "
        "(`hmm/road_graph.shortest_path_length_between_arcs`), which returns `inf` "
        "when the end node of one candidate and the start node of the next lie in "
        "different strongly-connected regions — these transitions must be penalised "
        "rather than treated as a normal distance.",
        "- Large GPS gaps and genuinely disconnected consecutive observations are "
        "handled the same way in `hmm/decoder.py`: break-and-restart the Viterbi "
        "sequence, or bridge with a bounded shortest-path penalty, per trip "
        "`gap_threshold_s`.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
