"""
Road-network graph construction for PathSnap (Person 3).

Builds a NetworkX directed graph from a trip's .nodes/.arcs tables, with
edge lengths computed in projected meters (never degree-based Euclidean
distance). Used by the connectivity EDA and by the Viterbi decoder's
transition model.
"""
from typing import Optional
import networkx as nx
import numpy as np
import pandas as pd

from data.preprocessor import project_coords


def build_road_graph(nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Builds a directed graph: node_id -> node_id for each arc, keyed by arc_id.

    A MultiDiGraph is required (not DiGraph): several distinct arcs can
    share the same (from_node, to_node) pair (parallel road segments), and
    a plain DiGraph would silently drop all but one arc_id per node pair.

    Each node carries projected (x, y) coordinates in meters (local
    equirectangular projection centered on the node set's median lon/lat).
    Each edge carries: arc_id, length_m (projected straight-line length).
    """
    ref_lon = nodes_df["lon"].median()
    ref_lat = nodes_df["lat"].median()
    xs, ys = project_coords(nodes_df["lon"].values, nodes_df["lat"].values, ref_lon, ref_lat)

    graph = nx.MultiDiGraph(ref_lon=float(ref_lon), ref_lat=float(ref_lat))
    for node_id, x, y in zip(nodes_df["node_id"], xs, ys):
        graph.add_node(int(node_id), x=float(x), y=float(y))

    for u, v, arc_id in zip(arcs_df["from_node"], arcs_df["to_node"], arcs_df["arc_id"]):
        u, v, arc_id = int(u), int(v), int(arc_id)
        ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
        vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
        length_m = float(np.hypot(vx - ux, vy - uy))
        graph.add_edge(u, v, key=arc_id, arc_id=arc_id, length_m=length_m)

    return graph


def arc_endpoints(graph: nx.MultiDiGraph, arc_id: int) -> Optional[tuple]:
    """Returns (from_node, to_node) for a given arc_id, or None if not found."""
    for u, v, data in graph.edges(data=True):
        if data["arc_id"] == arc_id:
            return (u, v)
    return None


def shortest_path_length_between_arcs(
    graph: nx.MultiDiGraph, arc_from: int, arc_to: int, arc_to_edge: dict
) -> float:
    """
    Shortest network path distance (meters) from the end node of arc_from
    to the start node of arc_to. Returns inf if unreachable.

    arc_to_edge maps arc_id -> (u, v) for O(1) lookups; build it once per
    trip rather than scanning edges per call.
    """
    if arc_from == arc_to:
        return 0.0
    _, u_end = arc_to_edge[arc_from]
    v_start, _ = arc_to_edge[arc_to]
    try:
        return float(nx.shortest_path_length(graph, source=u_end, target=v_start, weight="length_m"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def build_arc_edge_lookup(graph: nx.MultiDiGraph) -> dict:
    """arc_id -> (from_node, to_node), built once per trip graph."""
    return {data["arc_id"]: (u, v) for u, v, data in graph.edges(data=True)}


def connectivity_summary(graph: nx.MultiDiGraph) -> dict:
    """
    Connectivity diagnostics for the EDA checklist: graph size, isolated
    nodes, and weakly/strongly connected component counts and sizes.
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    isolated = [n for n in graph.nodes if graph.degree(n) == 0]

    weak_components = list(nx.weakly_connected_components(graph))
    strong_components = list(nx.strongly_connected_components(graph))
    weak_sizes = sorted((len(c) for c in weak_components), reverse=True)
    strong_sizes = sorted((len(c) for c in strong_components), reverse=True)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_isolated_nodes": len(isolated),
        "n_weakly_connected_components": len(weak_components),
        "largest_weak_component_size": weak_sizes[0] if weak_sizes else 0,
        "largest_weak_component_frac": (weak_sizes[0] / n_nodes) if n_nodes and weak_sizes else 0.0,
        "n_strongly_connected_components": len(strong_components),
        "largest_strong_component_size": strong_sizes[0] if strong_sizes else 0,
        "largest_strong_component_frac": (strong_sizes[0] / n_nodes) if n_nodes and strong_sizes else 0.0,
    }
