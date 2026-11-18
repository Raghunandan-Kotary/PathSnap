"""
End-to-end smoke test for Person 3's track: candidate generation (Person 1)
-> stand-in Viterbi decoding -> route metrics -> Folium map, run on a real
small trip before any classifier exists.
"""
from pathlib import Path

import pytest

from data.loader import load_arcs, load_nodes, load_route, load_track
from evaluation.route_metrics import candidate_level_ground_truth, compute_route_metrics
from features.candidate_generator import generate_candidates
from hmm.decoder import ViterbiRouteDecoder
from hmm.road_graph import build_road_graph
from maps.folium_generator import generate_trip_map

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIP_ID = "00000026"  # smallest train trip by GPS point count
TRIP_DIR = REPO_ROOT / "data" / "raw" / "train" / TRIP_ID


@pytest.fixture(scope="module")
def trip_data():
    if not TRIP_DIR.exists():
        pytest.skip("raw trip data not present in this checkout")
    return {
        "track": load_track(TRIP_DIR / f"{TRIP_ID}.track"),
        "nodes": load_nodes(TRIP_DIR / f"{TRIP_ID}.nodes"),
        "arcs": load_arcs(TRIP_DIR / f"{TRIP_ID}.arcs"),
        "route": load_route(TRIP_DIR / f"{TRIP_ID}.route"),
    }


def test_standin_decode_end_to_end(trip_data, tmp_path):
    track_df, nodes_df, arcs_df, route = (
        trip_data["track"], trip_data["nodes"], trip_data["arcs"], trip_data["route"]
    )

    cand_result = generate_candidates(track_df, nodes_df, arcs_df, route_arcs=route, radius_m=60.0)
    candidates_df = cand_result["candidates_df"]
    candidates_df["feat_dist_to_arc"] = 20.0  # stand-in: real features come from Person 1's pipeline

    assert cand_result["candidate_miss_rate"] < 1.0, "every observation missed its ground-truth arc"

    road_graph = build_road_graph(nodes_df, arcs_df)

    decoder = ViterbiRouteDecoder.from_config(REPO_ROOT / "hmm" / "config.yaml")
    predicted_arcs = decoder.decode(candidates_df, probabilities=None, road_graph=road_graph)

    n_obs = len(track_df)
    assert len(predicted_arcs) == n_obs

    ground_truth_arcs = candidate_level_ground_truth(candidates_df, obs_range=range(n_obs))
    metrics = compute_route_metrics(predicted_arcs, ground_truth_arcs, road_graph)
    for key in ("pointwise_accuracy", "route_continuity", "sequence_score"):
        assert 0.0 <= metrics[key] <= 1.0

    html = generate_trip_map(
        track_df, nodes_df, arcs_df,
        predicted_route=[a for a in predicted_arcs if a >= 0],
        ground_truth_route=route,
        save_path=tmp_path / f"{TRIP_ID}.html",
    )
    assert "<html" in html.lower()
    assert (tmp_path / f"{TRIP_ID}.html").exists()


def test_decode_accepts_pipeline_calling_convention(trip_data):
    """api/pipeline.py calls decode(candidates_df, probs, arcs_df, nodes_df) -- not road_graph."""
    track_df, nodes_df, arcs_df, route = (
        trip_data["track"], trip_data["nodes"], trip_data["arcs"], trip_data["route"]
    )
    cand_result = generate_candidates(track_df, nodes_df, arcs_df, route_arcs=route, radius_m=60.0)
    candidates_df = cand_result["candidates_df"]
    candidates_df["feat_dist_to_arc"] = 20.0

    decoder = ViterbiRouteDecoder.from_config(REPO_ROOT / "hmm" / "config.yaml")
    predicted_arcs = decoder.decode(candidates_df, None, arcs_df, nodes_df)
    assert len(predicted_arcs) == len(track_df)
