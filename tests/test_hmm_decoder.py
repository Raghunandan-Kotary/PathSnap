"""
End-to-end smoke test for Person 3's track: candidate generation (Person 1)
-> Viterbi decoding -> route metrics -> Folium map, on a real small trip.
"""
from pathlib import Path

import pytest

from evaluation.route_metrics import compute_route_metrics, load_trip_reference
from features.candidate_generator import generate_candidates
from hmm.decoder import ViterbiRouteDecoder
from hmm.road_graph import build_road_graph
from maps.folium_generator import generate_trip_map

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIP_ID = "00000026"  # smallest train trip by GPS point count


@pytest.fixture(scope="module")
def trip():
    if not (REPO_ROOT / "data" / "raw" / "train" / TRIP_ID).exists():
        pytest.skip("raw trip data not present in this checkout")
    ref = load_trip_reference(TRIP_ID)
    ref["candidates"] = generate_candidates(ref["track"], ref["nodes"], ref["arcs"], radius_m=60.0)["candidates_df"]
    return ref


def test_decode_end_to_end(trip, tmp_path):
    decoder = ViterbiRouteDecoder.from_config(REPO_ROOT / "hmm" / "config.yaml")
    predicted = decoder.decode(trip["candidates"], None, build_road_graph(trip["nodes"], trip["arcs"]))
    assert len(predicted) == len(trip["track"])

    m = compute_route_metrics(predicted, trip["gt"], trip["arcs"])
    for key in ("pointwise_accuracy", "route_continuity", "arc_change_continuity", "sequence_score"):
        assert 0.0 <= m[key] <= 1.0
    assert m["pointwise_accuracy"] > 0.5  # distance-only decoding already beats chance by far

    html = generate_trip_map(trip["track"], trip["nodes"], trip["arcs"], predicted,
                             ground_truth_route=trip["route"], save_path=tmp_path / "map.html")
    assert "<html" in html.lower() and (tmp_path / "map.html").exists()


def test_decode_accepts_pipeline_calling_convention(trip):
    """api/pipeline.py calls decode(candidates_df, probs, arcs_df, nodes_df)."""
    decoder = ViterbiRouteDecoder.from_config(REPO_ROOT / "hmm" / "config.yaml")
    probs = [0.5] * len(trip["candidates"])
    assert len(decoder.decode(trip["candidates"], probs, trip["arcs"], trip["nodes"])) == len(trip["track"])
