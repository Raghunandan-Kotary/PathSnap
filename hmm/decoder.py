"""
Log-space Viterbi route decoder for PathSnap (Person 3).

Turns independent per-candidate classifier probabilities into one
connected, chronological sequence of predicted arc_ids per trip.

Can run two ways:
  - Stand-in mode (probabilities=None): emission comes only from
    `feat_dist_to_arc`, so the decoder can be built and tested before any
    classifier exists (see docs/.../Person 3 system work, step 1).
  - Model mode: emission is the classifier's calibrated probability for
    each (observation, candidate) pair, per Interface 2.

Viterbi recurrence (log space, per contiguous segment of observations):
    V[0][s]   = log_emission(obs_0, s)
    V[t][s]   = log_emission(obs_t, s) + max_{s'} ( V[t-1][s'] + log_transition(s', s) )
    backptr[t][s] = argmax_{s'} ( V[t-1][s'] + log_transition(s', s) )
Best path recovered by backtracking argmax(V[last]) through backptr.

A "segment" is a maximal run of consecutive observations with no large
time gap and no missing-candidate observation between them (see
`_split_into_segments`); each segment is decoded independently and the
results are concatenated, with -1 standing in for any observation that
had no candidates at all.
"""
from pathlib import Path
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from data.preprocessor import haversine_distance
from hmm.road_graph import build_arc_edge_lookup, build_road_graph

NO_CANDIDATE_ARC = -1
_LOG_EPS = 1e-9


class ViterbiRouteDecoder:
    def __init__(
        self,
        emission_weight: float = 1.0,
        distance_scale_m: float = 30.0,
        transition_scale_m: float = 50.0,
        disconnect_penalty: float = 25.0,
        gap_threshold_s: float = 10.0,
    ):
        """
        emission_weight: blend between classifier probability and the
            distance-based stand-in score, in [0, 1]. 1.0 = classifier only
            (falls back to distance-only automatically when probabilities
            is None).
        distance_scale_m: decay scale for the distance-based emission term
            exp(-feat_dist_to_arc / distance_scale_m).
        transition_scale_m: decay scale for the transition term
            exp(-|network_path_dist - straight_line_dist| / transition_scale_m).
        disconnect_penalty: fixed negative log-probability applied when two
            consecutive candidates have no directed path between them at all
            (different strongly-connected regions) -- NOT used for merely
            long detours, which are instead handled naturally by the distance
            term above (a long network path vs. straight line already scores
            low).
        gap_threshold_s: GPS time gaps larger than this break the sequence
            into independent Viterbi segments (per EDA finding that ~97% of
            steps are 1s; larger gaps indicate dropouts, not a continuous
            drive).
        """
        self.emission_weight = emission_weight
        self.distance_scale_m = distance_scale_m
        self.transition_scale_m = transition_scale_m
        self.disconnect_penalty = disconnect_penalty
        self.gap_threshold_s = gap_threshold_s

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "ViterbiRouteDecoder":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(
            emission_weight=cfg.get("emission_weight", 1.0),
            distance_scale_m=cfg.get("distance_scale_m", 30.0),
            transition_scale_m=cfg.get("transition_scale_m", 50.0),
            disconnect_penalty=cfg.get("disconnect_penalty", 25.0),
            gap_threshold_s=cfg.get("gap_threshold_s", 10.0),
        )

    def decode(
        self,
        candidates_df: pd.DataFrame,
        probabilities: Optional[np.ndarray] = None,
        road_graph: Optional[Union[nx.MultiDiGraph, pd.DataFrame]] = None,
        nodes_df: Optional[pd.DataFrame] = None,
    ) -> List[int]:
        """
        candidates_df: one row per (obs_idx, candidate arc_id), columns at
            least ["obs_idx", "gps_lon", "gps_lat", "gps_time", "arc_id",
            "from_node", "to_node", "feat_dist_to_arc"], per Interface 1.
        probabilities: 1D array aligned row-for-row with candidates_df
            (Interface 2), or None for stand-in distance-only decoding.
        road_graph: a prebuilt NetworkX DiGraph (Interface 3.3), OR the raw
            `arcs_df` DataFrame -- accepted for compatibility with
            api/pipeline.py, which calls decode(candidates_df, probs,
            arcs_df, nodes_df). If a DataFrame is passed here, `nodes_df`
            must also be given so the graph can be built.

        Returns one predicted arc_id per GPS observation, in chronological
        obs_idx order (including every obs_idx in [min, max] seen in
        candidates_df, even ones with zero candidates), using
        NO_CANDIDATE_ARC (-1) where no candidate exists for that observation.
        """
        if isinstance(road_graph, pd.DataFrame):
            if nodes_df is None:
                raise ValueError("decode() got arcs_df as road_graph but no nodes_df to build the graph from")
            road_graph = build_road_graph(nodes_df, road_graph)
        if road_graph is None:
            raise ValueError("decode() requires a road_graph (or arcs_df + nodes_df)")

        if candidates_df.empty:
            return []

        df = candidates_df.copy()
        df["_prob"] = probabilities if probabilities is not None else np.nan
        arc_edge_lookup = build_arc_edge_lookup(road_graph)

        obs_min, obs_max = int(df["obs_idx"].min()), int(df["obs_idx"].max())
        by_obs = {obs_idx: g for obs_idx, g in df.groupby("obs_idx")}

        predictions = [NO_CANDIDATE_ARC] * (obs_max - obs_min + 1)
        for segment in self._split_into_segments(by_obs, obs_min, obs_max):
            path = self._viterbi_segment(segment, road_graph, arc_edge_lookup)
            for obs_idx, arc_id in path:
                predictions[obs_idx - obs_min] = arc_id

        return predictions

    def _split_into_segments(self, by_obs: dict, obs_min: int, obs_max: int) -> List[List[pd.DataFrame]]:
        """Groups consecutive, gap-free, candidate-bearing observations into segments."""
        segments: List[List[pd.DataFrame]] = []
        current: List[pd.DataFrame] = []
        prev_time = None

        for obs_idx in range(obs_min, obs_max + 1):
            group = by_obs.get(obs_idx)
            if group is None or group.empty:
                if current:
                    segments.append(current)
                    current = []
                prev_time = None
                continue

            gps_time = float(group["gps_time"].iloc[0])
            if prev_time is not None and (gps_time - prev_time) > self.gap_threshold_s:
                if current:
                    segments.append(current)
                current = []

            current.append(group)
            prev_time = gps_time

        if current:
            segments.append(current)
        return [s for s in segments if s]

    def _emission_log_prob(self, group: pd.DataFrame) -> np.ndarray:
        dist_score = np.exp(-group["feat_dist_to_arc"].values / self.distance_scale_m)
        dist_score = dist_score / max(dist_score.sum(), _LOG_EPS)

        if group["_prob"].notna().all():
            clf_score = np.clip(group["_prob"].values, _LOG_EPS, 1.0)
            clf_score = clf_score / max(clf_score.sum(), _LOG_EPS)
            blended = self.emission_weight * clf_score + (1 - self.emission_weight) * dist_score
        else:
            blended = dist_score

        return np.log(np.clip(blended, _LOG_EPS, None))

    def _network_dist_from_source(self, source_node: int, road_graph: nx.MultiDiGraph) -> dict:
        """Single-source shortest path lengths (meters) from source_node to every reachable node."""
        try:
            return nx.single_source_dijkstra_path_length(road_graph, source_node, weight="length_m")
        except nx.NodeNotFound:
            return {}

    def _transition_log_prob_from_lengths(
        self,
        prev_arc: int,
        curr_row: pd.Series,
        lengths_from_prev_end: dict,
        arc_edge_lookup: dict,
        prev_gps: tuple,
    ) -> float:
        curr_arc = int(curr_row["arc_id"])
        if prev_arc == curr_arc:
            return 0.0  # staying on the same arc between consecutive observations

        curr_start, _ = arc_edge_lookup[curr_arc]
        network_dist = lengths_from_prev_end.get(curr_start, float("inf"))
        if np.isinf(network_dist):
            return -self.disconnect_penalty

        straight_dist = haversine_distance(prev_gps[0], prev_gps[1], curr_row["gps_lon"], curr_row["gps_lat"])
        score = np.exp(-abs(network_dist - straight_dist) / self.transition_scale_m)
        return float(np.log(max(score, _LOG_EPS)))

    def _viterbi_segment(
        self,
        segment: List[pd.DataFrame],
        road_graph: nx.MultiDiGraph,
        arc_edge_lookup: dict,
    ) -> List[tuple]:
        emissions = [self._emission_log_prob(g) for g in segment]

        V = [emissions[0]]
        backptrs: List[np.ndarray] = []

        for t in range(1, len(segment)):
            prev_group, curr_group = segment[t - 1], segment[t]
            n_prev, n_curr = len(prev_group), len(curr_group)

            # One single-source Dijkstra per previous candidate's end node,
            # reused across every current candidate (instead of one Dijkstra
            # per (prev, curr) pair).
            prev_lengths = []
            for i in range(n_prev):
                prev_arc = int(prev_group["arc_id"].iloc[i])
                _, prev_end = arc_edge_lookup[prev_arc]
                prev_lengths.append(self._network_dist_from_source(prev_end, road_graph))

            scores = np.full(n_curr, -np.inf)
            backptr = np.zeros(n_curr, dtype=int)

            for j in range(n_curr):
                curr_row = curr_group.iloc[j]
                best_score, best_i = -np.inf, 0
                for i in range(n_prev):
                    prev_arc = int(prev_group["arc_id"].iloc[i])
                    prev_gps = (prev_group["gps_lon"].iloc[i], prev_group["gps_lat"].iloc[i])
                    trans = self._transition_log_prob_from_lengths(
                        prev_arc, curr_row, prev_lengths[i], arc_edge_lookup, prev_gps
                    )
                    candidate_score = V[t - 1][i] + trans
                    if candidate_score > best_score:
                        best_score, best_i = candidate_score, i
                scores[j] = best_score + emissions[t][j]
                backptr[j] = best_i

            V.append(scores)
            backptrs.append(backptr)

        # Backtrack from the best final state.
        path_indices = [int(np.argmax(V[-1]))]
        for t in range(len(segment) - 1, 0, -1):
            path_indices.append(int(backptrs[t - 1][path_indices[-1]]))
        path_indices.reverse()

        return [
            (int(segment[t]["obs_idx"].iloc[0]), int(segment[t]["arc_id"].iloc[idx]))
            for t, idx in enumerate(path_indices)
        ]
