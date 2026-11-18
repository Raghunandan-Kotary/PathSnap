"""
Log-space Viterbi route decoder for PathSnap (Person 3).

Turns independent per-candidate classifier probabilities into one connected,
chronological sequence of predicted arc_ids per trip.

Hidden states: at GPS observation t, the states are that observation's
candidate arcs (rows of the candidate table), so the candidate-to-arc mapping
is preserved exactly; the decoder returns one arc_id per observation.

Emission (log space), for candidate arc a at observation t:
    log e_t(a) = w * log p_t(a) + (1 - w) * ( -0.5 * (d_t(a) / sigma)^2 )
  p_t(a) = classifier probability (calibrated), d_t(a) = perpendicular distance
  from the GPS point to the arc in metres, w = emission_weight,
  sigma = distance_scale_m. With no classifier (probabilities=None) only the
  Gaussian distance term is used ("stand-in" mode).

Transition (log space), from arc a at t-1 to arc b at t (Newson & Krumm 2009):
    route(a, b) = max(o_b - o_a, 0)            if a == b and o_b >= o_a - backtrack_tolerance_m
                = (L_a - o_a) + sp(end(a), start(b)) + o_b       otherwise
  (arcs are directed: on a two-way street the opposite-direction twin arc
  would need the car to move "backwards", which only fits by going round)
    log T(a, b) = -| route(a, b) - gc(t-1, t) | / beta
  o = distance along the arc of the GPS point's projection, L_a = arc length,
  sp = directed shortest network path, gc = straight-line distance between the
  two GPS points, beta = transition_scale_m. A car that really drove from a to
  b covers about as much road as the GPS points moved apart, so |route - gc| is
  small; jumps to parallel or unconnected roads make it large.
  If b is not reachable from a within dijkstra_cutoff_m metres of road (one-way
  conflict, disconnected component, or an implausible detour), the transition
  gets a fixed log-probability of -disconnect_penalty instead of -inf, so the
  decoder can still recover from a bad candidate set.

Viterbi recurrence (per segment):
    V_0(s)  = log e_0(s)
    V_t(s)  = log e_t(s) + max_{s'} [ V_{t-1}(s') + log T(s', s) ]
    ptr_t(s) = argmax_{s'} [ V_{t-1}(s') + log T(s', s) ]
  then backtrack from argmax_s V_last(s).

Segments: the sequence is split, and each part decoded independently, where
two consecutive observations are more than gap_threshold_s seconds apart
(GPS dropout: the car could be anywhere in between) or where an observation
has no candidates at all (that observation is returned as -1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from data.preprocessor import project_coords

NO_CANDIDATE_ARC = -1
_P_FLOOR = 1e-6
_DIJKSTRA_CELLS = 10_000_000  # sources x nodes per scipy dijkstra call (~80 MB)


@dataclass
class RoadIndex:
    """Array view of one trip's road network, in metres (local projection)."""

    arc_from: np.ndarray
    arc_to: np.ndarray
    arc_len: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    ref_lon: float
    ref_lat: float
    adjacency: csr_matrix

    @classmethod
    def from_frames(cls, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> "RoadIndex":
        ref_lon, ref_lat = float(nodes_df["lon"].median()), float(nodes_df["lat"].median())
        node_x, node_y = project_coords(nodes_df["lon"].to_numpy(), nodes_df["lat"].to_numpy(), ref_lon, ref_lat)
        arc_from = arcs_df["from_node"].to_numpy(dtype=np.int64)
        arc_to = arcs_df["to_node"].to_numpy(dtype=np.int64)
        return cls._build(arc_from, arc_to, np.asarray(node_x), np.asarray(node_y), ref_lon, ref_lat)

    @classmethod
    def from_graph(cls, graph: nx.MultiDiGraph) -> "RoadIndex":
        if "ref_lon" not in graph.graph:
            raise ValueError("road_graph has no ref_lon/ref_lat; build it with hmm.road_graph.build_road_graph")
        n_nodes = max(graph.nodes) + 1
        node_x, node_y = np.zeros(n_nodes), np.zeros(n_nodes)
        for n, d in graph.nodes(data=True):
            node_x[n], node_y[n] = d["x"], d["y"]
        edges = [(d["arc_id"], u, v) for u, v, d in graph.edges(data=True)]
        n_arcs = max(e[0] for e in edges) + 1
        arc_from, arc_to = np.zeros(n_arcs, dtype=np.int64), np.zeros(n_arcs, dtype=np.int64)
        for arc_id, u, v in edges:
            arc_from[arc_id], arc_to[arc_id] = u, v
        return cls._build(arc_from, arc_to, node_x, node_y, graph.graph["ref_lon"], graph.graph["ref_lat"])

    @classmethod
    def _build(cls, arc_from, arc_to, node_x, node_y, ref_lon, ref_lat) -> "RoadIndex":
        arc_len = np.hypot(node_x[arc_to] - node_x[arc_from], node_y[arc_to] - node_y[arc_from])
        keep = arc_from != arc_to  # self-loops never help a shortest path
        # Parallel arcs share a (from, to) pair; keep the shortest. A zero weight
        # would read as "no edge" in a sparse matrix, hence the 1 mm floor.
        edges = (pd.DataFrame({"u": arc_from[keep], "v": arc_to[keep], "w": np.maximum(arc_len[keep], 1e-3)})
                 .groupby(["u", "v"], as_index=False)["w"].min())
        n = len(node_x)
        adjacency = csr_matrix((edges["w"].to_numpy(), (edges["u"].to_numpy(), edges["v"].to_numpy())), shape=(n, n))
        return cls(arc_from, arc_to, arc_len, node_x, node_y, ref_lon, ref_lat, adjacency)


@dataclass
class _Step:
    obs_idx: int
    rows: np.ndarray                 # positions of this observation's candidates in candidates_df
    arcs: np.ndarray
    perp_m: np.ndarray               # GPS point -> arc distance
    trans_cost: Optional[np.ndarray]  # (n_prev, n_this) |route - gc| in metres, inf = unreachable


@dataclass
class PreparedTrip:
    """Everything about a trip that does not depend on the tunable weights."""

    obs_min: int
    obs_max: int
    segments: List[List[_Step]]


class ViterbiRouteDecoder:
    def __init__(
        self,
        emission_weight: float = 1.0,
        distance_scale_m: float = 10.0,
        transition_scale_m: float = 10.0,
        disconnect_penalty: float = 25.0,
        gap_threshold_s: float = 10.0,
        dijkstra_cutoff_m: float = 1000.0,
        backtrack_tolerance_m: float = 10.0,
        emission_blend: str = "mixture",
    ):
        """
        emission_weight: w in the emission formula; 1.0 = classifier only.
            Ignored (distance-only) when decode() gets probabilities=None.
        distance_scale_m: sigma of the Gaussian GPS-error term (metres).
        transition_scale_m: beta of the transition term (metres).
        disconnect_penalty: -log T used when the next arc is not reachable.
        gap_threshold_s: time gap that splits the sequence into segments.
        dijkstra_cutoff_m: longest road path considered between consecutive
            observations; beyond it the transition counts as unreachable.
        backtrack_tolerance_m: how far the projected position may move
            backwards along the same arc between two fixes (GPS jitter while
            slow or stopped) before it counts as driving round the block.
        """
        self.emission_weight = emission_weight
        self.distance_scale_m = distance_scale_m
        self.transition_scale_m = transition_scale_m
        self.disconnect_penalty = disconnect_penalty
        self.gap_threshold_s = gap_threshold_s
        self.dijkstra_cutoff_m = dijkstra_cutoff_m
        self.backtrack_tolerance_m = backtrack_tolerance_m
        if emission_blend not in ("mixture", "log_linear"):
            raise ValueError(f"emission_blend must be 'mixture' or 'log_linear', got {emission_blend!r}")
        self.emission_blend = emission_blend

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "ViterbiRouteDecoder":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        defaults = cls()
        return cls(**{k: cfg.get(k, getattr(defaults, k)) for k in (
            "emission_weight", "distance_scale_m", "transition_scale_m",
            "disconnect_penalty", "gap_threshold_s", "dijkstra_cutoff_m", "backtrack_tolerance_m",
            "emission_blend")})

    def decode(
        self,
        candidates_df: pd.DataFrame,
        probabilities: Optional[np.ndarray] = None,
        road_graph: Optional[Union[nx.MultiDiGraph, pd.DataFrame, RoadIndex]] = None,
        nodes_df: Optional[pd.DataFrame] = None,
    ) -> List[int]:
        """
        candidates_df: one row per (obs_idx, candidate arc_id), with at least
            obs_idx, gps_lon, gps_lat, gps_time, arc_id (Interface 1).
        probabilities: 1D array aligned row-for-row with candidates_df
            (Interface 2), or None for distance-only decoding.
        road_graph: hmm.road_graph.build_road_graph() output, a RoadIndex, or
            the raw arcs_df DataFrame together with nodes_df (the form
            api/pipeline.py uses).

        Returns one arc_id per observation from min to max obs_idx, -1 where
        an observation had no candidates.
        """
        if candidates_df.empty:
            return []
        road = self._road_index(road_graph, nodes_df)
        return self.decode_prepared(self.prepare(candidates_df, road), probabilities)

    @staticmethod
    def _road_index(road_graph, nodes_df) -> RoadIndex:
        if isinstance(road_graph, RoadIndex):
            return road_graph
        if isinstance(road_graph, pd.DataFrame):
            if nodes_df is None:
                raise ValueError("decode() got arcs_df as road_graph but no nodes_df to build the graph from")
            return RoadIndex.from_frames(nodes_df, road_graph)
        if isinstance(road_graph, nx.Graph):
            return RoadIndex.from_graph(road_graph)
        raise ValueError("decode() requires a road_graph (or arcs_df + nodes_df)")

    def prepare(self, candidates_df: pd.DataFrame, road: RoadIndex) -> PreparedTrip:
        """Geometry, segmentation and transition costs for one trip (the slow part, done once)."""
        obs = candidates_df["obs_idx"].to_numpy(dtype=np.int64)
        order = np.argsort(obs, kind="stable")
        obs_sorted = obs[order]
        uniq_obs, starts = np.unique(obs_sorted, return_index=True)
        ends = np.append(starts[1:], len(order))

        arcs = candidates_df["arc_id"].to_numpy(dtype=np.int64)
        gx, gy = project_coords(candidates_df["gps_lon"].to_numpy(), candidates_df["gps_lat"].to_numpy(),
                                road.ref_lon, road.ref_lat)
        gx, gy = np.asarray(gx), np.asarray(gy)
        gtime = candidates_df["gps_time"].to_numpy(dtype=float)

        x1, y1 = road.node_x[road.arc_from[arcs]], road.node_y[road.arc_from[arcs]]
        x2, y2 = road.node_x[road.arc_to[arcs]], road.node_y[road.arc_to[arcs]]
        dx, dy = x2 - x1, y2 - y1
        seg2 = np.maximum(dx * dx + dy * dy, 1e-12)
        frac = np.clip(((gx - x1) * dx + (gy - y1) * dy) / seg2, 0.0, 1.0)
        perp = np.hypot(gx - (x1 + frac * dx), gy - (y1 + frac * dy))
        offset = frac * road.arc_len[arcs]

        # Segment boundaries: missing observation index or a long time gap.
        first_row = order[starts]
        t_obs, x_obs, y_obs = gtime[first_row], gx[first_row], gy[first_row]
        breaks = np.ones(len(uniq_obs), dtype=bool)
        breaks[1:] = (np.diff(uniq_obs) != 1) | (np.diff(t_obs) > self.gap_threshold_s)

        # All shortest paths needed by the trip, computed in one batch.
        trans_pairs = [(k - 1, k) for k in range(1, len(uniq_obs)) if not breaks[k]]
        sources = np.unique(np.concatenate([road.arc_to[arcs[order[starts[p]:ends[p]]]] for p, _ in trans_pairs])) \
            if trans_pairs else np.array([], dtype=np.int64)
        targets = np.unique(np.concatenate([road.arc_from[arcs[order[starts[c]:ends[c]]]] for _, c in trans_pairs])) \
            if trans_pairs else np.array([], dtype=np.int64)
        sp = self._shortest_paths(road, sources, targets)
        src_pos = {n: i for i, n in enumerate(sources)}
        tgt_pos = {n: i for i, n in enumerate(targets)}

        segments: List[List[_Step]] = []
        prev: Optional[_Step] = None
        for k, o in enumerate(uniq_obs):
            rows = order[starts[k]:ends[k]]
            step = _Step(int(o), rows, arcs[rows], perp[rows], None)
            if breaks[k]:
                segments.append([step])
            else:
                a, b = prev.arcs, step.arcs
                s_idx = np.fromiter((src_pos[n] for n in road.arc_to[a]), dtype=np.int64, count=len(a))
                t_idx = np.fromiter((tgt_pos[n] for n in road.arc_from[b]), dtype=np.int64, count=len(b))
                route = (road.arc_len[a] - offset[prev.rows])[:, None] + sp[np.ix_(s_idx, t_idx)] + offset[rows][None, :]
                same = a[:, None] == b[None, :]
                if same.any():
                    # Arcs are directed: moving forward along the same arc costs the distance
                    # covered; moving back more than the GPS-jitter tolerance means going
                    # round to the arc's start again (the `route` value above, through
                    # sp(end(a), start(a))), which is what rules out the opposite-direction
                    # twin of a two-way street.
                    moved = offset[rows][None, :] - offset[prev.rows][:, None]
                    along = np.where(moved >= -self.backtrack_tolerance_m, np.maximum(moved, 0.0), route)
                    route = np.where(same, along, route)
                gc = np.hypot(x_obs[k] - x_obs[k - 1], y_obs[k] - y_obs[k - 1])
                step.trans_cost = np.abs(route - gc).astype(np.float32)
                segments[-1].append(step)
            prev = step
        return PreparedTrip(int(uniq_obs[0]), int(uniq_obs[-1]), segments)

    def _shortest_paths(self, road: RoadIndex, sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Directed road distance from each source node to each target node (inf beyond the cutoff)."""
        out = np.full((len(sources), len(targets)), np.inf, dtype=np.float32)
        if len(sources) == 0 or len(targets) == 0:
            return out
        chunk = max(1, _DIJKSTRA_CELLS // road.adjacency.shape[0])
        for i in range(0, len(sources), chunk):
            d = dijkstra(road.adjacency, directed=True, indices=sources[i:i + chunk], limit=self.dijkstra_cutoff_m)
            out[i:i + chunk] = d[:, targets]
        return out

    def decode_prepared(self, prepared: PreparedTrip, probabilities: Optional[np.ndarray] = None) -> List[int]:
        """Viterbi over a prepared trip with the current weights (fast; used for tuning)."""
        probs = None if probabilities is None else np.clip(np.asarray(probabilities, dtype=float), _P_FLOOR, 1.0)
        w = self.emission_weight if probs is not None else 0.0
        predictions = [NO_CANDIDATE_ARC] * (prepared.obs_max - prepared.obs_min + 1)
        for segment in prepared.segments:
            path = self._viterbi(segment, probs, w)
            for step, j in zip(segment, path):
                predictions[step.obs_idx - prepared.obs_min] = int(step.arcs[j])
        return predictions

    def _emission(self, step: _Step, probs: Optional[np.ndarray], w: float) -> np.ndarray:
        dist_term = -0.5 * (step.perp_m / self.distance_scale_m) ** 2
        if probs is None or w == 0.0:
            return dist_term
        p = probs[step.rows]
        if self.emission_blend == "mixture":
            # A confidently wrong classifier score cannot veto the true arc: the
            # GPS-distance likelihood keeps it alive.
            return np.log(w * p + (1.0 - w) * np.exp(dist_term))
        return w * np.log(p) + (1.0 - w) * dist_term

    def _viterbi(self, segment: List[_Step], probs: Optional[np.ndarray], w: float) -> List[int]:
        v = self._emission(segment[0], probs, w)
        backptrs = []
        for step in segment[1:]:
            cost = step.trans_cost
            log_t = np.where(np.isfinite(cost), -cost / self.transition_scale_m, -self.disconnect_penalty)
            scores = v[:, None] + log_t
            ptr = np.argmax(scores, axis=0)
            v = scores[ptr, np.arange(scores.shape[1])] + self._emission(step, probs, w)
            backptrs.append(ptr)
        path = [int(np.argmax(v))]
        for ptr in reversed(backptrs):
            path.append(int(ptr[path[-1]]))
        path.reverse()
        return path
