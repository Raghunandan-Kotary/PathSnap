import os
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from api.config import settings
from api.schemas import PredictionResponse, RouteMetricsResponse, TimingBreakdown
from evaluation.route_metrics import compute_route_metrics

try:
    from maps.folium_generator import generate_folium_map
except ImportError:
    generate_folium_map = None

class PathSnapPipeline:
    def __init__(self):
        self.model = None
        self.decoder = None
        self._load_artifacts_if_available()

    def _load_artifacts_if_available(self):
        if settings.MODEL_PATH.exists():
            try:
                import joblib
                self.model = joblib.load(settings.MODEL_PATH)
            except Exception:
                self.model = None

    def find_trip_directory(self, trip_id: str) -> Path:
        """Locates the directory containing trip files across test partitions."""
        candidates = [
            settings.STUDENT_TEST_DIR / trip_id,
            settings.EVALUATOR_TEST_DIR / trip_id,
            settings.RAW_DATA_DIR / "train" / trip_id
        ]
        for p in candidates:
            if p.exists() and p.is_dir():
                return p
        raise FileNotFoundError(f"Trip '{trip_id}' not found in any data directory.")

    def run_pipeline(self, trip_id: str, include_map: bool = True) -> PredictionResponse:
        """
        Executes end-to-end map matching:
        1. Load raw trip files (.track, .nodes, .arcs, .route)
        2. Feature generation
        3. Model probability scoring
        4. Viterbi sequential decoding
        5. Server-side ground truth evaluation (NEVER exposed to UI)
        6. Folium interactive map generation
        """
        t0 = time.perf_counter()
        trip_dir = self.find_trip_directory(trip_id)
        
        # 1. Load raw data
        t_load_start = time.perf_counter()
        track_path = trip_dir / f"{trip_id}.track"
        nodes_path = trip_dir / f"{trip_id}.nodes"
        arcs_path = trip_dir / f"{trip_id}.arcs"
        route_path = trip_dir / f"{trip_id}.route"
        
        track_df = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "time"])
        nodes_df = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"])
        arcs_df = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["from_node", "to_node"])
        
        # Server-side Ground Truth (.route file: sequence of ground truth arc_ids)
        gt_arcs = []
        if route_path.exists():
            with open(route_path, "r", encoding="utf-8") as f:
                gt_arcs = [int(line.strip()) for line in f if line.strip().isdigit()]
        t_load = (time.perf_counter() - t_load_start) * 1000

        # 2. Candidate Generation & Feature Extraction
        t_feat_start = time.perf_counter()
        num_obs = len(track_df)
        predicted_arcs: List[int] = []
        
        try:
            from features.extract_features import generate_trip_candidates_and_features
            candidates_df = generate_trip_candidates_and_features(track_df, nodes_df, arcs_df)
        except Exception:
            candidates_df = None
        t_feat = (time.perf_counter() - t_feat_start) * 1000

        # 3. Model Scoring & 4. Decoding
        t_infer_start = time.perf_counter()
        t_infer = 0.0
        t_decode_start = time.perf_counter()
        
        try:
            if self.model is not None and candidates_df is not None:
                probs = self.model.predict_proba(candidates_df)
                t_infer = (time.perf_counter() - t_infer_start) * 1000
                t_decode_start = time.perf_counter()
                from hmm.decoder import ViterbiRouteDecoder
                decoder = ViterbiRouteDecoder.from_config(settings.HMM_CONFIG_PATH)
                predicted_arcs = decoder.decode(candidates_df, probs, arcs_df, nodes_df)
            else:
                raise RuntimeError("Model or candidates not yet trained")
        except Exception:
            # Baseline spatial matcher ensures API works end-to-end immediately
            t_infer = (time.perf_counter() - t_infer_start) * 1000
            t_decode_start = time.perf_counter()
            predicted_arcs = self._baseline_spatial_matcher(track_df, nodes_df, arcs_df)
            
        t_decode = (time.perf_counter() - t_decode_start) * 1000

        # 5. Server-Side Ground Truth Evaluation (NEVER leaks to UI)
        raw_metrics = compute_route_metrics(predicted_arcs, gt_arcs, arcs_df)
        metrics = RouteMetricsResponse(
            pointwise_top1_accuracy=raw_metrics["pointwise_top1_accuracy"],
            route_continuity=raw_metrics["route_continuity"],
            total_observations=raw_metrics["total_observations"],
            matched_observations=raw_metrics["matched_observations"]
        )

        # 6. Folium Map Generation
        map_html = None
        if include_map:
            try:
                folium_map = generate_folium_map(
                    track_df=track_df,
                    nodes_df=nodes_df,
                    arcs_df=arcs_df,
                    predicted_arcs=predicted_arcs,
                    ground_truth_arcs=gt_arcs,
                    trip_id=trip_id
                )
                map_html = folium_map._repr_html_()
            except Exception:
                map_html = None
        
        total_time = (time.perf_counter() - t0) * 1000
        
        timing = TimingBreakdown(
            data_load_ms=round(t_load, 2),
            feature_extraction_ms=round(t_feat, 2),
            model_inference_ms=round(t_infer, 2),
            viterbi_decode_ms=round(t_decode, 2),
            total_latency_ms=round(total_time, 2)
        )

        return PredictionResponse(
            trip_id=trip_id,
            status="success",
            num_observations=num_obs,
            predicted_arcs=predicted_arcs,
            metrics=metrics,
            timing=timing,
            map_html=map_html
        )

    def _baseline_spatial_matcher(self, track_df: pd.DataFrame, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> List[int]:
        """Spatial matcher using midpoint distance approximation."""
        from_nodes = nodes_df.iloc[arcs_df["from_node"]].reset_index(drop=True)
        to_nodes = nodes_df.iloc[arcs_df["to_node"]].reset_index(drop=True)
        arc_mid_lon = (from_nodes["lon"].values + to_nodes["lon"].values) / 2.0
        arc_mid_lat = (from_nodes["lat"].values + to_nodes["lat"].values) / 2.0
        
        predictions = []
        for _, row in track_df.iterrows():
            d2 = (arc_mid_lon - row["lon"]) ** 2 + (arc_mid_lat - row["lat"]) ** 2
            best_arc = int(np.argmin(d2))
            predictions.append(best_arc)
        return predictions

pipeline_instance = PathSnapPipeline()
