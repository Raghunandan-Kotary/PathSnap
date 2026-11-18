import os
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from api.config import settings, BASE_DIR
from api.schemas import PredictionResponse, RouteMetricsResponse, TimingBreakdown
from api.logger import api_logger
from data.loader import load_track, load_nodes, load_arcs, load_route
from evaluation.route_metrics import compute_route_metrics

try:
    from maps.folium_generator import generate_folium_map
except ImportError:
    generate_folium_map = None

class PathSnapPipeline:
    def __init__(self):
        self.model = None
        self._load_artifacts_if_available()

    def _load_artifacts_if_available(self):
        # Look for model in models/saved/best_model.joblib or person2/models/final/model.joblib
        search_paths = [
            settings.MODEL_PATH,
            BASE_DIR / "person2" / "models" / "final" / "model.joblib"
        ]
        for p in search_paths:
            if p.exists():
                try:
                    from models.loader import ModelWrapper
                    self.model = ModelWrapper.load(p)
                    api_logger.info(f"Loaded classifier model artifact from {p}")
                    break
                except Exception as e:
                    api_logger.warning(f"Failed to load model from {p}: {e}")
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

    def run_pipeline(
        self,
        trip_id: str,
        include_map: bool = True,
        strict_mode: bool = False
    ) -> PredictionResponse:
        """
        Executes end-to-end map matching:
        1. Load raw trip files (.track, .nodes, .arcs, .route)
        2. Feature generation
        3. Model probability scoring (if model available)
        4. Viterbi sequential decoding
        5. Server-side ground truth evaluation (NEVER exposed to UI)
        6. Folium interactive map generation
        """
        t0 = time.perf_counter()
        trip_dir = self.find_trip_directory(trip_id)
        
        # 1. Load raw data using standard loader
        t_load_start = time.perf_counter()
        track_df = load_track(trip_dir / f"{trip_id}.track")
        nodes_df = load_nodes(trip_dir / f"{trip_id}.nodes")
        arcs_df = load_arcs(trip_dir / f"{trip_id}.arcs")
        gt_arcs = load_route(trip_dir / f"{trip_id}.route")
        t_load = (time.perf_counter() - t_load_start) * 1000

        # 2. Candidate Generation & Feature Extraction
        t_feat_start = time.perf_counter()
        num_obs = len(track_df)
        predicted_arcs: List[int] = []
        
        from features.extract_features import generate_trip_candidates_and_features
        candidates_df = generate_trip_candidates_and_features(track_df, nodes_df, arcs_df)
        t_feat = (time.perf_counter() - t_feat_start) * 1000

        # 3. Model Scoring & 4. Decoding
        t_infer_start = time.perf_counter()
        t_infer = 0.0
        t_decode = 0.0
        execution_mode = "unknown"

        probs = None
        if self.model is not None and not candidates_df.empty:
            try:
                probs = self.model.predict_proba(candidates_df)
                t_infer = (time.perf_counter() - t_infer_start) * 1000
                execution_mode = "model_plus_viterbi"
            except Exception as e:
                api_logger.warning(f"Model prediction failed: {e}")
                if strict_mode:
                    raise RuntimeError(f"Strict mode enabled: Model inference failed with {e}")

        # Attempt Viterbi decoding
        t_decode_start = time.perf_counter()
        try:
            from hmm.decoder import ViterbiRouteDecoder
            decoder = ViterbiRouteDecoder.from_config(settings.HMM_CONFIG_PATH) if settings.HMM_CONFIG_PATH.exists() else ViterbiRouteDecoder()
            predicted_arcs = decoder.decode(
                candidates_df=candidates_df,
                probabilities=probs,
                road_graph=arcs_df,
                nodes_df=nodes_df
            )
            t_decode = (time.perf_counter() - t_decode_start) * 1000
            if execution_mode != "model_plus_viterbi":
                execution_mode = "viterbi_distance_emissions"
                api_logger.info(f"Trip {trip_id}: Executed real Viterbi decoder with distance-based emissions.")
        except Exception as viterbi_err:
            api_logger.error(f"Viterbi decoder error for trip {trip_id}: {viterbi_err}")
            if strict_mode:
                raise RuntimeError(f"Strict mode enabled: Viterbi decoding failed: {viterbi_err}")
            
            # Loudly log that fallback path is taken
            api_logger.warning(f"[LOUD WARNING] Trip {trip_id}: FALLBACK to baseline spatial matcher triggered due to: {viterbi_err}")
            execution_mode = "baseline_spatial_matcher"
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
        if include_map and generate_folium_map is not None:
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
            except Exception as map_err:
                api_logger.warning(f"Folium map generation failed: {map_err}")
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
            execution_mode=execution_mode,
            num_observations=num_obs,
            predicted_arcs=predicted_arcs,
            metrics=metrics,
            timing=timing,
            map_html=map_html
        )

    def _baseline_spatial_matcher(self, track_df: pd.DataFrame, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> List[int]:
        """Fallback spatial matcher using midpoint distance approximation."""
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
