from typing import Optional, List
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: str

class TripListResponse(BaseModel):
    trips: List[str] = Field(..., description="List of evaluation trip IDs available on server")
    count: int

class PointPrediction(BaseModel):
    obs_idx: int
    predicted_arc_id: int
    confidence: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

class RouteMetricsResponse(BaseModel):
    pointwise_top1_accuracy: float = Field(..., description="Top-1 accuracy against server-side ground truth")
    route_continuity: float = Field(..., description="Percentage of consecutive predicted arcs with topological connectivity")
    total_observations: int
    matched_observations: int

class TimingBreakdown(BaseModel):
    data_load_ms: float
    feature_extraction_ms: float
    model_inference_ms: float
    viterbi_decode_ms: float
    total_latency_ms: float

class PredictionResponse(BaseModel):
    trip_id: str
    status: str = "success"
    num_observations: int
    predicted_arcs: List[int]
    metrics: RouteMetricsResponse
    timing: TimingBreakdown
    map_html: Optional[str] = Field(None, description="Embedded interactive Folium HTML representation")

class ErrorResponse(BaseModel):
    detail: str
