from datetime import datetime
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.schemas import HealthResponse, TripListResponse, PredictionResponse, ErrorResponse
from api.pipeline import pipeline_instance
from api.logger import api_logger, log_evaluation_request

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="PathSnap Distributed Map-Matching & Route Reconstruction Backend (Laptop 1)"
)

# Enable CORS for Laptop 2 access across LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    api_logger.info("PathSnap FastAPI Service started successfully on Laptop 1.")
    api_logger.info(f"Loaded configuration: Host={settings.HOST}, Port={settings.PORT}")

@app.get("/", tags=["General"])
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": "/health",
        "trips_url": "/trips"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """System health verification endpoint."""
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

@app.get("/trips", response_model=TripListResponse, tags=["Evaluation"])
def list_evaluation_trips():
    """Returns the list of evaluation trip IDs available on the server."""
    trips = []
    
    # Check student_test directory first
    if settings.STUDENT_TEST_DIR.exists():
        trips = [
            d.name for d in settings.STUDENT_TEST_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
    
    # If not populated yet, fall back to default student test trips
    if not trips:
        trips = settings.DEFAULT_TEST_TRIPS
        
    trips = sorted(list(set(trips)))
    return TripListResponse(trips=trips, count=len(trips))

@app.post(
    "/predict/{trip_id}",
    response_model=PredictionResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Evaluation"]
)
def predict_trip(trip_id: str, request: Request):
    """
    Executes map matching on the selected evaluation trip:
    - Loads raw trajectory (.track, .nodes, .arcs)
    - Generates candidate features and model probabilities
    - Runs log-space Viterbi sequential decoder
    - Computes server-side accuracy against ground truth (never exposed)
    - Logs request for evaluator review
    """
    client_ip = request.client.host if request.client else "unknown"
    t_start = time.perf_counter()
    
    try:
        response = pipeline_instance.run_pipeline(trip_id)
        latency = (time.perf_counter() - t_start) * 1000
        
        # Log request to CMD and file for evaluator audit
        log_evaluation_request(
            trip_id=trip_id,
            client_ip=client_ip,
            latency_ms=latency,
            accuracy=response.metrics.pointwise_top1_accuracy,
            continuity=response.metrics.route_continuity
        )
        return response
    except FileNotFoundError as fnf:
        api_logger.warning(f"Trip not found: {trip_id} from {client_ip}")
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        api_logger.error(f"Inference pipeline error for {trip_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=True)
