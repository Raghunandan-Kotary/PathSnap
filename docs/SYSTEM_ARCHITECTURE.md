# PathSnap: Distributed System Architecture & Deployment

## 1. Physical Architecture

The system operates across two separate machines communicating over a local TCP/IP network:

```
+------------------------------------+          +------------------------------------+
|             LAPTOP 1               |          |             LAPTOP 2               |
|      (FastAPI Server & Model)      |          |          (Evaluator UI)            |
|                                    |          |                                    |
|  IP: 192.168.x.x (or 10.x.x.x)     |          |  IP: 192.168.x.y                   |
|  Port: 8000                        |  HTTP    |  Port: 8501                        |
|                                    | <======> |                                    |
|  - Uvicorn / FastAPI Backend       |  REST    |  - Streamlit Evaluator Dashboard   |
|  - Feature Engineering Engine      |  JSON    |  - 10 Evaluation Trip Selector     |
|  - Model Inference Engine          |          |  - Interactive Folium Map Viewer   |
|  - Log-Space Viterbi HMM Decoder   |          |  - Performance Metrics Cards       |
|  - Server-Side Ground Truth (.route|          |  - Configurable Backend Address    |
|  - Request Audit Logger (File & CLI|          |                                    |
+------------------------------------+          +------------------------------------+
```

---

## 2. API Endpoints

| Method | Endpoint | Description | Request Body | Response Body |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health check | None | `{"status": "ok", "version": "1.0.0", "timestamp": "..."}` |
| `GET` | `/trips` | Returns list of available evaluation trip IDs | None | `{"trip_ids": ["00000000", ...], "count": 10}` |
| `POST`| `/predict/{trip_id}` | Runs full pipeline for specified trip | None | Prediction JSON: predicted arcs, timing, pointwise accuracy, route continuity |

---

## 3. Data Flow & Security Guardrails

1. **Isolation of Ground Truth**:
   The reference routes (`.route` files) are held strictly in Laptop 1's server memory or local storage. When `/predict/{trip_id}` is invoked:
   - Only `.track`, `.nodes`, and `.arcs` are fed into feature extraction and candidate scoring.
   - The HMM decoder receives no knowledge of the reference route.
   - The server computes pointwise accuracy internally by comparing `predicted_arcs` with `ground_truth_arcs`.
   - The returned payload includes the accuracy score and predicted route, never the raw ground-truth route sequence.

2. **Evaluator Audit Logging**:
   To satisfy grading requirements, every inference invocation triggers a synchronous log entry in `api/logs/requests.log` and standard output:
   `[TIMESTAMP] INFO: Evaluator request for Trip ID 00000040 | Processing Time: 184ms | Top-1 Accuracy: 89.4% | Route Continuity: 96.2%`

3. **Cross-Origin Resource Sharing (CORS)**:
   FastAPI incorporates `CORSMiddleware` configured with `allow_origins=["*"]` to ensure seamless communication across different local IP subnets and browser clients.
