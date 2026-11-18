# PathSnap Evaluator UI (Laptop 2)

This module provides the evaluator dashboard running on **Laptop 2**. It connects to the FastAPI backend running on **Laptop 1** over local Wi-Fi / hotspot.

## Execution
```cmd
# Activate environment
.venv\Scripts\activate

# Launch Streamlit app
streamlit run ui/app.py
```

## Features
- **Configurable Backend Host**: Enter Laptop 1's IP address (e.g. `http://192.168.1.50:8000`) and test connectivity.
- **Trip Selector**: Displays the 10 reserved evaluation trips (`00000000`, `00000007`, `00000009`, ...).
- **KPI Metrics**: Real-time display of Observation-level Top-1 Accuracy, Route Continuity, and Latency Breakdown.
- **Interactive Map**: Displays raw GPS points and predicted road network arcs.
