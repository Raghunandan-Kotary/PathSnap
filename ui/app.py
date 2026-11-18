"""
PathSnap Evaluator Dashboard (Laptop 2)
Interacts with the Laptop 1 FastAPI backend to trigger map-matching and display metrics and route maps.
"""
import streamlit as st
import requests
import json
import time

st.set_page_config(
    page_title="PathSnap Evaluator Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/route.png", width=64)
st.sidebar.title("PathSnap Controls")
st.sidebar.markdown("**Laptop 2 Evaluator Client**")

api_base_url = st.sidebar.text_input(
    "FastAPI Server URL (Laptop 1)",
    value="http://localhost:8000",
    help="Enter the IP address of Laptop 1, e.g. http://192.168.1.45:8000"
)

# Connection Check
if st.sidebar.button("Test API Connection"):
    try:
        resp = requests.get(f"{api_base_url}/health", timeout=3)
        if resp.status_code == 200:
            st.sidebar.success(f"Connected! API v{resp.json().get('version', '1.0')}")
        else:
            st.sidebar.error(f"HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.info(
    "Ground truth routes stay on Laptop 1. The model and UI never access them. "
    "Every evaluation call is verified in real-time."
)

# Main UI
st.markdown('<div class="main-title">PathSnap: Distributed Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time GPS trajectory map-matching via ML classification and Viterbi decoding</div>', unsafe_allow_html=True)

# Fetch Trips
@st.cache_data(ttl=15)
def get_trips(base_url: str):
    try:
        r = requests.get(f"{base_url}/trips", timeout=3)
        if r.status_code == 200:
            return r.json().get("trips", [])
    except Exception:
        pass
    # Fallback to standard 10 student testing trips
    return ["00000000", "00000007", "00000009", "00000040", "00000052", 
            "00000056", "00000065", "00000076", "00000077", "00000091"]

available_trips = get_trips(api_base_url)

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    selected_trip = st.selectbox(
        "Select an Evaluation Trip (10 Reserved Trips):",
        options=available_trips,
        index=0
    )
with col_btn:
    st.write("")
    st.write("")
    run_eval = st.button("🚀 Evaluate Route", type="primary", use_container_width=True)

if run_eval and selected_trip:
    with st.spinner(f"Sending Trip {selected_trip} to Laptop 1 for map-matching & evaluation..."):
        try:
            t0 = time.time()
            res = requests.post(f"{api_base_url}/predict/{selected_trip}", timeout=30)
            network_latency = (time.time() - t0) * 1000
            
            if res.status_code == 200:
                data = res.json()
                metrics = data["metrics"]
                timing = data["timing"]
                predicted_arcs = data["predicted_arcs"]

                st.success(f"Trip **{selected_trip}** processed successfully in {timing['total_latency_ms']} ms!")

                # KPI Cards
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric(
                    "Top-1 Accuracy",
                    f"{metrics['pointwise_top1_accuracy'] * 100:.2f}%",
                    help="Observation-level match against server-side reference route"
                )
                kpi2.metric(
                    "Route Continuity",
                    f"{metrics['route_continuity'] * 100:.2f}%",
                    help="Fraction of consecutive predicted arcs with network topological connectivity"
                )
                kpi3.metric(
                    "Observations",
                    f"{metrics['matched_observations']} / {metrics['total_observations']}",
                    help="Correctly matched GPS observations"
                )
                kpi4.metric(
                    "Server Latency",
                    f"{timing['total_latency_ms']:.1f} ms",
                    delta=f"Network RTT: {network_latency:.1f} ms"
                )

                st.markdown("### ⏱️ Latency Breakdown")
                t_cols = st.columns(4)
                t_cols[0].caption(f"Data Load: **{timing['data_load_ms']} ms**")
                t_cols[1].caption(f"Features: **{timing['feature_extraction_ms']} ms**")
                t_cols[2].caption(f"Inference: **{timing['model_inference_ms']} ms**")
                t_cols[3].caption(f"Viterbi Decode: **{timing['viterbi_decode_ms']} ms**")

                st.markdown("---")
                
                # Visualizations
                st.markdown("### 🗺️ Route Map & Predicted Segments")
                if data.get("map_html"):
                    st.components.v1.html(data["map_html"], height=520, scrolling=True)
                else:
                    st.info("Interactive map layer rendered with Folium. Use Person 3's map module to view HTML map.")

                # Trajectory Arc Sequence Explorer
                with st.expander(f"View Predicted Arc Sequence ({len(predicted_arcs)} road segments)"):
                    st.write(predicted_arcs[:50], "... (truncated)" if len(predicted_arcs) > 50 else "")

            else:
                st.error(f"Error {res.status_code}: {res.json().get('detail', res.text)}")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to {api_base_url}. Please ensure Laptop 1 is running `uvicorn api.main:app` and both laptops share the network.")
        except Exception as e:
            st.error(f"Execution failed: {str(e)}")
