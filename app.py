import streamlit as st
import time
import json
from backend.ipc.client import IPCClient
from backend.config import config

st.set_page_config(page_title="Motor Health IPC Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1040 40%, #0d1b3e 100%) !important;
        color: white;
    }
    h1, h2, h3, p, div {
        color: white !important;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("AeroForge IPC Dashboard")
st.markdown("Connected to Backend Daemon via ZeroMQ. Machine Learning inference is temporarily disabled during ONNX migration.")

@st.cache_resource
def get_ipc_client():
    cfg = config.get_ipc_config()
    return IPCClient(address=cfg['address'], timeout_ms=cfg['timeout_ms'])

try:
    client = get_ipc_client()
    
    # Send requests
    health_res = client.send_request("health")
    ver_res = client.send_request("version")
    stats_res = client.send_request("buffer_stats")
    tel_res = client.send_request("latest_telemetry")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Backend Status", health_res.get("status", "Offline"))
        
    with col2:
        st.metric("Version", ver_res.get("version", "N/A"))
        
    with col3:
        if "error" not in stats_res:
            usage = f"{stats_res.get('size', 0)} / {stats_res.get('capacity', 0)}"
            st.metric("Buffer Usage", usage)
        else:
            st.metric("Buffer Usage", "Error")
            
    st.subheader("Latest Telemetry (Live)")
    if "error" not in tel_res and tel_res.get("data"):
        st.json(tel_res.get("data"))
    else:
        st.info("No telemetry available yet. Ensure MQTT broker is sending data.")
        
    time.sleep(2)
    st.rerun()

except Exception as e:
    st.error(f"Failed to connect to Backend Daemon: {e}")
    st.info("Ensure the Backend Daemon is running.")
    time.sleep(2)
    st.rerun()
