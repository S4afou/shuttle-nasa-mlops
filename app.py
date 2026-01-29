import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import wandb

# ==========================================
# 1. CONFIGURATION
# ==========================================
# REPLACE WITH YOUR DETAILS
ENTITY = "safou-seds" 
PROJECT = "nasa-shuttle-mlops"
MODEL_ARTIFACT = "production_lof_model:latest"

st.set_page_config(
    page_title="NASA Shuttle AI Diagnostic",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize W&B for Real-Time Monitoring
try:
    if wandb.run is None:
        wandb.init(project=PROJECT, entity=ENTITY, job_type="app_inference")
except:
    pass

# CSS
st.markdown("""
    <style>
    .stApp { background-color: #f5f7fa; }
    .success-box { padding: 20px; background-color: #d4edda; border-left: 5px solid #28a745; }
    .error-box { padding: 20px; background-color: #f8d7da; border-left: 5px solid #dc3545; }
    </style>
    """, unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# ==========================================
# 2. CONTINUOUS DEPLOYMENT (CD) LOADER
# ==========================================
@st.cache_resource
def get_production_model():
    """
    Downloads the LATEST 'production' model from W&B.
    This ensures if Retraining happens, the App gets the update.
    """
    try:
        print(f"⬇️ Connecting to Registry: {ENTITY}/{PROJECT}...")
        api = wandb.Api()
        artifact = api.artifact(f"{ENTITY}/{PROJECT}/{MODEL_ARTIFACT}")
        dir_path = artifact.download()
        model_path = os.path.join(dir_path, "shuttle_lof_pipeline.pkl")
        
        return joblib.load(model_path), f"Version: {artifact.version}"
    except Exception as e:
        return None, str(e)

# Load Model
model, version_info = get_production_model()

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=180)
    st.title("🎛️ Flight Control")
    
    if model:
        st.success(f"🟢 System Online\n{version_info}")
    else:
        st.error(f"🔴 System Offline\nError: {version_info}")
        st.stop()

    st.markdown("---")
    st.subheader("⚡ Simulators")
    if st.button("✅ Force Normal"):
        st.session_state.preset = [0, 76, 0, 28, 18, 40, 48, 8]
        st.rerun()
    if st.button("🚨 Force Anomaly"):
        st.session_state.preset = [0, 92, 0, 0, 26, 36, 92, 56]
        st.rerun()

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.title("🚀 Shuttle Radiator Anomaly Detection")
st.caption(f"MLOps Status: Monitoring Active • Registry Connected")

# Handle Presets
if 'preset' in st.session_state:
    vals = st.session_state.preset
    del st.session_state.preset
else:
    vals = [0, 76, 0, 28, 18, 40, 48, 8]

st.subheader("📡 Live Sensor Telemetry")
c1, c2, c3, c4 = st.columns(4)
with c1: s2 = st.number_input("A2 (Temp)", value=vals[0])
with c2: s3 = st.number_input("A3 (Flow)", value=vals[1])
with c3: s4 = st.number_input("A4 (Pressure)", value=vals[2])
with c4: s5 = st.number_input("A5 (Rad)", value=vals[3])
c5, c6, c7, c8 = st.columns(4)
with c5: s6 = st.number_input("A6", value=vals[4])
with c6: s7 = st.number_input("A7", value=vals[5])
with c7: s8 = st.number_input("A8", value=vals[6])
with c8: s9 = st.number_input("A9", value=vals[7])

# Bar Chart
df_chart = pd.DataFrame({
    'Sensor': ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'],
    'Value': [s2, s3, s4, s5, s6, s7, s8, s9]
})
st.plotly_chart(px.bar(df_chart, x='Sensor', y='Value', color='Value'), use_container_width=True)

# ==========================================
# 5. PREDICTION & MONITORING
# ==========================================
if st.button("RUN DIAGNOSTICS", type="primary"):
    
    # 1. Prepare Input (Array for LOF)
    input_data = np.array([[s2, s3, s4, s5, s6, s7, s8, s9]])
    
    # 2. Predict
    pred = model.predict(input_data)[0]
    try: score = model.decision_function(input_data)[0]
    except: score = 0.0
    
    status = "Normal" if pred == 1 else "Anomaly"
    
    # 3. MLOPS LOGGING (This appears in W&B Project Dashboard)
    try:
        wandb.log({
            "live_input": {
                "s2": s2, "s3": s3, "s4": s4, "s5": s5, 
                "s6": s6, "s7": s7, "s8": s8, "s9": s9
            },
            "live_prediction": status,
            "live_score": score,
            "timestamp": datetime.now().isoformat()
        })
    except:
        pass # Don't crash if internet fails

    # 4. Display
    if pred == 1:
        st.markdown(f"""<div class="success-box">
            <h3>✅ SYSTEM NOMINAL</h3>
            <p>Score: {score:.4f}</p>
        </div>""", unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""<div class="error-box">
            <h3>🚨 ANOMALY DETECTED</h3>
            <p>Score: {score:.4f}</p>
        </div>""", unsafe_allow_html=True)
    
    # 5. Local History Table
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Status": status,
        "Score": f"{score:.4f}"
    })

if st.session_state.history:
    st.markdown("---")
    st.write("### 📜 Event Log")
    st.dataframe(pd.DataFrame(st.session_state.history).tail(5), use_container_width=True)