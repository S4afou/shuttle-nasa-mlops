import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import wandb # MLOps Integration

# ==========================================
# 1. CONFIGURATION & MLOPS INIT
# ==========================================
# MLOPS CONFIG (REPLACE WITH YOUR W&B DETAILS)
ENTITY = "safou-seds-mlops-org" 
PROJECT = "nasa-shuttle-mlops"
MODEL_ARTIFACT = "production_lof_model:latest"

st.set_page_config(
    page_title="NASA Shuttle AI Diagnostic",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MLOPS: Initialize Monitoring Run ---
try:
    if wandb.run is None:
        wandb.init(project=PROJECT, entity=ENTITY, job_type="production_monitor")
except:
    pass

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #f5f7fa; }
    .success-box { padding: 25px; background-color: #d4edda; color: #155724; border-radius: 12px; border-left: 6px solid #28a745; box-shadow: 0 2px 8px rgba(40, 167, 69, 0.15); }
    .error-box { padding: 25px; background-color: #f8d7da; color: #721c24; border-radius: 12px; border-left: 6px solid #dc3545; box-shadow: 0 2px 8px rgba(220, 53, 69, 0.15); }
    </style>
    """, unsafe_allow_html=True)

# Session State Initialization (for history)
if 'history' not in st.session_state:
    st.session_state.history = []
if 'sensor_values' not in st.session_state:
    st.session_state.sensor_values = [0, 76, 0, 28, 18, 40, 48, 8] # Default normal state

# ==========================================
# 2. MLOPS MODEL LOADING (Continuous Deployment)
# ==========================================
@st.cache_resource
def load_production_model():
    """Downloads the LOF model directly from the W&B Registry."""
    try:
        api = wandb.Api()
        artifact = api.artifact(f"{ENTITY}/{PROJECT}/{MODEL_ARTIFACT}")
        dir_path = artifact.download()
        model_path = os.path.join(dir_path, "shuttle_lof_pipeline.pkl")
        
        return joblib.load(model_path), f"Version: {artifact.version}"
    except Exception as e:
        return None, str(e)

# Load Model
model, version_info = load_production_model()

# ==========================================
# 3. SIDEBAR & HEADER
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=180)
    st.title("🎛️ System Configuration")
    
    if model:
        st.success(f"🟢 Model Online: LOF\nVersion: {version_info}")
    else:
        st.error(f"🔴 Deployment Failed! {version_info}")
        st.stop()

    # --- Sidebar Presets Logic ---
    st.markdown("---")
    st.subheader("⚡ Quick Presets")

    # Preserve preset values
    preset_values = {
        'normal': [0, 76, 0, 28, 18, 40, 48, 8],
        'anomaly': [0, 92, 0, 0, 26, 36, 92, 56],
        'reset': [0, 0, 0, 0, 0, 0, 0, 0]
    }
    
    if st.button("🟢 Normal Values", use_container_width=True):
        st.session_state.sensor_values = preset_values['normal']
        st.rerun()
    if st.button("🔴 Anomaly Test", use_container_width=True):
        st.session_state.sensor_values = preset_values['anomaly']
        st.rerun()
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.sensor_values = preset_values['reset']
        st.rerun()
# ==========================================

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.title("🚀 NASA Shuttle Radiator Anomaly Detection")
st.markdown(f"**System Status:** **<span style='color:green'>🟢 ONLINE</span>**", unsafe_allow_html=True)
st.markdown("---")
st.header("📡 Telemetry Input Panel")

# Get values from session state to set the initial value of number_input
vals = st.session_state.sensor_values

# Sensor Input Grid (Original structure restored)
col1, col2, col3, col4 = st.columns(4)
with col1: s2 = st.number_input("🔵 Sensor A2", value=vals[0])
with col2: s3 = st.number_input("🟢 Sensor A3", value=vals[1])
with col3: s4 = st.number_input("🟡 Sensor A4", value=vals[2])
with col4: s5 = st.number_input("🟠 Sensor A5", value=vals[3])

col5, col6, col7, col8 = st.columns(4)
with col5: s6 = st.number_input("🔵 Sensor A6", value=vals[4])
with col6: s7 = st.number_input("🟢 Sensor A7", value=vals[5])
with col7: s8 = st.number_input("🟡 Sensor A8", value=vals[6])
with col8: s9 = st.number_input("🟠 Sensor A9", value=vals[7])

# Visualization of Inputs
s_vals = [s2, s3, s4, s5, s6, s7, s8, s9]
df_chart = pd.DataFrame({'Sensor': ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'], 'Value': s_vals})
st.plotly_chart(px.bar(df_chart, x='Sensor', y='Value', title='Current Sensor Readings', color='Value', color_continuous_scale='RdYlGn_r'), use_container_width=True)

# ==========================================
# 5. PREDICTION & MLOPS MONITORING
# ==========================================
if st.button("RUN DIAGNOSTICS", type="primary"):
    
    # 1. Prepare Input
    feature_names = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
    input_data_df = pd.DataFrame([s_vals], columns=feature_names) # Use current input values
    
    with st.spinner("🔄 Analyzing Sensor Patterns..."):
        # Predict
        prediction = model.predict(input_data_df)[0]
        score = model.decision_function(input_data_df)[0]
        status = "Normal" if prediction == 1 else "Anomaly"

        # 2. MLOPS MONITORING (Log to W&B)
        try:
            wandb.log({
                "production_input": input_data_df.to_dict(orient='list'), 
                "prediction_label": status,
                "anomaly_score": score,
                "model_version": version_info,
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass 

        # 3. Update Local History
        st.session_state.history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'score': score,
            'status': status
        })
        
        st.rerun()

# ==========================================
# 6. RESULTS DISPLAY & HISTORY
# ==========================================
if len(st.session_state.history) > 0:
    st.markdown("### 📋 Diagnostic Results")
    latest = st.session_state.history[-1]
    prediction = latest['prediction']
    score = latest['score']
    
    if prediction == 1:
        st.markdown(f"""
        <div class="success-box">
            <h2>✅ SYSTEM NOMINAL</h2>
            <p><b>🎯 Stability Score:</b> <b>{score:.4f}</b> (Positive = Normal)</p>
        </div>""", unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
        <div class="error-box">
            <h2>🚨 ANOMALY DETECTED</h2>
            <p><b>⚠️ Deviation Score:</b> <b>{score:.4f}</b> (Negative = Anomaly)</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.header("📈 Diagnostic History")
    history_df = pd.DataFrame([
        {'Time': h['timestamp'].strftime('%H:%M:%S'), 'Status': h['status'], 'Score': f"{h['score']:.4f}"}
        for h in st.session_state.history[-10:]
    ])
    st.dataframe(history_df, use_container_width=True)