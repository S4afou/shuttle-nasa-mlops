import wandb
import pandas as pd
import joblib
import numpy as np
import time
import os
from datetime import datetime, timedelta
from sklearn.metrics import f1_score, precision_score, recall_score

# CONFIG
ENTITY = "safou-seds-mlops-org" 
PROJECT = "nasa-shuttle-mlops"

run = wandb.init(
    project=PROJECT,
    job_type="monitor",
    notes="Monitoring run tracking F1, Precision, and Recall drift."
)

# ... (Loading Model and Data - Assume successful) ...
try:
    artifact_model = run.use_artifact(f'{ENTITY}/{PROJECT}/production_lof_model:latest')
    model_dir = artifact_model.download()
    model = joblib.load(os.path.join(model_dir, "shuttle_lof_pipeline.pkl"))

    artifact_data = run.use_artifact(f'{ENTITY}/{PROJECT}/shuttle_cleaned_data:latest')
    data_dir = artifact_data.download()
    df = pd.read_csv(os.path.join(data_dir, "clean_shuttle.csv"))
    
    X = df.drop('label', axis=1)
    y = df['label']
    
except Exception as e:
    print(f"❌ Error during loading: {e}")
    run.finish()
    exit()

# Simulation Parameters
num_batches = 10
batch_size = 300
start_time = datetime.now()

for i in range(num_batches):
    print(f"--- Processing Batch {i+1}/{num_batches} ---")
    time.sleep(1) 

    # 1. Sample Batch & Inject Noise (Concept Drift)
    indices = np.random.choice(len(X), batch_size, replace=False)
    X_batch = X.iloc[indices].copy()
    y_batch_ground_truth = y.iloc[indices].copy()
    if i >= num_batches - 3: # Induce drift on last 3 batches
        X_batch['A2'] = X_batch['A2'] + np.random.normal(0, 50.0, batch_size)

    # 2. Predict & Evaluate
    y_pred_raw = model.predict(X_batch) 
    y_pred_mapped = [1 if x == -1 else 0 for x in y_pred_raw] 

    # --- LOG ALL 3 METRICS FOR MONITORING ---
    f1 = f1_score(y_batch_ground_truth, y_pred_mapped, zero_division=0)
    precision = precision_score(y_batch_ground_truth, y_pred_mapped, zero_division=0)
    recall = recall_score(y_batch_ground_truth, y_pred_mapped, zero_division=0)
    
    # 3. Log to W&B
    log_time = start_time + timedelta(hours=i)

    wandb.log({
        "production_f1": f1,
        "production_precision": precision,
        "production_recall": recall,
        "log_timestamp": log_time
    })
    
    print(f"   Batch F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

print("✅ Monitoring Simulation Complete.")
run.finish()