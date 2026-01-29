import wandb
import pandas as pd
import joblib
import numpy as np
import time
import os
from datetime import datetime, timedelta
from sklearn.metrics import f1_score, precision_score, recall_score

# CONFIG
ENTITY = "safou-seds"  # REPLACE WITH YOUR USERNAME
PROJECT = "nasa-shuttle-mlops"

# 1. Initialize Monitoring Run
run = wandb.init(
    project=PROJECT,
    job_type="monitor",
    notes="Simulating production traffic and tracking F1 score drift."
)

print("🚀 Starting Production Monitoring Simulation...")

# 2. Load the Production Model
try:
    # We grab the latest model tagged 'production' from registry
    print("⬇️  Downloading Production Model...")
    artifact = run.use_artifact(f'{ENTITY}/{PROJECT}/production_lof_model:latest')
    model_dir = artifact.download()
    model = joblib.load(os.path.join(model_dir, "shuttle_lof_pipeline.pkl"))
    print("✅ Model Loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    run.finish()
    exit()

# 3. Load Test Data (To simulate incoming traffic)
# In real life, this would be a live stream. We will sample from our saved clean data.
data_artifact = run.use_artifact(f'{ENTITY}/{PROJECT}/shuttle_cleaned_data:latest')
data_dir = data_artifact.download()
df = pd.read_csv(os.path.join(data_dir, "clean_shuttle.csv"))

X = df.drop('label', axis=1)
y = df['label']

# 4. Simulate Batches
num_batches = 10
batch_size = 200
start_time = datetime.now()

for i in range(num_batches):
    print(f"--- Processing Batch {i+1}/{num_batches} ---")
    
    # Simulate time delay (e.g. 1 hour between checks)
    # We use sleep(1) so you don't have to wait forever
    time.sleep(1) 
    
    # A. Sample a random batch of data
    # We purposefully sample to create variance in the F1 score
    indices = np.random.choice(len(X), batch_size, replace=False)
    X_batch = X.iloc[indices]
    y_batch_ground_truth = y.iloc[indices]
    
    # B. Inject Noise (To simulate degradation/drift)
    # On the last few batches, we make the data "noisy" to force the score down
    # This ensures Phase 6 (Retraining) will actually trigger!
    if i >= 7: 
        print("⚠️ Simulating Sensor Drift (Injecting Noise)...")
        X_batch = X_batch + np.random.normal(0, 2.0, X_batch.shape)

    # C. Predict
    preds = model.predict(X_batch)
    # Map LOF output (-1=Anomaly, 1=Normal) to (1=Anomaly, 0=Normal)
    preds_mapped = [1 if x == -1 else 0 for x in preds]
    
    # D. Calculate Metrics
    # We calculate metrics against the ground truth to see how well we are doing
    f1 = f1_score(y_batch_ground_truth, preds_mapped, zero_division=0)
    anom_rate = sum(preds_mapped) / len(preds_mapped)
    
    # Create fake timestamp for the graph
    log_time = start_time + timedelta(hours=i)

    # E. Log to W&B
    wandb.log({
        "production_f1": f1,
        "anomaly_rate": anom_rate,
        "batch_size": batch_size,
        "log_timestamp": log_time
    })
    
    print(f"   Batch F1: {f1:.4f} | Anomaly Rate: {anom_rate:.2%}")

print("✅ Monitoring Simulation Complete.")
run.finish()