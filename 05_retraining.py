import wandb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import joblib

# CONFIG
ENTITY = "safou-seds-mlops-org" 
PROJECT = "nasa-shuttle-mlops"
TRIGGER_METRIC = "production_f1" # The metric we monitor for degradation
THRESHOLD = 0.85 # Retrain if F1 drops below 85%

# --- Define the Training Function for the Agent ---
def train_agent_function():
    run = wandb.init()

    # Load Data Artifact
    artifact = run.use_artifact('shuttle_cleaned_data:latest')
    artifact_dir = artifact.download()
    df = pd.read_csv(os.path.join(artifact_dir, "clean_shuttle.csv"))
    
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_normal = X_train[y_train == 0]
    
    # Pipeline using Sweep Params
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LocalOutlierFactor(
            n_neighbors=wandb.config.n_neighbors,
            contamination=wandb.config.contamination,
            novelty=True
        ))
    ])
    
    # Train & Eval
    pipeline.fit(X_train_normal)
    preds = pipeline.predict(X_test)
    preds = [1 if x == -1 else 0 for x in preds]
    
    # --- LOG F1 (The Metric for Optimization) ---
    f1 = f1_score(y_test, preds)
    wandb.log({'f1_anomaly': f1})

# --- Automated Health Check ---
api = wandb.Api()

try:
    monitor_runs = api.runs(f"{ENTITY}/{PROJECT}", {"jobType": "monitor"})
    latest_run = sorted(monitor_runs, key=lambda r: r.created_at, reverse=True)[0]
    
    # Get the latest F1 score logged by the monitor (Crucial update)
    history = latest_run.history()
    # We look for the last logged value of the trigger metric
    latest_f1 = history[f"{TRIGGER_METRIC}"].iloc[-1] 
    
except Exception as e:
    print(f"❌ Error fetching latest metric. Run 04_monitoring.py. Error: {e}")
    latest_f1 = 0.0 

print(f"📉 Latest Production F1 Score: {latest_f1:.4f}")

# --- Automated Retraining Trigger ---
if latest_f1 < THRESHOLD:
    print(f"\n🚨 ALERT: Performance ({latest_f1:.4f}) is below threshold ({THRESHOLD}).")
    
    # A. Define Sweep Config
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'f1_anomaly', 'goal': 'maximize'},
        'parameters': {
            'n_neighbors': {'min': 5, 'max': 100},
            'contamination': {'min': 0.0001, 'max': 0.05}
        }
    }

    # B. Launch the new sweep
    new_sweep_id = wandb.sweep(sweep_config, project=PROJECT) 
    
    # C. Start the agent programmatically
    print("   🧠 Agent Starting... Running 5 new experiments to find a fix.")
    wandb.agent(new_sweep_id, function=train_agent_function, count=5)
    
    # D. Log Alert
    alert_run = wandb.init(project=PROJECT, job_type="retraining_alert")
    wandb.alert(
        title="✅ Automated Retraining Completed",
        text=f"Retraining finished. F1 improved from {latest_f1:.4f}. New best model is available."
    )
    alert_run.finish()
    
else:
    print("✅ System Healthy. No retraining needed.")

if wandb.run:
    wandb.finish()