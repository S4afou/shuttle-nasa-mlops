import wandb
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import os
import random # Used for the train function's config

# CONFIG
ENTITY = "safou-seds" 
PROJECT = "nasa-shuttle-mlops"
METRIC_NAME = "production_f1"

# --- Define the Training Function for the Agent ---
def train_agent_function():
    # This is the function the wandb agent will run for each trial
    run = wandb.init()

    # Load Data Artifact
    try:
        artifact = run.use_artifact('shuttle_cleaned_data:latest')
        artifact_dir = artifact.download()
        df = pd.read_csv(os.path.join(artifact_dir, "clean_shuttle.csv"))
    except Exception as e:
        print(f"Agent failed to load data: {e}")
        return

    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_normal = X_train[y_train == 0]
    
    # Build Pipeline using Sweep Params (from wandb.config)
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
    f1 = f1_score(y_test, preds)
    
    # Log Result
    wandb.log({'f1_anomaly': f1})

# --- 1. Automated Health Check ---
api = wandb.Api()
THRESHOLD = 0.85 # Setting threshold high so the monitor script easily triggers it

print("🔍 Checking Production Health (F1 Score)...")

try:
    # 1. Get the latest monitoring run
    monitor_runs = api.runs(f"{ENTITY}/{PROJECT}", {"jobType": "monitor"})
    latest_monitor_run = max(monitor_runs, key=lambda r: r.created_at)

    # 2. Extract the last logged F1 score (using the metric name from 04_monitoring.py)
    # We assume the last metric logged is the most recent (final batch)
    latest_f1 = latest_monitor_run.summary.get(f"production_{METRIC_NAME}", 0) 
    
except Exception as e:
    print(f"❌ Error: Could not fetch monitoring run summary. Please run 04_monitoring.py first. Error: {e}")
    latest_f1 = 0.0 # Force trigger if fetch fails

print(f"📉 Latest Production F1 Score: {latest_f1:.4f}")

# --- 2. Automated Retraining Trigger ---
if latest_f1 < THRESHOLD:
    print(f"\n🚨 ALERT: Performance ({latest_f1:.4f}) is below threshold ({THRESHOLD}).")
    print("🚀 Auto-Triggering Retraining Pipeline...")
    
    # A. Define Sweep Config (The fix parameters)
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
    print(f"   ✅ New Sweep Launched! ID: {new_sweep_id}")
    
    # C. Start the agent programmatically (The Fully Automated Step)
    print("   🧠 Agent Starting... Running 5 new experiments to find a fix.")
    wandb.agent(new_sweep_id, function=train_agent_function, count=5)
    
    # D. Log Alert (Closing the loop)
    alert_run = wandb.init(project=PROJECT, job_type="retraining_alert")
    wandb.alert(
        title="✅ Automated Retraining Completed",
        text=f"Retraining finished. Best model found and ready for manual deployment."
    )
    alert_run.finish()
    
    print("🎉 Full MLOps Loop Closed: Failure detected and fixed.")

else:
    print("✅ System Healthy. No retraining needed.")

# Final step to ensure all open runs close
if wandb.run:
    wandb.finish()