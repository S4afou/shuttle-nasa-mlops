import wandb
import pandas as pd
import joblib
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# CONFIGURATION
ENTITY = "safou-seds-mlops-org" 
PROJECT = "nasa-shuttle-mlops"
METRIC_NAME = "f1_anomaly"

def get_global_best_run():
    api = wandb.Api()
    
    print(f"🔍 Connecting to W&B Project: {ENTITY}/{PROJECT}...")
    
    # --- FIX START ---
    # 1. Fetch ALL runs in the project (simplest API call)
    all_runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    if not all_runs:
        raise ValueError("❌ No runs found in this project! Run 02_sweep.py first.")
    
    print(f"✅ Found {len(all_runs)} runs. Filtering and sorting locally...")

    # 2. Filter for valid training runs and runs that have the metric
    valid_runs = []
    for run in all_runs:
        # Check if the run is from a training job and has the F1 score logged
        # We also filter out runs that aren't finished (state != "finished")
        if run.job_type != 'data_prep' and run.job_type != 'monitor' and run.job_type != 'retraining_alert':
            if METRIC_NAME in run.summary and run.state == 'finished':
                valid_runs.append(run)

    if not valid_runs:
        raise ValueError("❌ No valid, finished training runs found with the target metric.")

    # 3. Sort locally by the Metric Name
    valid_runs.sort(key=lambda r: r.summary.get(METRIC_NAME, -1.0), reverse=True)
    
    global_best_run = valid_runs[0]
    global_best_score = global_best_run.summary.get(METRIC_NAME)
    
    # --- FIX END ---

    print(f"\n🏆 GLOBAL BEST RUN FOUND: {global_best_run.name}")
    print(f"   Sweep ID: {global_best_run.sweep.id if global_best_run.sweep else 'N/A'}")
    print(f"   F1-Score: {global_best_score:.4f}")
    print(f"   Params:   n_neighbors={global_best_run.config['n_neighbors']}, contamination={global_best_run.config['contamination']:.6f}")
    
    return global_best_run

def register_best_model():
    best_run = get_global_best_run()
    
    run = wandb.init(project=PROJECT, entity=ENTITY, job_type="model_registry")
    
    # 1. Load Data
    print("⬇️ Downloading clean data...")
    artifact = run.use_artifact('shuttle_cleaned_data:latest')
    data_dir = artifact.download()
    df = pd.read_csv(os.path.join(data_dir, "clean_shuttle.csv"))
    
    # Split Data
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_normal = X_train[y_train == 0]
    
    print("⚙️ Retraining Final Model with Best Params...")
    
    # Build Pipeline with WINNING params from the best run's config
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LocalOutlierFactor(
            n_neighbors=best_run.config['n_neighbors'],
            contamination=best_run.config['contamination'],
            novelty=True
        ))
    ])
    
    pipeline.fit(X_train_normal)
    
    # 2. Save & Register
    model_filename = "shuttle_lof_pipeline.pkl"
    joblib.dump(pipeline, model_filename)
    
    model_artifact = wandb.Artifact("production_lof_model", type="model")
    model_artifact.add_file(model_filename)
    run.log_artifact(model_artifact)
    
    print("✅ Model Registered Successfully!")
    run.finish()

if __name__ == "__main__":
    register_best_model()