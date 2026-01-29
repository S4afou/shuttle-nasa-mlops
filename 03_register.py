import wandb
import pandas as pd
import joblib
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# CONFIGURATION
ENTITY = "safou-seds" 
PROJECT = "nasa-shuttle-mlops"
METRIC_NAME = "f1_anomaly" # The metric we want to maximize

def get_global_best_run():
    api = wandb.Api()
    print(f"🔍 Connecting to W&B Project: {ENTITY}/{PROJECT}...")
    
    # 1. Get ALL sweeps in the project
    sweeps = api.project(PROJECT, entity=ENTITY).sweeps()
    
    if len(sweeps) == 0:
        raise ValueError("❌ No sweeps found in this project!")
    
    print(f"found {len(sweeps)} sweeps. Searching for the global winner...")
    
    global_best_run = None
    global_best_score = -1.0
    
    # 2. Iterate through every sweep to find the champion
    for sweep in sweeps:
        best_run_in_sweep = sweep.best_run()
        
        # Check if the sweep actually has a run and the metric exists
        if best_run_in_sweep and METRIC_NAME in best_run_in_sweep.summary:
            score = best_run_in_sweep.summary[METRIC_NAME]
            
            print(f"   - Sweep {sweep.id}: Best F1 = {score:.4f}")
            
            if score > global_best_score:
                global_best_score = score
                global_best_run = best_run_in_sweep
    
    if global_best_run is None:
        raise ValueError("❌ Could not find any runs with the target metric.")

    print(f"\n🏆 GLOBAL BEST RUN FOUND: {global_best_run.name}")
    print(f"   Sweep ID: {global_best_run.sweep.id}")
    print(f"   F1-Score: {global_best_score:.4f}")
    print(f"   Params:   n_neighbors={global_best_run.config['n_neighbors']}, contamination={global_best_run.config['contamination']}")
    
    return global_best_run

def register_best_model():
    # Get the absolute best run from history
    best_run = get_global_best_run()
    
    # Initialize Registry Run
    run = wandb.init(project=PROJECT, job_type="model_registry")
    
    # Load Data Artifact
    print("\n⬇️ Downloading clean data...")
    artifact = run.use_artifact('shuttle_cleaned_data:latest')
    data_dir = artifact.download()
    df = pd.read_csv(os.path.join(data_dir, "clean_shuttle.csv"))
    
    # Split Data
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_normal = X_train[y_train == 0]
    
    print("⚙️ Retraining Final Model with Best Params...")
    
    # Build Pipeline with WINNING params
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LocalOutlierFactor(
            n_neighbors=best_run.config['n_neighbors'],
            contamination=best_run.config['contamination'],
            novelty=True
        ))
    ])
    
    pipeline.fit(X_train_normal)
    
    # Save Locally
    model_filename = "shuttle_lof_pipeline.pkl"
    joblib.dump(pipeline, model_filename)
    
    # Upload to Registry
    print("⬆️  Registering to W&B...")
    model_artifact = wandb.Artifact("production_lof_model", type="model")
    model_artifact.add_file(model_filename)
    run.log_artifact(model_artifact)
    
    print("✅ Global Best Model Registered Successfully!")
    run.finish()

if __name__ == "__main__":
    register_best_model()