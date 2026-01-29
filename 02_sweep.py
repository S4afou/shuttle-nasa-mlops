import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
import os

# Configuration (REPLACE WITH YOUR W&B DETAILS)
ENTITY = "safou-seds-mlops-org" 
PROJECT = "nasa-shuttle-mlops"

# Define Sweep Config (We maximize F1)
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'f1_anomaly', 'goal': 'maximize'},
    'parameters': {
        'n_neighbors': {'min': 5, 'max': 100},
        'contamination': {'min': 0.0001, 'max': 0.05}
    }
}

def train():
    run = wandb.init()
    
    # Load Data Artifact
    artifact = run.use_artifact('shuttle_cleaned_data:latest')
    artifact_dir = artifact.download(root="artifacts_cache")
    data_path = os.path.join(artifact_dir, "clean_shuttle.csv")
    df = pd.read_csv(data_path)
    
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_normal = X_train[y_train == 0]
    
    # Build Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LocalOutlierFactor(
            n_neighbors=wandb.config.n_neighbors,
            contamination=wandb.config.contamination,
            novelty=True
        ))
    ])
    
    pipeline.fit(X_train_normal)
    
    # Evaluate
    preds = pipeline.predict(X_test)
    preds = [1 if x == -1 else 0 for x in preds]
    
    # --- LOG ALL 3 METRICS ---
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    
    wandb.log({
        'f1_anomaly': f1,
        'precision_anomaly': precision,
        'recall_anomaly': recall
    })

# Run Sweep
print("Launching Sweep...")
sweep_id = wandb.sweep(sweep_config, project=PROJECT, entity=ENTITY)
print(f"Sweep ID: {sweep_id}")
wandb.agent(sweep_id, train, count=15)