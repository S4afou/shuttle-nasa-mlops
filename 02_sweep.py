import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# 1. Define Sweep Config (Your Search Space)
sweep_config = {
    'method': 'bayes', # Smarter than random search
    'metric': {'name': 'f1_anomaly', 'goal': 'maximize'},
    'parameters': {
        'n_neighbors': {'min': 5, 'max': 100},
        'contamination': {'min': 0.0001, 'max': 0.05}
    }
}

def train():
    run = wandb.init()
    
    # 2. Load Data Artifact
    artifact = run.use_artifact('shuttle_cleaned_data:latest')
    artifact_dir = artifact.download()
    df = pd.read_csv(f"{artifact_dir}/clean_shuttle.csv")
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Semi-Supervised: Train only on Normal (Class 0)
    X_train_normal = X_train[y_train == 0]
    
    # 3. Build Pipeline
    # We put Scaler inside pipeline (Best Practice)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LocalOutlierFactor(
            n_neighbors=wandb.config.n_neighbors,
            contamination=wandb.config.contamination,
            novelty=True
        ))
    ])
    
    # Fit
    pipeline.fit(X_train_normal)
    
    # Predict
    preds_raw = pipeline.predict(X_test)
    preds = [1 if x == -1 else 0 for x in preds_raw]
    
    # Evaluate
    f1 = f1_score(y_test, preds)
    
    # Log
    wandb.log({'f1_anomaly': f1})

# 4. Initialize Sweep
sweep_id = wandb.sweep(sweep_config, project="nasa-shuttle-mlops")
wandb.agent(sweep_id, train, count=15) # Run 15 experiments (Same as your notebook)