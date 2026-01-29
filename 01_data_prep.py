import pandas as pd
import wandb
import os
from sklearn.datasets import fetch_openml

# CONFIGURATION
ENTITY = "safou-seds-mlops-org" 
PROJECT = "nasa-shuttle-mlops"

run = wandb.init(project=PROJECT, entity=ENTITY, job_type="data_prep")

print("⬇️ Loading Data from OpenML...")
shuttle = fetch_openml("shuttle", version=1, parser="auto")
df = shuttle.frame

print(f"Original Shape: {df.shape}")

# Preprocessing (from your notebook)
if 'A1' in df.columns: df = df.drop('A1', axis=1)
# Map Class: 1 -> 0 (Normal), Others -> 1 (Anomaly)
df['label'] = df['class'].apply(lambda x: 0 if str(x) == '1' else 1)
df = df.drop('class', axis=1)
df = df.drop_duplicates()

print(f"Cleaned Shape: {df.shape}")

# Save & Version
if not os.path.exists("data"): os.makedirs("data")
df.to_csv("data/clean_shuttle.csv", index=False)

# Log to W&B Artifacts (Data Versioning)
artifact = wandb.Artifact(
    name="shuttle_cleaned_data", 
    type="dataset",
    description="Preprocessed Shuttle Data (Time dropped, Duplicates removed, Binary classes)"
)
artifact.add_file("data/clean_shuttle.csv")
run.log_artifact(artifact)

print("✅ Data Versioned to W&B!")
run.finish()