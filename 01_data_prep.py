import pandas as pd
import wandb
import os
from sklearn.datasets import fetch_openml

# 1. Initialize W&B Run
run = wandb.init(project="nasa-shuttle-mlops", job_type="data_prep")

print("⬇️ Loading Data from OpenML...")
shuttle = fetch_openml("shuttle", version=1, parser="auto")
df = shuttle.frame

print(f"Original Shape: {df.shape}")

# 2. Preprocessing (Logic from your notebook)
# A. Drop Time Column (A1)
if 'A1' in df.columns:
    df = df.drop('A1', axis=1)

# B. Drop Duplicates
df = df.drop_duplicates().reset_index(drop=True)

# C. Class Mapping (1=Normal->0, Others->1)
# Note: OpenML targets might be integers or strings, enforcing str comparison for safety
df['label'] = df['class'].apply(lambda x: 0 if str(x) == '1' else 1)
df = df.drop('class', axis=1)

print(f"Cleaned Shape: {df.shape}")

# 3. Save & Version
if not os.path.exists("data"): os.makedirs("data")
df.to_csv("data/clean_shuttle.csv", index=False)

# Log to W&B
artifact = wandb.Artifact(
    name="shuttle_cleaned_data", 
    type="dataset",
    description="Preprocessed Shuttle Data (No Time, No Duplicates)"
)
artifact.add_file("data/clean_shuttle.csv")
run.log_artifact(artifact)

print("✅ Data Versioned to W&B!")
run.finish()