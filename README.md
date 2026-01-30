# 🚀 NASA Shuttle Anomaly Detection: Full MLOps Lifecycle

This repository contains a professional MLOps implementation for detecting anomalies in the **NASA Shuttle Radiator** telemetry data. The project implements the core MLOps cycle—Data Versioning, Experiment Tracking, Model Registry, Production Monitoring, and Automated Retraining—utilizing **Weights & Biases (W&B)** and **Streamlit**.

## 📁 Project Organization
The repository is structured to ensure 100% reproducibility and modularity:

```text
shuttle-nasa-ML/
├── 01_data_prep.py        # [PHASE 1] Data versioning & preprocessing
├── 02_sweep.py            # [PHASE 2] Hyperparameter search (Sweeps)
├── 03_register.py         # [PHASE 3] Best model selection & registration
├── 04_monitoring.py       # [PHASE 5] Production traffic & drift simulation
├── 05_retraining.py       # [PHASE 6] Automated retraining (Closing the loop)
├── app.py                 # [DEPLOYMENT] Interactive Streamlit interface
├── requirements.txt       # System dependencies
└── README.md              # Documentation & Walkthrough
```

## 🛠️ Core MLOps Lifecycle Implementation
This project respects the MLOps principles by implementing the following stages:

| MLOps Stage | Status | Implementation Detail | W&B Proof Point |
| :--- | :---: | :--- | :--- |
| **Data Versioning** | ✅ | Cleans raw data and stores it as an immutable artifact. | `shuttle_cleaned_data:latest` |
| **Experimentation** | ✅ | Executes Bayesian optimization via W&B Sweeps. | Runs Table (F1, Precision, Recall) |
| **Model Registry** | ✅ | Identifies the global best run and registers the production model. | `production_lof_model:latest` |
| **Deployment** | ✅ | Streamlit app pulling the latest model version from the registry. | Live Production Endpoint |
| **Monitoring** | ✅ | Simulates 10 production batches with injected Concept Drift. | Time-series metrics charts |
| **Retraining** | ✅ | Automated threshold check triggering a new Sweep on failure. | W&B Retraining Alert |

---

## ⚙️ Reproducibility: Environment Setup
To replicate the environment and execute the pipeline, follow these instructions.
NB: you need python 3.10 for this to work.

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/S4afou/shuttle-nasa-ML.git
cd shuttle-nasa-ML

# Install dependencies
pip install -r requirements.txt

# Authenticate your machine with Weights & Biases
wandb login
```

## 🚀 Execution Flow (Step-by-Step Walkthrough)
Execute the scripts in the following order to generate the MLOps artifacts:

1. **Phase 1: Version Data**
   `python 01_data_prep.py`
   *Registers the cleaned Shuttle dataset in the W&B Artifacts cloud.*

2. **Phase 2: Experimentation**
   `python 02_sweep.py`
   *Launches the Bayesian sweep. Execute the agent command printed in the terminal to run trials.*

3. **Phase 3: Registration**
   `python 03_register.py`
   *Finds the absolute best run across all sweeps and registers it for production.*

4. **Phase 4: Deployment**
   `streamlit run app.py`
   *Runs the web application, pulling the production model dynamically from the cloud.*

5. **Phase 5: Monitoring**
   `python 04_monitoring.py`
   *Simulates 10 batches of live telemetry. Injects noise in the final batches to simulate sensor drift.*

6. **Phase 6: Retraining**
   `python 05_retraining.py`
   *Detects performance degradation (F1 < 0.85) and automatically triggers a new healing sweep.*

---
