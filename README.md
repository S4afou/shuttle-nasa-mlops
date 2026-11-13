# NASA shuttle anomaly detection: an MLOps lifecycle

I built a working MLOps loop around an anomaly detector for NASA shuttle radiator
telemetry. Six phases, one script each: data versioning, hyperparameter search, model
registration, deployment, drift monitoring and automated retraining. Weights and Biases
holds the runs, the artifacts and the sweep engine; Streamlit is the deployment.

The detector itself, with its three pre-trained pipelines and the diagnostic panel, lives in
[shuttle-nasa-ML](https://github.com/S4afou/shuttle-nasa-ML).

## Phases

| Phase | Script | What it does | Proof in W&B |
| --- | --- | --- | --- |
| 1. Data versioning | `01_data_prep.py` | Fetches the shuttle data, drops the time column, de-duplicates and stores the cleaned frame as an immutable artifact | `shuttle_cleaned_data` |
| 2. Experimentation | `02_sweep.py` | Launches a Bayesian sweep over the LOF hyperparameters | Runs table with F1, precision and recall |
| 3. Registration | `03_register.py` | Finds the best run across all sweeps and registers it as the production model | `production_lof_model` |
| 4. Deployment | `app.py` | Streamlit panel that pulls the registered model version at start | Live interface |
| 5. Monitoring | `04_monitoring.py` | Replays ten production batches and injects sensor drift into the last ones | Time-series metrics |
| 6. Retraining | `05_retraining.py` | Checks the latest production F1 and triggers a new sweep below the threshold | Retraining run |

## Results

- Registered sweep: anomaly F1 **0.979**.
- After injected drift: production F1 **0.432**.
- Retraining threshold: **0.80**. The drift run falls below it, which is exactly the event
  the loop is built to catch.

The point of the project is the last row of that table. A model that reports 0.979 and then
silently degrades to 0.432 is the normal failure mode in production, so the interesting part
is the trigger, not the score.

## Setup

Python 3.10 is required by the pinned dependencies.

```bash
git clone https://github.com/S4afou/shuttle-nasa-mlops.git
cd shuttle-nasa-mlops
pip install -r requirements.txt
wandb login
```

## Run the loop

```bash
python 01_data_prep.py     # version the data as an artifact
python 02_sweep.py         # launch the sweep, then run the agent command it prints
python 03_register.py      # register the best run as production
streamlit run app.py       # serve the registered model
python 04_monitoring.py    # replay ten batches, with drift at the end
python 05_retraining.py    # check the threshold and retrain if needed
```

## Repository layout

```text
01_data_prep.py     phase 1, data versioning
02_sweep.py         phase 2, Bayesian hyperparameter search
03_register.py      phase 3, best-run selection and registry
04_monitoring.py    phase 5, production traffic and drift simulation
05_retraining.py    phase 6, threshold check and automated retraining
app.py              phase 4, Streamlit deployment
requirements.txt
```

## Notes

- Nothing is hard-coded to a local path. Each phase reads the artifact it needs from the
  registry, so the loop can restart at any phase.
- I simulate ten batches of live telemetry in the monitoring script and inject noise into
  the final ones, which is how the drift is produced without a real sensor feed.
- `wandb/` and `data/` are ignored. Runs live in the cloud project, not in git.

## Dataset

OpenML `shuttle` (version 1): nine numeric sensor channels, about 56,000 rows, seven
original classes mapped to normal against anomaly. The class balance is heavily skewed, so
the pipeline reports F1 rather than accuracy.

More about this project on my portfolio:
https://safouane-medjahri.vercel.app/projects/shuttle-anomaly-detection
