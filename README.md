# Bank marketing campaign models

## Train

With pandas, numpy, scikit-learn, joblib, and xgboost installed in your Python environment:

```bash
python3 src/data_ingestion_pipeline_xgboost.py
```

Training reads `parameters.json` and saves classifiers and evaluation reports to its configured `models_dir`.

## Run the frontend and backend

To train models and start the server in one step:

```bash
./build_and_run.sh
# Optional port:
./build_and_run.sh --port 8080
```

The launcher works from any directory, uses an active virtual environment or the project's `.venv` when available, checks dependencies, and stops if training fails. Models are rebuilt using `parameters.json` on each launch. The frontend is served directly without a build step. To choose a Python interpreter explicitly, use `PROJECT_PYTHON=/path/to/python ./build_and_run.sh`. Stop the server with Ctrl+C.

From the project root, using the same Python environment used for training:

```bash
python3 src/server.py
```

Open **http://127.0.0.1:8000**. The dashboard compares saved evaluation results and predicts term deposit subscription using any of the four classifiers. No Node installation or frontend build is required. Optional server arguments: `--port 8080` and `--host 127.0.0.1`.

- `frontend/`: HTML, CSS, and browser JavaScript.
- `src/server.py`: local HTTP backend, serving the frontend and JSON API on the same origin.
- `GET /api/dashboard`: saved model scores and training metadata.
- `POST /api/predict`: `{ "model": "xgboost", "features": { ... } }`; accepts the 15 pre-call predictor fields shown in the form and returns `probability`, `prediction`, and `threshold`.

The frontend requires models trained with `include_duration: false`. The server supplies the raw schema's required duration placeholder, which the saved pipeline removes before inference. Model files are read from the directory in `parameters.json`; predictions load the selected file on each request so a completed retraining run is picked up without restarting. Use only your own trusted joblib artifacts.

Refresh results after training. If model files are missing, the UI displays an actionable error. The server defaults to localhost and is intended for local development, without authentication or production hosting configuration. Customer form submissions are processed in memory and are not stored.

## Verify

```bash
python3 -m unittest discover -s tests
```
