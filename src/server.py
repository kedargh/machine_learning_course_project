"""Local dashboard and inference API. Run: python3 src/server.py."""
from __future__ import annotations

import argparse
import json
import math
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd

# Registers the custom transformer module alias used in saved joblib pipelines.
import data_ingestion_pipeline_xgboost as training

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAMES = ('logistic_regression', 'decision_tree', 'random_forest', 'xgboost')


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, models_dir: Path, **kwargs):
        self.models_dir = models_dir
        super().__init__(*args, directory=str(ROOT / 'frontend'), **kwargs)

    def respond(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self, name):
        return json.loads((self.models_dir / name).read_text())

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/dashboard':
            try:
                metadata = self.read_json('training_metadata.json')
                scores = self.read_json('model_scores.json')
                for score in scores:
                    score.pop('brier_score', None)
                self.respond(200, {'scores': scores, 'metadata': metadata})
            except FileNotFoundError:
                self.respond(503, {'error': 'No trained results found. Run the training pipeline first.'})
            except (ValueError, OSError):
                self.respond(503, {'error': 'Training results could not be read. Complete a training run and try again.'})
        elif path.startswith('/api/'):
            self.respond(404, {'error': 'Endpoint not found.'})
        else:
            super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != '/api/predict':
            self.respond(404, {'error': 'Endpoint not found.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                raise ValueError('Request body must be between 1 and 16384 bytes.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Expected a JSON object.')
            name = payload.get('model', 'xgboost')
            if name not in MODEL_NAMES:
                raise ValueError('Choose a supported classifier.')
            features = payload.get('features')
            if not isinstance(features, dict):
                raise ValueError('Provide customer features as an object.')
            required = set(training.BankFeatureEngineer.RAW_PREDICTOR_COLUMNS) - {'duration'}
            missing = sorted(required - features.keys())
            if missing:
                raise ValueError('Missing fields: ' + ', '.join(missing))
            features = {key: features[key] for key in required}
            metadata = self.read_json('training_metadata.json')
            if metadata.get('duration_included'):
                raise ValueError('This dashboard requires pre-call models. Retrain with include_duration=false.')
            features['duration'] = 0  # Required by the raw schema; excluded by pre-call models.
            for key in training.BankFeatureEngineer.NUMERIC_COLUMNS:
                try:
                    features[key] = float(features[key])
                except (TypeError, ValueError):
                    raise ValueError(f'{key} must be a number.') from None
                if not math.isfinite(features[key]):
                    raise ValueError(f'{key} must be finite.')
            if features['age'] < 18 or not 1 <= features['day'] <= 31:
                raise ValueError('Age must be at least 18 and day must be between 1 and 31.')
            if features['campaign'] < 1 or features['previous'] < 0 or features['pdays'] < -1:
                raise ValueError('Invalid campaign, previous contacts, or days since contact.')
            for key in required - set(training.BankFeatureEngineer.NUMERIC_COLUMNS):
                if not isinstance(features[key], str) or not features[key].strip():
                    raise ValueError(f'{key} must be a nonempty string.')
            model = joblib.load(self.models_dir / f'{name}.joblib')
            probability = float(model.predict_proba(pd.DataFrame([features]))[0, 1])
            threshold = float(metadata.get('decision_threshold', 0.5))
            self.respond(200, {'model': name, 'probability': probability,
                               'threshold': threshold, 'prediction': 'yes' if probability >= threshold else 'no'})
        except FileNotFoundError:
            self.respond(503, {'error': 'Trained model files are missing. Run the training pipeline first.'})
        except (ValueError, TypeError) as error:
            self.respond(400, {'error': str(error)})
        except Exception:
            self.log_error('Model inference failed')
            self.respond(500, {'error': 'Model could not be loaded or evaluated. Retrain models in this Python environment.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    config = json.loads((ROOT / 'parameters.json').read_text())
    models_dir = Path(config.get('models_dir') or 'src/models').expanduser()
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    handler = partial(DashboardHandler, models_dir=models_dir.resolve())
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f'Bank campaign dashboard: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
