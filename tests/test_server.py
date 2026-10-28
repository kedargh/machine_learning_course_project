"""Exercise real saved pipelines and HTTP handler behavior without binding a port."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from server import DashboardHandler, MODEL_NAMES


class RequestHarness(DashboardHandler):
    def __init__(self, path, payload=None, models_dir=None):
        self.path = path
        body = json.dumps(payload).encode() if payload is not None else b''
        self.headers = {'Content-Length': str(len(body))}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.models_dir = models_dir or ROOT / 'models'
        self.directory = str(ROOT / 'frontend')
        self.command = 'GET'
    def send_response(self, code, message=None):
        self.status = code
    def send_header(self, *args):
        pass
    def end_headers(self):
        pass
    def log_error(self, *args):
        pass
    def result(self):
        return json.loads(self.wfile.getvalue())


class ApiTests(unittest.TestCase):
    def profile(self):
        return dict(age=42, job='management', marital='married', education='secondary',
                    balance=1500, default='no', housing='no', loan='no', contact='cellular',
                    day=15, month='may', campaign=1, pdays=-1, previous=0, poutcome='unknown')
    def test_dashboard(self):
        handler = RequestHarness('/api/dashboard')
        handler.do_GET()
        self.assertEqual(handler.status, 200)
        self.assertEqual(len(handler.result()['scores']), 4)
        self.assertTrue(all('brier_score' not in row for row in handler.result()['scores']))
    def test_all_saved_classifiers(self):
        for model in MODEL_NAMES:
            with self.subTest(model=model):
                handler = RequestHarness('/api/predict', {'model': model, 'features': self.profile()})
                handler.do_POST()
                self.assertEqual(handler.status, 200, handler.result())
                result = handler.result()
                self.assertTrue(0 <= result['probability'] <= 1)
                self.assertEqual(result['prediction'], 'yes' if result['probability'] >= result['threshold'] else 'no')
    def test_bad_requests(self):
        bad = [[], {}, {'model': '../../etc/passwd', 'features': self.profile()},
               {'features': {}}, {'features': dict(self.profile(), age='NaN')},
               {'features': dict(self.profile(), day=32)}]
        for payload in bad:
            handler = RequestHarness('/api/predict', payload)
            handler.do_POST()
            self.assertEqual(handler.status, 400, handler.result())
    def test_missing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            handler = RequestHarness('/api/dashboard', models_dir=Path(folder))
            handler.do_GET()
            self.assertEqual(handler.status, 503)
    def test_frontend_served(self):
        handler = RequestHarness('/')
        handler.do_GET()
        self.assertEqual(handler.status, 200)
        self.assertIn(b'CampaignLab', handler.wfile.getvalue())


if __name__ == '__main__':
    unittest.main()
