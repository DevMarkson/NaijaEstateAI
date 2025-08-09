import json
from pathlib import Path
import subprocess
import sys
import os

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BEST_MODEL_PATH, METRICS_PATH, DATA_PATH


def test_training_runs(tmp_path):
    # Run training on a small slice for speed
    cmd = [
        sys.executable, "train.py",
        "--data", str(DATA_PATH),
        "--test-size", "0.3",
        "--skip-xgb",
        "--cv-folds", "2"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Training failed: {res.stderr}\n{res.stdout}"
    assert BEST_MODEL_PATH.exists(), "Best model file not created"
    assert METRICS_PATH.exists(), "Metrics file not created"
    data = json.loads(METRICS_PATH.read_text())
    assert "best" in data and data["best"], "Best model not recorded"
    assert "results" in data and isinstance(data["results"], dict), "Results missing"
