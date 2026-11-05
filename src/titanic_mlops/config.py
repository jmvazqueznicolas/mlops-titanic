"""Single source of truth for paths, names and feature lists.

Training, batch scoring and the API all import FEATURES from here, so a feature cannot be
defined one way for training and another way for serving (training-serving skew).
"""
import os
from pathlib import Path

import yaml

_repo = Path(__file__).resolve().parents[2]
# Editable install (pip install -e .): the repo root. Inside the Docker image: the working directory.
ROOT = _repo if (_repo / "params.yaml").exists() else Path.cwd()
DATA = ROOT / "data" / "titanic.csv"
DATA_HASH_FILE = ROOT / "data" / "titanic.csv.sha256"
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())

TARGET = "Survived"
NUMERIC = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL

EXPERIMENT = "titanic-survival"
MODEL_NAME = "titanic-survival"
CHAMPION_ALIAS = "champion"

os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
