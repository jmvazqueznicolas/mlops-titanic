"""Infrastructure tests: the API answers, rejects bad input, and serves the model we trained."""
import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from titanic_mlops.config import FEATURES

GOOD = {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 80.0, "Embarked": "S"}


@pytest.fixture(scope="module")
def client(saved_model_uri, monkeypatch_module):
    monkeypatch_module.setenv("MODEL_URI", saved_model_uri)
    import titanic_mlops.serve as serve
    importlib.reload(serve)
    return TestClient(serve.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_valid_passenger(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200 and 0 <= r.json()["survival_probability"] <= 1


@pytest.mark.parametrize("bad", [{**GOOD, "Sex": "Male"}, {**GOOD, "Pclass": 4}, {**GOOD, "Fare": -5},
                                 {k: v for k, v in GOOD.items() if k != "Sex"}])
def test_invalid_input_is_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_served_prediction_equals_trained_model(client, trained):
    expected = trained[0].predict_proba(pd.DataFrame([GOOD])[FEATURES])[0, 1]
    assert abs(client.post("/predict", json=GOOD).json()["survival_probability"] - round(expected, 4)) < 1e-9
