import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(scope="session")
def trained():
    """A model trained exactly as train.py does, without touching the registry."""
    from titanic_mlops.config import FEATURES, TARGET
    from titanic_mlops.data import load_historical, split
    from titanic_mlops.model import build_pipeline

    train, test = split(load_historical())
    pipe = build_pipeline("logreg").fit(train[FEATURES], train[TARGET])
    return pipe, train, test


@pytest.fixture(scope="session")
def saved_model_uri(trained, tmp_path_factory):
    import mlflow.sklearn
    path = tmp_path_factory.mktemp("model") / "model"
    mlflow.sklearn.save_model(trained[0], str(path), skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"])
    return str(path)
