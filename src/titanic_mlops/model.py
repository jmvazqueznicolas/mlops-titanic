"""The model is a scikit-learn Pipeline: preprocessing + classifier travel together.

Because imputation, scaling and encoding are inside the pipeline, the API and batch scoring
apply exactly the transformation learned in training. Nothing is re-implemented for serving.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic_mlops.config import CATEGORICAL, NUMERIC, PARAMS


def build_pipeline(model: str) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        # handle_unknown="ignore": an unseen category becomes all zeros instead of an error.
        # Convenient, and exactly why the data contract must run first (see batch_02).
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    p = PARAMS["models"][model]
    if model == "logreg":
        clf = LogisticRegression(C=p["C"], max_iter=1000)
    elif model == "random_forest":
        clf = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                     random_state=p["seed"], n_jobs=-1)
    else:
        raise ValueError(f"unknown model {model!r}; options: {list(PARAMS['models'])}")
    return Pipeline([("pre", pre), ("clf", clf)])
