"""Monitoring a batch against the training data.

    python -m titanic_mlops.monitor data/batches/batch_01.csv
    python -m titanic_mlops.monitor data/batches/batch_03_drift.csv
    python -m titanic_mlops.monitor data/batches/batch_01.csv --labels data/labels/batch_01_labels.csv

Data layer (no labels needed): PSI per feature. PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 shift.
These cut-offs are an industry convention, not a statistical test.
Model layer: share predicted to survive now vs in training; accuracy once labels arrive.
"""
import argparse

import mlflow.sklearn
import numpy as np
import pandas as pd

from titanic_mlops.config import CATEGORICAL, CHAMPION_ALIAS, FEATURES, MODEL_NAME, NUMERIC
from titanic_mlops.data import load_historical
from titanic_mlops.stats import wilson

EPS = 1e-4


def psi(ref: pd.Series, new: pd.Series, categorical: bool) -> float:
    if categorical:
        cats = sorted(set(ref.dropna().astype(str)) | set(new.dropna().astype(str)))
        r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        n = new.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.nanquantile(ref, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref.dropna(), edges)[0] / ref.notna().sum()
        n = np.histogram(new.dropna(), edges)[0] / new.notna().sum()
    r, n = np.clip(r, EPS, None), np.clip(n, EPS, None)
    return float(np.sum((n - r) * np.log(n / r)))


def verdict(v: float) -> str:
    return "SHIFT" if v > 0.25 else ("watch" if v > 0.1 else "stable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--labels")
    args = ap.parse_args()
    ref, new = load_historical(), pd.read_csv(args.csv)

    print(f"Batch {args.csv}: {len(new)} passengers vs {len(ref)} historical\n")
    print(f"{'feature':<10}{'PSI':>8}  verdict")
    for f in FEATURES:
        v = psi(ref[f], new[f], f in CATEGORICAL)
        print(f"{f:<10}{v:>8.3f}  {verdict(v)}")

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    p_new = (model.predict_proba(new[FEATURES])[:, 1] >= 0.5).mean()
    p_ref = (model.predict_proba(ref[FEATURES])[:, 1] >= 0.5).mean()
    print(f"\nPredicted survivors: {p_new:.1%} in this batch vs {p_ref:.1%} on historical data")

    if args.labels:
        lab = pd.read_csv(args.labels).merge(new, on="PassengerId")
        k = int((model.predict(lab[FEATURES]) == lab["Survived"]).sum()); n = len(lab)
        lo, hi = wilson(k, n)
        print(f"Labels arrived for {n}: accuracy {k / n:.3f} [95 % CI {lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
