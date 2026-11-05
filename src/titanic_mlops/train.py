"""Train one model, log everything that defines it, and register it as a candidate.

    python -m titanic_mlops.train --model logreg
    python -m titanic_mlops.train --model random_forest
"""
import argparse
import subprocess

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, roc_auc_score

from titanic_mlops.config import EXPERIMENT, FEATURES, MODEL_NAME, PARAMS, ROOT, TARGET
from titanic_mlops.contract import validate
from titanic_mlops.data import load_historical, sha256, split
from titanic_mlops.model import build_pipeline
from titanic_mlops.stats import wilson


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return (out.stdout.strip() or "no-commit") + ("-dirty" if dirty else "")
    except FileNotFoundError:
        return "no-git"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(PARAMS["models"]), default="logreg")
    args = ap.parse_args()

    df = validate(load_historical(), training=True)          # 1. right data version, 2. contract holds
    train, test = split(df)
    pipe = build_pipeline(args.model).fit(train[FEATURES], train[TARGET])

    proba = pipe.predict_proba(test[FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)
    n, k = len(test), int((pred == test[TARGET]).sum())
    lo, hi = wilson(k, n)
    baseline = max(test[TARGET].mean(), 1 - test[TARGET].mean())   # always predict the majority class

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.model) as run:
        mlflow.log_params({"model": args.model, **{f"{args.model}.{k_}": v for k_, v in PARAMS["models"][args.model].items()},
                           "split.test_size": PARAMS["split"]["test_size"], "split.seed": PARAMS["split"]["seed"],
                           "data_sha256": sha256()[:12], "git_commit": git_commit(),
                           "n_train": len(train), "n_test": n})
        mlflow.log_metrics({"test_accuracy": k / n, "test_accuracy_ci_low": lo, "test_accuracy_ci_high": hi,
                            "test_roc_auc": roc_auc_score(test[TARGET], proba), "baseline_accuracy": baseline})
        # Integer columns cannot hold missing values, so the recorded input schema uses floats.
        X_sig = test[FEATURES].astype({c: "float64" for c in ["Pclass", "SibSp", "Parch"]})
        info = mlflow.sklearn.log_model(
            pipe, name="model", registered_model_name=MODEL_NAME,
            input_example=X_sig.head(3), signature=infer_signature(X_sig, proba),
            # MLflow 3 saves scikit-learn models with skops, which refuses types it does not trust.
            # OneHotEncoder stores a numpy dtype and random forests store sklearn Tree objects;
            # we declare exactly those as safe instead of falling back to pickle.
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
    print(f"run {run.info.run_id[:8]}  {args.model}: accuracy {k / n:.3f} [95 % CI {lo:.3f}, {hi:.3f}]"
          f"  vs baseline {baseline:.3f}  ->  registered {MODEL_NAME} v{info.registered_model_version}")


if __name__ == "__main__":
    main()
