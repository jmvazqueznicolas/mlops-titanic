"""Batch scoring: contract first, then predict a whole file.

    python -m titanic_mlops.predict_batch data/batches/batch_01.csv
    python -m titanic_mlops.predict_batch data/batches/batch_02_schema_change.csv
    python -m titanic_mlops.predict_batch data/batches/batch_02_schema_change.csv --skip-contract   # what NOT to do
"""
import argparse
import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from pandera.errors import SchemaErrors

from titanic_mlops.config import CHAMPION_ALIAS, FEATURES, MODEL_NAME, ROOT
from titanic_mlops.contract import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--skip-contract", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    if not args.skip_contract:
        try:
            validate(df)
        except SchemaErrors as e:
            print(f"STOPPED: {args.csv} breaks the data contract ({len(e.failure_cases)} violations). "
                  "No predictions were written. First violations:")
            print(e.failure_cases[["column", "check", "failure_case"]].drop_duplicates().head(5).to_string(index=False))
            sys.exit(1)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    out = df[["PassengerId"]].copy()
    out["survival_probability"] = model.predict_proba(df[FEATURES])[:, 1].round(4)
    out["survived"] = (out["survival_probability"] >= 0.5).astype(int)
    dest = ROOT / "predictions" / f"{Path(args.csv).stem}_predictions.csv"
    dest.parent.mkdir(exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"{len(out)} predictions -> {dest.relative_to(ROOT)} | predicted survivors: {out['survived'].mean():.1%}"
          + ("   (contract SKIPPED)" if args.skip_contract else ""))


if __name__ == "__main__":
    main()
