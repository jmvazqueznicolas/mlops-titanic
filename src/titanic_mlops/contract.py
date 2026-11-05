"""Data contract: rules every batch must satisfy BEFORE the model may use it.

    python -m titanic_mlops.contract data/titanic.csv --training
    python -m titanic_mlops.contract data/batches/batch_02_schema_change.csv
"""
import argparse
import sys

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

_columns = {
    "PassengerId": pa.Column(int, unique=True),
    "Pclass": pa.Column(int, pa.Check.isin([1, 2, 3])),
    "Name": pa.Column(str),
    "Sex": pa.Column(str, pa.Check.isin(["male", "female"])),
    "Age": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
    "SibSp": pa.Column(int, pa.Check.in_range(0, 10)),
    "Parch": pa.Column(int, pa.Check.in_range(0, 10)),
    "Ticket": pa.Column(str),
    "Fare": pa.Column(float, pa.Check.in_range(0, 1000)),
    "Cabin": pa.Column(str, nullable=True),
    "Embarked": pa.Column(str, pa.Check.isin(["C", "Q", "S"]), nullable=True),
}
# Historical data: 19.9 % of ages are missing. A batch with far more is a broken feed, not new passengers.
_completeness = pa.Check(lambda d: d["Age"].isna().mean() < 0.35, error="more than 35 % of Age is missing")

BATCH_SCHEMA = pa.DataFrameSchema(_columns, strict=True, coerce=False, checks=[_completeness])
TRAINING_SCHEMA = BATCH_SCHEMA.add_columns({"Survived": pa.Column(int, pa.Check.isin([0, 1]))})


def validate(df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """Raises SchemaErrors listing every violation (lazy=True collects all of them)."""
    return (TRAINING_SCHEMA if training else BATCH_SCHEMA).validate(df, lazy=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--training", action="store_true", help="the file must also contain Survived")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    try:
        validate(df, training=args.training)
    except SchemaErrors as e:
        print(f"CONTRACT FAILED for {args.csv}: {len(e.failure_cases)} violation(s)")
        print(e.failure_cases[["column", "check", "failure_case"]].drop_duplicates().head(10).to_string(index=False))
        sys.exit(1)
    print(f"CONTRACT OK for {args.csv}: {len(df)} rows")


if __name__ == "__main__":
    main()
