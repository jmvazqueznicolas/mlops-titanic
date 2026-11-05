"""Load the historical data, check its version, and build the fixed train/test split.

    python -m titanic_mlops.data --verify     # fails if data/titanic.csv changed
"""
import argparse
import hashlib
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from titanic_mlops.config import DATA, DATA_HASH_FILE, PARAMS, TARGET


def sha256(path=DATA) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_sha256() -> str:
    return DATA_HASH_FILE.read_text().split()[0]


def load_historical(verify: bool = True) -> pd.DataFrame:
    if verify and sha256() != expected_sha256():
        raise SystemExit(
            f"data/titanic.csv does not match its recorded hash.\n"
            f"  expected {expected_sha256()}\n  found    {sha256()}\n"
            "If the change is intended, record the new version: update data/titanic.csv.sha256 and commit it."
        )
    return pd.read_csv(DATA)


def split(df: pd.DataFrame):
    """Same seed and same data -> the same test set for every run."""
    p = PARAMS["split"]
    train, test = train_test_split(df, test_size=p["test_size"], random_state=p["seed"], stratify=df[TARGET])
    return train, test


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        ok = sha256() == expected_sha256()
        print(f"data/titanic.csv sha256 {sha256()[:12]}…  {'OK' if ok else 'MISMATCH'}")
        sys.exit(0 if ok else 1)
