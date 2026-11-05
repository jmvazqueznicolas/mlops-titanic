"""How the files in data/ were produced from Kaggle's Titanic train.csv (891 passengers).

Run once by the instructor; students do NOT need to run it. It is kept so that every
data file in the repo can be rebuilt and audited.

- data/titanic.csv            791 passengers with labels: the "historical" data we train on.
- data/batches/batch_01.csv   100 held-out passengers WITHOUT labels: a normal new batch.
- data/labels/batch_01_labels.csv   their true outcome, which "arrives later".
- data/batches/batch_02_schema_change.csv   the same 100 passengers, but an upstream form
                              now writes Sex as "Male"/"Female". SIMULATED.
- data/batches/batch_03_drift.csv   150 passengers resampled so that 3rd-class men dominate,
                              as if a different population arrived. SIMULATED.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "titanic_kaggle_train_original.csv"
rng = np.random.default_rng(2026)

df = pd.read_csv(SRC)
assert df.shape == (891, 12) and df["Survived"].sum() == 342, "not Kaggle's train.csv"

future_idx = rng.choice(df.index, size=100, replace=False)
future = df.loc[sorted(future_idx)]
historical = df.drop(index=future_idx)

historical.to_csv(ROOT / "data" / "titanic.csv", index=False)
future.drop(columns="Survived").to_csv(ROOT / "data" / "batches" / "batch_01.csv", index=False)
future[["PassengerId", "Survived"]].to_csv(ROOT / "data" / "labels" / "batch_01_labels.csv", index=False)

changed = future.drop(columns="Survived").copy()
changed["Sex"] = changed["Sex"].str.capitalize()
changed.to_csv(ROOT / "data" / "batches" / "batch_02_schema_change.csv", index=False)

w = np.where((historical["Pclass"] == 3) & (historical["Sex"] == "male"), 8.0, 1.0)
drift_idx = rng.choice(historical.index, size=150, replace=True, p=w / w.sum())
drift = historical.loc[drift_idx].drop(columns="Survived").copy()
drift["PassengerId"] = np.arange(5001, 5001 + len(drift))
drift.to_csv(ROOT / "data" / "batches" / "batch_03_drift.csv", index=False)

digest = hashlib.sha256((ROOT / "data" / "titanic.csv").read_bytes()).hexdigest()
(ROOT / "data" / "titanic.csv.sha256").write_text(f"{digest}  titanic.csv\n")
print("historical", historical.shape, "survived", int(historical.Survived.sum()))
print("batch_01", future.shape[0], "| drift share 3rd-class men:",
      round(((drift.Pclass == 3) & (drift.Sex == "male")).mean(), 3),
      "vs historical", round(((historical.Pclass == 3) & (historical.Sex == "male")).mean(), 3))
print("sha256", digest)
