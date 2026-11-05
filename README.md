# mlops-titanic — an end-to-end MLOps example

A small, complete ML system around a classic problem: predict whether a Titanic passenger survived.
The model is deliberately simple. **The point is everything around it**: data versioning, a data
contract, experiment tracking, a model registry with a promotion rule, an API, a Docker image,
monitoring and CI — the lifecycle from the MLOps session, running for real.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jmvazqueznicolas/mlops-titanic)

| Step | Lifecycle idea (MLOps deck) | Command |
|---|---|---|
| 1 | Reproducibility: rebuild the exact environment | `make setup` |
| 2 | Data version + data contract | `make check` |
| 3 | Every run leaves a record | `make train-logreg`, `make train-rf`, `make ui` |
| 4 | Registry: *trained* ≠ *approved* | `make promote VERSION=1`, then `VERSION=2` |
| 5 | Batch serving, contract first; a silent failure | `make batch BATCH=…` |
| 6 | Online serving: API by alias, then Docker | `make serve`, `make export docker-build docker-run` |
| 7 | Monitoring: drift without labels, accuracy with labels | `make monitor BATCH=…` |
| 8 | CI: tests on every push | `make test`, then push |

---

## 0. Open the project

**Option A — Codespaces (recommended in class).** Click the badge above, or *Code → Codespaces →
Create codespace on main*. Wait for the container to finish installing (a few minutes the first
time). Everyone gets the same Python, the same packages and Docker, with nothing to install.

**Option B — your own computer** (Python 3.12 recommended):

```bash
git clone https://github.com/jmvazqueznicolas/mlops-titanic.git
cd mlops-titanic
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
make setup                                             # or: pip install -r requirements.txt && pip install -e .
```

No `make` on Windows? Each target in the `Makefile` is one line: copy it into the terminal.
Docker is only needed for the second half of step 6.

## 1. Reproducibility

`requirements.txt` pins exact versions. `params.yaml` holds every setting that defines a run (split,
seed, hyperparameters). The data file has a recorded hash. Code + data + environment + config + seed:
the five things that must be pinned to rebuild a model.

## 2. Data version and data contract

```bash
make check
```

```
data/titanic.csv sha256 8eb04eca7581…  OK
CONTRACT OK for data/titanic.csv: 791 rows
```

Try it: delete one line of `data/titanic.csv` and run `make check` again — the hash no longer
matches and **training refuses to run**. Restore it with `git checkout data/titanic.csv`.

The contract (`src/titanic_mlops/contract.py`) declares every column, its type, its allowed values,
and a limit on missing ages. `strict=True` rejects any column nobody declared.

## 3. Train and track

```bash
make train-logreg
make train-rf
make ui            # open port 5000 (Codespaces: "Ports" tab → MLflow UI)
```

Expected:

```
logreg:        accuracy 0.768 [95 % CI 0.704, 0.821]  vs baseline 0.611  ->  registered titanic-survival v1
random_forest: accuracy 0.778 [95 % CI 0.715, 0.830]  vs baseline 0.611  ->  registered titanic-survival v2
```

In the UI, open the experiment *titanic-survival*, select both runs and click **Compare**. Each run
records its parameters, the data hash, the git commit, accuracy with its confidence interval, ROC AUC,
the majority-class baseline, and the model file.

## 4. Registry and promotion

```bash
make promote VERSION=1     # no champion yet -> v1 becomes @champion
make promote VERSION=2     # does the forest beat it?
```

```
champion  v1: accuracy 0.768
candidate v2: accuracy 0.778
disagreements: champion right & candidate wrong = 7, candidate right & champion wrong = 9; McNemar exact p = 0.804
NOT PROMOTED v2: not distinguishable from champion (p = 0.804). @champion stays on v1.
```

One more point of accuracy on 198 test passengers comes from 16 disagreements, split 7 vs 9. That
is noise, so the rule refuses the promotion. A person can still override it, but only with a
recorded reason:

```bash
python -m titanic_mlops.promote --version 2 --force --reason "<who>: <why>"
```

The reason is stored as a tag on the model version: that is the audit trail.

## 5. Batch serving — and a silent failure

```bash
make batch BATCH=data/batches/batch_01.csv
# 100 predictions -> predictions/batch_01_predictions.csv | predicted survivors: 35.0%

make batch BATCH=data/batches/batch_02_schema_change.csv
# STOPPED: ... breaks the data contract ... Sex isin(['male', 'female'])  Male / Female
```

(`make` then reports `Error 1`. That is the point: the pipeline stopped.)

`batch_02` contains **the same 100 passengers**, but an upstream form now writes `Sex` as
"Male"/"Female". Now skip the contract, as a pipeline without one would:

```bash
python -m titanic_mlops.predict_batch data/batches/batch_02_schema_change.csv --skip-contract
# 100 predictions ... predicted survivors: 43.0%   (contract SKIPPED)
```

No error, no warning — and 34 of the 100 passengers get a different answer. The one-hot encoder
treats the unknown "Male" as "no category at all". This is why the contract runs **before** the model.

## 6. Online serving: API, then Docker

```bash
make serve         # port 8000; docs at /docs
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 80, "Embarked": "S"}'
# {"survival_probability":0.9239,"survived":1, ...}
```

Send `"Sex": "Male"` and the API answers **422**: the input contract at the door. The API loads
`models:/titanic-survival@champion`, so promoting another version changes no API code.

Package it (Docker required; in Codespaces it is already there):

```bash
make export          # copies the current @champion to model_export/
make docker-build    # image = API + that one model version
make docker-run      # same curl as above; /health shows which version is inside
```

## 7. Monitoring

```bash
make monitor BATCH=data/batches/batch_01.csv
make monitor BATCH=data/batches/batch_03_drift.csv
python -m titanic_mlops.monitor data/batches/batch_01.csv --labels data/labels/batch_01_labels.csv
```

`batch_01` is stable (every PSI < 0.1). `batch_03` is a different population — mostly 3rd-class men:
PSI flags `Pclass`, `Sex` and `Fare`, and predicted survivors drop from 35 % to 11 %. Nothing here
needs labels. When the labels of `batch_01` "arrive", accuracy is 0.850 [0.767, 0.907] — on 100
passengers, an interval that wide is the honest answer.

## 8. Continuous integration

```bash
make test     # 15 tests: data, model and API
```

`.github/workflows/ci.yml` runs the hash check, the contract and the tests on every push. Try
breaking it: in `contract.py` change the `Fare` range to `(0, 100)`, commit and push. The *Actions*
tab turns red, because the historical data no longer satisfies the contract.

---

## Exercises

a) Change `C` in `params.yaml` to `0.1`, retrain, and try to promote. What decides?
b) Which parameters in the UI prove two runs are comparable?
c) Which line of the contract stops `batch_02`? Comment it out and run step 5 again: what changes?
d) Why does the API reject "Male" while `--skip-contract` accepts it?
e) Which feature would you watch first in production, and with what threshold?
f) Write the `--reason` you would accept for a forced promotion, and one you would refuse.

## The data

- `data/titanic_kaggle_train_original.csv`: Kaggle's *Titanic* `train.csv` (891 passengers, 342
  survivors), from the `datasciencedojo/datasets` mirror on GitHub.
- `data/titanic.csv`: 791 of them, the historical data for training (its hash is in `titanic.csv.sha256`).
- `data/batches/batch_01.csv` + `data/labels/batch_01_labels.csv`: the other 100, without and with labels.
- `batch_02_schema_change.csv` and `batch_03_drift.csv` are **simulated** from the real passengers.
  `scripts/make_batches.py` shows exactly how every file was produced (seed 2026).

## Honest limitations

- Nobody deploys a Titanic model. The business framing is artificial; the engineering is not.
- PSI cut-offs (0.1 / 0.25) are an industry convention, not a statistical test.
- The tracking store and registry are a local SQLite file. In a team they live on a shared server;
  the code does not change, only `MLFLOW_TRACKING_URI`.

## Repository map

```
src/titanic_mlops/   config (features, paths) · data (hash, split) · contract · model (pipeline)
                     train · promote · predict_batch · serve (FastAPI) · monitor · export · stats
tests/               data, model and API tests (what CI runs)
params.yaml          every setting of a run
Dockerfile           serving image: API + one baked model version (requirements-serve.txt)
.devcontainer/       the Codespaces environment
```
