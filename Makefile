# One command per step of the lifecycle. `make help` lists them.
# Windows without make: copy the command after each target into the terminal.
export MLFLOW_DISABLE_AGENT_HINT := 1
PY := python
VERSION ?= 2
BATCH ?= data/batches/batch_01.csv

.PHONY: help setup check train-logreg train-rf ui promote batch monitor export serve docker-build docker-run test clean

help:          ## list the steps
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*## /\t/'

setup:         ## 1. rebuild the exact environment
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

check:         ## 2. data version + data contract on the training data
	$(PY) -m titanic_mlops.data --verify
	$(PY) -m titanic_mlops.contract data/titanic.csv --training

train-logreg:  ## 3. train and register a logistic regression
	$(PY) -m titanic_mlops.train --model logreg

train-rf:      ## 3. train and register a random forest
	$(PY) -m titanic_mlops.train --model random_forest

ui:            ## 3. open the MLflow UI on port 5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

promote:       ## 4. try to promote VERSION=n to @champion
	$(PY) -m titanic_mlops.promote --version $(VERSION)

batch:         ## 5. score BATCH=file (contract first)
	$(PY) -m titanic_mlops.predict_batch $(BATCH)

monitor:       ## 7. drift report for BATCH=file
	$(PY) -m titanic_mlops.monitor $(BATCH)

export:        ## 6. copy @champion out of the registry for the image
	$(PY) -m titanic_mlops.export

serve:         ## 6. run the API locally on port 8000
	uvicorn titanic_mlops.serve:app --host 0.0.0.0 --port 8000

docker-build:  ## 6. build the serving image with the exported model
	docker build -t titanic-api:latest .

docker-run:    ## 6. run the serving image on port 8000
	docker run --rm -p 8000:8000 titanic-api:latest

test:          ## 8. what CI runs on every push
	pytest -q

clean:         ## start over: delete runs, registry, exports and predictions
	rm -rf mlruns mlflow.db model_export predictions
