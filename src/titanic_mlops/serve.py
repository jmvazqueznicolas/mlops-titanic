"""Online serving: a function behind an HTTP endpoint.

    uvicorn titanic_mlops.serve:app --port 8000
    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
         -d '{"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 80, "Embarked": "S"}'

The model is loaded by alias (models:/titanic-survival@champion), so promoting a new version
changes no code here. Inside Docker, MODEL_URI points to the model baked into the image.
"""
import json
import os
from pathlib import Path
from typing import Literal, Optional

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from titanic_mlops.config import CHAMPION_ALIAS, FEATURES, MODEL_NAME

MODEL_URI = os.environ.get("MODEL_URI", f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
model = mlflow.sklearn.load_model(MODEL_URI)
_meta = Path(MODEL_URI).parent / "VERSION.json"
MODEL_INFO = json.loads(_meta.read_text()) if _meta.exists() else {"model_uri": MODEL_URI}

app = FastAPI(title="Titanic survival", version="1.0")


class Passenger(BaseModel):
    """Input contract at the door: anything else is rejected with HTTP 422."""
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(default=None, ge=0, le=100)
    SibSp: int = Field(ge=0, le=10)
    Parch: int = Field(ge=0, le=10)
    Fare: float = Field(ge=0, le=1000)
    Embarked: Optional[Literal["C", "Q", "S"]] = None


@app.get("/health")
def health():
    return {"status": "ok", **MODEL_INFO}


@app.post("/predict")
def predict(p: Passenger):
    X = pd.DataFrame([p.model_dump()])[FEATURES]
    prob = float(model.predict_proba(X)[0, 1])
    return {"survival_probability": round(prob, 4), "survived": int(prob >= 0.5), **MODEL_INFO}
