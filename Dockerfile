# Serving image: the API plus ONE fixed model version, baked in at build time.
# Build after `make export`, so model_export/ holds the current @champion.
FROM python:3.12-slim

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml params.yaml ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

COPY model_export/ model_export/
ENV MODEL_URI=/app/model_export/model \
    MLFLOW_DISABLE_AGENT_HINT=1

EXPOSE 8000
CMD ["uvicorn", "titanic_mlops.serve:app", "--host", "0.0.0.0", "--port", "8000"]
