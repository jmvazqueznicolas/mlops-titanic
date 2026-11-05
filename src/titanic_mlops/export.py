"""Copy the current @champion out of the registry so it can be baked into a Docker image.

    python -m titanic_mlops.export      # -> model_export/model + model_export/VERSION.json

The image then contains one fixed model version: what is tested is exactly what runs.
"""
import json
import shutil

import mlflow
from mlflow import MlflowClient

from titanic_mlops.config import CHAMPION_ALIAS, MODEL_NAME, ROOT


def main():
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    params = client.get_run(mv.run_id).data.params
    dest = ROOT / "model_export"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir()
    mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{mv.version}", dst_path=str(dest / "model"))
    info = {"model_name": MODEL_NAME, "model_version": int(mv.version), "run_id": mv.run_id,
            "model": params.get("model"), "data_sha256": params.get("data_sha256"), "git_commit": params.get("git_commit")}
    (dest / "VERSION.json").write_text(json.dumps(info, indent=2))
    print(f"exported {MODEL_NAME} v{mv.version} ({info['model']}) -> model_export/")


if __name__ == "__main__":
    main()
