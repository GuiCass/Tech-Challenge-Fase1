"""Exporta o modelo treinado e o preprocessor do MLflow para a pasta models/.

Uso:
    python scripts/export_artifacts.py
    python scripts/export_artifacts.py --run-id <RUN_ID>

Lê do banco SQLite do MLflow gerado pelos notebooks de treinamento e
grava dois arquivos consumidos pelo serviço FastAPI:
    models/mlp_churn.pt          — state_dict do PyTorch
    models/preprocessor.joblib   — ColumnTransformer ajustado do sklearn
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import mlflow
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///notebooks/mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-model-comparison")
MODELS_DIR = Path("models")


def get_best_run(experiment_name: str) -> mlflow.entities.Run:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado no MLflow.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'mlp'",
        order_by=["metrics.val_roc_auc DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise ValueError(
            "Nenhum run MLP encontrado. Execute o notebook de treinamento primeiro."
        )

    return runs[0]


def export(run_id: str | None = None) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    MODELS_DIR.mkdir(exist_ok=True)

    if run_id:
        run = mlflow.get_run(run_id)
        logger.info("Usando run especificado: %s", run_id)
    else:
        run = get_best_run(EXPERIMENT_NAME)
        logger.info(
            "Melhor run MLP encontrado: %s (val_roc_auc=%.4f)",
            run.info.run_id,
            run.data.metrics.get("val_roc_auc", float("nan")),
        )

    run_id = run.info.run_id

    preprocessor_local = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="preprocessing/mlp_preprocessor.joblib",
    )
    dest_preprocessor = MODELS_DIR / "preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_local)
    joblib.dump(preprocessor, dest_preprocessor)
    logger.info("Preprocessor salvo em %s", dest_preprocessor)

    model_uri = f"runs:/{run_id}/model"
    pytorch_model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    dest_model = MODELS_DIR / "mlp_churn.pt"
    torch.save(pytorch_model.state_dict(), dest_model)
    logger.info("State_dict do modelo salvo em %s", dest_model)

    logger.info("Exportação concluída. Arquivos prontos para o serviço FastAPI.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exporta artefatos do MLflow para models/"
    )
    parser.add_argument(
        "--run-id", default=None, help="ID específico do run no MLflow (opcional)"
    )
    args = parser.parse_args()
    export(run_id=args.run_id)
