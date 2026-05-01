"""FastAPI serviço de inferencia para predição de churn em clientes de telecom."""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.model import ChurnMLP
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "api.log"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(_JsonFormatter())

    root.addHandler(console)
    root.addHandler(file_handler)


_setup_logging()
logger = logging.getLogger(__name__)

_state: dict = {}

THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib")
MODEL_PATH = os.getenv("MODEL_PATH", "models/mlp_churn.pt")
INPUT_DIM = int(os.getenv("INPUT_DIM", "45"))


def _load_artifacts() -> None:
    mlflow_uri = os.getenv("MLFLOW_MODEL_URI")
    if mlflow_uri:
        import mlflow

        logger.info("Carregando modelo do MLflow URI: %s", mlflow_uri)
        _state["model"] = mlflow.pytorch.load_model(mlflow_uri)
    elif os.path.exists(MODEL_PATH):
        logger.info("Carregando modelo do caminho local: %s", MODEL_PATH)
        model = ChurnMLP(input_dim=INPUT_DIM)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        _state["model"] = model
    else:
        raise RuntimeError(
            f"Nenhum modelo encontrado. Setando MLFLOW_MODEL_URI ou colocando um modelo em {MODEL_PATH}."
        )

    _state["model"].eval()

    if not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(
            f"Preprocessor não encontrado em {PREPROCESSOR_PATH}. "
            "Execute o notebook de treinamento e exporte o preprocessor primeiro."
        )
    logger.info("Carregando preprocessor de: %s", PREPROCESSOR_PATH)
    _state["preprocessor"] = joblib.load(PREPROCESSOR_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    logger.info("Modelo e preprocessor carregados. API pronta.")
    yield
    _state.clear()
    logger.info("API encerrada. Artefatos liberados.")


app = FastAPI(
    title="API de Predição de Churn para Telecom",
    description="Prediz a probabilidade de churn de clientes de telecom usando um MLP do PyTorch.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method=%s path=%s status=%d latency_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


class CustomerFeatures(BaseModel):
    tenure: Annotated[int, Field(ge=0, description="Meses como cliente")]
    MonthlyCharges: Annotated[float, Field(ge=0)]
    TotalCharges: Annotated[float, Field(ge=0)]
    SeniorCitizen: Annotated[int, Field(ge=0, le=1)]
    gender: Annotated[str, Field(examples=["Male", "Female"])]
    Partner: Annotated[str, Field(examples=["Yes", "No"])]
    Dependents: Annotated[str, Field(examples=["Yes", "No"])]
    PhoneService: Annotated[str, Field(examples=["Yes", "No"])]
    MultipleLines: Annotated[str, Field(examples=["Yes", "No", "No phone service"])]
    InternetService: Annotated[str, Field(examples=["DSL", "Fiber optic", "No"])]
    OnlineSecurity: Annotated[str, Field(examples=["Yes", "No", "No internet service"])]
    OnlineBackup: Annotated[str, Field(examples=["Yes", "No", "No internet service"])]
    DeviceProtection: Annotated[
        str, Field(examples=["Yes", "No", "No internet service"])
    ]
    TechSupport: Annotated[str, Field(examples=["Yes", "No", "No internet service"])]
    StreamingTV: Annotated[str, Field(examples=["Yes", "No", "No internet service"])]
    StreamingMovies: Annotated[
        str, Field(examples=["Yes", "No", "No internet service"])
    ]
    Contract: Annotated[str, Field(examples=["Month-to-month", "One year", "Two year"])]
    PaperlessBilling: Annotated[str, Field(examples=["Yes", "No"])]
    PaymentMethod: Annotated[
        str,
        Field(
            examples=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        ),
    ]


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold_used: float


@app.get("/health", summary="Verificação de saúde da API")
def health():
    model_loaded = "model" in _state and "preprocessor" in _state
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    return {"status": "ok", "model": "ChurnMLP", "threshold": THRESHOLD}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predição de churn para um cliente",
)
def predict(customer: CustomerFeatures):
    import pandas as pd

    try:
        row = customer.model_dump()
        df = pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        X = _state["preprocessor"].transform(df)
        tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            logit = _state["model"](tensor)
            prob = float(torch.sigmoid(logit).squeeze())

        logger.info("Predição: prob=%.4f churn=%s", prob, prob >= THRESHOLD)
        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_prediction=prob >= THRESHOLD,
            threshold_used=THRESHOLD,
        )
    except Exception as exc:
        logger.exception("Predição falhou: %s", exc)
        raise HTTPException(status_code=500, detail="Predição falhou") from exc
