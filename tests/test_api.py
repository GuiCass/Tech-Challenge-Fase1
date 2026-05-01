"""Testes da API: valida os endpoints /health e /predict."""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.model import ChurnMLP

CLIENTE_VALIDO = {
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "SeniorCitizen": 0,
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


class _PreprocessorFake:
    """Preprocessor falso que retorna um vetor de zeros com 45 features."""

    def transform(self, df):
        return np.zeros((len(df), 45), dtype=np.float32)


def _carregar_artefatos_fake():
    """Injeta modelo e preprocessor falsos no estado da API."""
    import src.api as api_module

    modelo = ChurnMLP(input_dim=45)
    modelo.eval()
    api_module._state["model"] = modelo
    api_module._state["preprocessor"] = _PreprocessorFake()


@pytest.fixture
def client():
    with patch("src.api._load_artifacts", side_effect=_carregar_artefatos_fake):
        from src.api import app

        with TestClient(app) as c:
            yield c


def test_health_retorna_200(client):
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["model"] == "ChurnMLP"


def test_predict_retorna_200_com_dado_valido(client):
    resposta = client.post("/predict", json=CLIENTE_VALIDO)
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert "churn_probability" in corpo
    assert "churn_prediction" in corpo
    assert "threshold_used" in corpo
    assert 0.0 <= corpo["churn_probability"] <= 1.0
    assert isinstance(corpo["churn_prediction"], bool)


def test_predict_retorna_422_sem_campo_obrigatorio(client):
    dado_incompleto = {k: v for k, v in CLIENTE_VALIDO.items() if k != "tenure"}
    resposta = client.post("/predict", json=dado_incompleto)
    assert resposta.status_code == 422


def test_predict_retorna_422_com_tenure_negativo(client):
    dado_invalido = {**CLIENTE_VALIDO, "tenure": -5}
    resposta = client.post("/predict", json=dado_invalido)
    assert resposta.status_code == 422


def test_health_retorna_503_sem_modelo():
    import src.api as api_module

    def _nao_carrega():
        pass

    with patch("src.api._load_artifacts", side_effect=_nao_carrega):
        from src.api import app

        api_module._state.clear()
        with TestClient(app, raise_server_exceptions=False) as c:
            resposta = c.get("/health")
            assert resposta.status_code == 503
