"""Smoke tests: verifica que os módulos principais importam e instanciam corretamente."""

import torch

from src.model import ChurnMLP
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
)


def test_churn_mlp_instancia():
    modelo = ChurnMLP(input_dim=45)
    assert isinstance(modelo, ChurnMLP)


def test_churn_mlp_forward_shape():
    modelo = ChurnMLP(input_dim=45)
    modelo.eval()
    entrada = torch.zeros(4, 45)
    with torch.no_grad():
        saida = modelo(entrada)
    assert saida.shape == (4, 1)


def test_preprocessing_constantes_definidas():
    assert len(NUMERIC_FEATURES) == 4
    assert len(CATEGORICAL_FEATURES) == 15
    assert TARGET_COLUMN == "target"


def test_build_preprocessor_retorna_column_transformer():
    from sklearn.compose import ColumnTransformer

    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_api_importa_sem_erros():
    import importlib

    modulo = importlib.import_module("src.api")
    assert hasattr(modulo, "app")
    assert hasattr(modulo, "predict")
    assert hasattr(modulo, "health")
