"""Schema tests: valida o schema dos dados de entrada com pandera."""

import pandas as pd
import pandera as pa
import pytest
from pandera import Column, DataFrameSchema

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SCHEMA_ENTRADA = DataFrameSchema(
    {
        "tenure": Column(int, pa.Check.ge(0)),
        "MonthlyCharges": Column(float, pa.Check.ge(0)),
        "TotalCharges": Column(float, pa.Check.ge(0)),
        "SeniorCitizen": Column(int, pa.Check.isin([0, 1])),
        "gender": Column(str, pa.Check.isin(["Male", "Female"])),
        "Partner": Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents": Column(str, pa.Check.isin(["Yes", "No"])),
        "PhoneService": Column(str, pa.Check.isin(["Yes", "No"])),
        "MultipleLines": Column(str, pa.Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"])
        ),
        "OnlineBackup": Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"])
        ),
        "DeviceProtection": Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"])
        ),
        "TechSupport": Column(str, pa.Check.isin(["Yes", "No", "No internet service"])),
        "StreamingTV": Column(str, pa.Check.isin(["Yes", "No", "No internet service"])),
        "StreamingMovies": Column(
            str, pa.Check.isin(["Yes", "No", "No internet service"])
        ),
        "Contract": Column(
            str, pa.Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": Column(str, pa.Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(
            str,
            pa.Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
    }
)

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


def test_schema_aceita_dado_valido():
    df = pd.DataFrame([CLIENTE_VALIDO])
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    validado = SCHEMA_ENTRADA.validate(df)
    assert len(validado) == 1


def test_schema_rejeita_tenure_negativo():
    dado_invalido = {**CLIENTE_VALIDO, "tenure": -1}
    df = pd.DataFrame([dado_invalido])
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    with pytest.raises(pa.errors.SchemaError):
        SCHEMA_ENTRADA.validate(df)


def test_schema_rejeita_contrato_invalido():
    dado_invalido = {**CLIENTE_VALIDO, "Contract": "Contrato Invalido"}
    df = pd.DataFrame([dado_invalido])
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    with pytest.raises(pa.errors.SchemaError):
        SCHEMA_ENTRADA.validate(df)


def test_features_numericas_e_categoricas_sem_sobreposicao():
    sobreposicao = set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert sobreposicao == set()


def test_todas_colunas_schema_cobrem_features():
    colunas_schema = set(SCHEMA_ENTRADA.columns.keys())
    features_esperadas = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert features_esperadas == colunas_schema
