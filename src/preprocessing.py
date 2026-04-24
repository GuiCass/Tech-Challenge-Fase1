"""Preprocessing pipeline for Telco Customer Churn dataset."""

import glob
import logging
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET_COLUMN = "target"
DROP_COLUMNS = ["customerID"]


def load_and_clean_data() -> pd.DataFrame:
    """Load and clean the Telco Customer Churn dataset from Kaggle.

    Applies the same transformations established in the EDA:
    - Convert TotalCharges from object to float
    - Impute missing TotalCharges with median
    - Rename Churn to target and encode as binary (0/1)
    """
    import kagglehub

    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in Kaggle dataset path: {path}")
    df = pd.read_csv(csv_files[0])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_count = int(df["TotalCharges"].isna().sum())
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)

    df.rename(columns={"Churn": TARGET_COLUMN}, inplace=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    logger.info(
        "Dataset carregado: %d linhas, %d colunas. %d valores imputados em TotalCharges.",
        df.shape[0],
        df.shape[1],
        missing_count,
    )
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Extract feature matrix X and target vector y from the dataframe."""
    X = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    class_dist = y.value_counts().to_dict()
    logger.info(
        "Features: %s | Distribuicao alvo: 0=%d (%.1f%%) 1=%d (%.1f%%)",
        X.shape,
        class_dist.get(0, 0),
        100 * class_dist.get(0, 0) / len(y),
        class_dist.get(1, 0),
        100 * class_dist.get(1, 0) / len(y),
    )
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer for the Telco Churn feature set.

    Applies StandardScaler to numeric features and OneHotEncoder to
    categorical features. Safe to fit multiple times (returns a new instance).
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    logger.info(
        "Preprocessor configurado: %d features numericas, %d features categoricas.",
        len(NUMERIC_FEATURES),
        len(CATEGORICAL_FEATURES),
    )
    return preprocessor
