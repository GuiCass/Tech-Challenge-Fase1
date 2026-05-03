"""Pipeline de pré-processamento para o dataset Telco Customer Churn."""

import glob
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
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
    """Carrega e limpa o dataset Telco Customer Churn do Kaggle.

    Aplica as mesmas transformações definidas na EDA:
    - Converte TotalCharges de object para float
    - Imputa valores ausentes de TotalCharges com a mediana
    - Renomeia Churn para target e codifica como binário (0/1)
    """
    import kagglehub

    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum arquivo CSV encontrado no caminho do dataset Kaggle: {path}"
        )
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
    """Extrai a matriz de features X e o vetor alvo y do dataframe."""
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
    """Retorna um ColumnTransformer para o conjunto de features do Telco Churn.

    Aplica StandardScaler nas features numéricas e OneHotEncoder nas
    categóricas. Seguro para ser instanciado múltiplas vezes.
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


def train_and_save_preprocessor(
    output_path: Path | str = "models/preprocessor.joblib",
) -> tuple:
    """Carrega dados, treina e salva o preprocessador.

    Args:
        output_path: Caminho onde salvar o preprocessador.joblib

    Returns:
        Tupla (X_train_processed, X_val_processed, y_train, y_val, preprocessor)
    """
    logger.info("Iniciando preparação do preprocessador...")

    # Carregar e preparar dados
    df = load_and_clean_data()
    X, y = prepare_features(df)

    # Dividir dados
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Dados divididos: treino=%d, validação=%d", len(X_train), len(X_val))

    # Construir e ajustar preprocessador
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Salvar preprocessador
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(preprocessor, output_path)
    logger.info("Preprocessador salvo em %s", output_path)

    return X_train_processed, X_val_processed, y_train, y_val, preprocessor
