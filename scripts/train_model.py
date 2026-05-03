import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import ChurnMLP
from src.preprocessing import train_and_save_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def train_model() -> None:
    """Carrega dados, constrói preprocessador, treina e salva artefatos."""
    logger.info("Iniciando treinamento do modelo...")

    # 1. Treinar e salvar preprocessador
    (
        X_train_processed,
        X_val_processed,
        y_train,
        y_val,
        preprocessor,
    ) = train_and_save_preprocessor(output_path=MODELS_DIR / "preprocessor.joblib")

    # 2. Preparar dados para PyTorch
    logger.info("Preparando tensores para PyTorch...")
    X_train_tensor = torch.FloatTensor(X_train_processed)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val_processed)
    y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. Treinar modelo (poucas épocas para CI)
    logger.info("Treinando modelo MLP...")
    model = ChurnMLP(input_dim=X_train_processed.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 3  # Poucas épocas para ser rápido
    device = torch.device("cpu")
    model.to(device)

    for epoch in range(num_epochs):
        # Treino
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f",
            epoch + 1,
            num_epochs,
            avg_train_loss,
            avg_val_loss,
        )

    # 4. Salvar modelo
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "mlp_churn.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Modelo salvo em %s", model_path)

    logger.info("Treinamento concluído!")


if __name__ == "__main__":
    train_model()
