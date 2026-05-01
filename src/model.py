import torch.nn as nn


class ChurnMLP(nn.Module):
    """Classificador MLP binário para predição de churn em telecomunicações.

    Arquitetura: input_dim → 128 → 64 → 32 → 1
    A saída é um logit bruto; aplique sigmoid para obter probabilidade.
    """

    def __init__(self, input_dim: int = 45) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)
