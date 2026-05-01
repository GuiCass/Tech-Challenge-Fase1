# Tech Challenge — Fase 1: Previsão de Churn em Telecomunicações

Pipeline profissional end-to-end de Machine Learning para previsão de churn de clientes de uma operadora de telecomunicações. O modelo principal é uma rede neural **MLP treinada com PyTorch**, comparada com baselines Scikit-Learn e rastreada com MLflow. A inferência é servida via **FastAPI**.

---

## Sumário

- [Contexto](#contexto)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Execução passo a passo](#execução-passo-a-passo)
- [API de Inferência](#api-de-inferência)
- [Testes](#testes)
- [MLflow](#mlflow)
- [CI/CD](#cicd)
- [Documentação](#documentação)

---

## Contexto

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. O objetivo é identificar clientes com alto risco de cancelamento para priorizar ações de retenção.

**Dataset:** [Telco Customer Churn — IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
**Problema:** Classificação binária — `1` churn, `0` não churn
**Modelo principal:** MLP (PyTorch) — arquitetura 45 → 128 → 64 → 32 → 1

---

## Estrutura do Projeto

```
Tech-Challenge-Fase1/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Pipeline de CI (lint + testes)
│       └── cd.yml              # Pipeline de CD (deploy AWS ECS)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (não versionado)
├── docs/
│   ├── ml_canvas.md            # ML Canvas com stakeholders e SLOs
│   ├── model_card.md           # Model Card com performance e limitações
│   ├── monitoring_plan.md      # Plano de monitoramento e playbook
│   └── deploy_architecture.md  # Arquitetura de deploy (AWS ECS Fargate)
├── models/                     # Artefatos exportados (não versionados)
│   ├── mlp_churn.pt            # State dict do modelo PyTorch
│   └── preprocessor.joblib     # ColumnTransformer ajustado
├── notebooks/
│   ├── EDA.ipynb               # Análise exploratória de dados
│   ├── baselines.ipynb         # Modelos baseline com MLflow
│   └── models_and_mlp.ipynb    # MLP PyTorch com MLflow
├── scripts/
│   ├── export_artifacts.py     # Exporta modelo e preprocessor do MLflow
│   └── open_mlflow_ui.py       # Abre MLflow UI local
├── src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI — /health e /predict
│   ├── model.py                # Definição da classe ChurnMLP
│   └── preprocessing.py        # Pipeline de pré-processamento
├── tests/
│   ├── test_smoke.py           # Smoke tests (imports e instâncias)
│   ├── test_schema.py          # Schema tests com pandera
│   └── test_api.py             # Testes dos endpoints da API
├── Dockerfile                  # Imagem Docker para deploy
├── Makefile                    # Comandos: install, lint, test, run-api
└── pyproject.toml              # Dependências, ruff e pytest
```

---

## Requisitos

- Python 3.10+
- pip

---

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd Tech-Challenge-Fase1

# 2. Instale as dependências
pip install -e ".[dev,notebook]"
```

> **Windows:** use `python -m pip install -e ".[dev,notebook]"`

---

## Execução passo a passo

### 1. Obter o dataset

O dataset é baixado automaticamente via `kagglehub` ao rodar os notebooks. Alternativamente, copie manualmente o arquivo para `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### 2. Rodar a EDA

Abra e execute o notebook:
```
notebooks/EDA.ipynb
```

### 3. Treinar os baselines

```
notebooks/baselines.ipynb
```
Registra DummyClassifier e Regressão Logística no MLflow.

### 4. Treinar o MLP

```
notebooks/models_and_mlp.ipynb
```
Treina a rede neural PyTorch com early stopping e registra todos os artefatos no MLflow.

### 5. Exportar artefatos do MLflow para `models/`

```bash
python scripts/export_artifacts.py
```

Isso gera `models/mlp_churn.pt` e `models/preprocessor.joblib`, necessários para a API.

### 6. Subir a API

```bash
# Com Makefile
make run-api

# Ou diretamente
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Acesse:
- **Swagger UI:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

---

## API de Inferência

### `GET /health`

Verifica se o modelo está carregado e a API está operacional.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model": "ChurnMLP", "threshold": 0.5}
```

### `POST /predict`

Recebe os dados de um cliente e retorna a probabilidade de churn.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "PaymentMethod": "Electronic check"
  }'
```

```json
{
  "churn_probability": 0.7821,
  "churn_prediction": true,
  "threshold_used": 0.5
}
```

**Threshold configurável** via variável de ambiente:
```bash
CHURN_THRESHOLD=0.4 python -m uvicorn src.api:app --port 8000
```

**Logs da API** em formato JSON: `logs/api.log`

---

## Testes

```bash
# Com Makefile
make test

# Ou diretamente
python -m pytest tests/ -v --tb=short
```

| Arquivo | Tipo | O que testa |
|---|---|---|
| `test_smoke.py` | Smoke | Imports, instância do ChurnMLP, constantes |
| `test_schema.py` | Schema | Validação de dados de entrada com pandera |
| `test_api.py` | API | Endpoints /health e /predict (modelo mockado) |

---

## MLflow

Visualize todos os experimentos, métricas e artefatos:

```bash
# Com Makefile
make mlflow

# Ou diretamente
python -m mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --port 5000
```

Acesse: http://localhost:5000

**Experimento:** `churn-model-comparison`
**Runs registrados:** DummyClassifier, Regressão Logística, ChurnMLP

---

## Lint

```bash
# Verificar
make lint

# Corrigir automaticamente
make format
```

---

## CI/CD

| Workflow | Trigger | O que faz |
|---|---|---|
| `ci.yml` | Push / PR para `main` | Lint (ruff) + testes (pytest) |
| `cd.yml` | CI com sucesso | Build Docker → push ECR → deploy ECS Fargate |

Para o deploy AWS funcionar, configure os seguintes secrets no GitHub:
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY`, `ECS_CLUSTER`, `ECS_SERVICE`, `ECS_TASK_DEFINITION`, `CONTAINER_NAME`

---

## Documentação

| Documento | Descrição |
|---|---|
| [docs/ml_canvas.md](docs/ml_canvas.md) | Problema de negócio, stakeholders, métricas e SLOs |
| [docs/model_card.md](docs/model_card.md) | Performance, limitações, vieses e cenários de falha |
| [docs/monitoring_plan.md](docs/monitoring_plan.md) | Métricas, alertas e playbook de incidentes |
| [docs/deploy_architecture.md](docs/deploy_architecture.md) | Decisão real-time vs batch e arquitetura AWS |

---

## Bibliotecas Principais

| Biblioteca | Uso |
|---|---|
| PyTorch | Construção e treinamento da rede neural MLP |
| Scikit-Learn | Pré-processamento e modelos baseline |
| MLflow | Rastreamento de experimentos e artefatos |
| FastAPI | API de inferência REST |
| Pydantic | Validação dos dados de entrada |
| Pandera | Validação de schema nos testes |
| Ruff | Linting e formatação de código |
| Pytest | Testes automatizados |
