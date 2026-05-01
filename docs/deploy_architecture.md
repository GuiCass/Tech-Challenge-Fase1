# Arquitetura de Deploy — ChurnMLP

## 1. Decisão: Real-time vs Batch

### Comparação

| Critério | Inferência em Tempo Real | Inferência em Batch |
|---|---|---|
| **Latência** | Milissegundos (< 200 ms) | Minutos a horas |
| **Trigger** | Requisição HTTP individual | Job agendado (diário/semanal) |
| **Integração** | API REST — qualquer sistema | Arquivo CSV / banco de dados |
| **Custo de infra** | Mais alto (serviço sempre ativo) | Mais baixo (executa e para) |
| **Escalabilidade** | Horizontal (containers) | Vertical / paralelismo de job |
| **Uso no CRM** | Score na hora da consulta | Score pré-calculado em lote |
| **Freshness dos dados** | Feature calculada no momento | Dados do último batch |

### Decisão adotada: **Real-time (API REST)**

**Justificativa:**

1. **Integração com CRM e sistemas internos:** as equipes de retenção e atendimento precisam consultar o score de churn de um cliente específico no momento do atendimento ou da tomada de decisão — batch não atenderia esse requisito de imediato.

2. **SLO de latência definido:** o projeto estabelece SLO de latência ≤ 500 ms (desejado ≤ 200 ms), o que é compatível com inferência em tempo real via FastAPI + MLP PyTorch em CPU.

3. **Volume de dados gerenciável:** o dataset tem ~7k clientes. Mesmo em produção, uma operadora de médio porte não geraria volume de requisições que inviabilizasse real-time.

4. **Flexibilidade para futuras integrações:** uma API REST pode ser consumida por qualquer sistema (CRM, app mobile, dashboard) sem necessidade de reprocessamento.

> **Nota:** para operadoras com bases de milhões de clientes e campanhas executadas diariamente em lote, uma arquitetura **híbrida** seria recomendada: batch overnight para gerar scores de toda a base + API real-time para consultas pontuais.

---

## 2. Arquitetura Escolhida: AWS ECS Fargate

```
                        Internet
                           │
                    ┌──────▼──────┐
                    │  AWS ALB    │  Application Load Balancer
                    │  (HTTPS)    │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │    AWS ECS Fargate      │
              │  ┌──────────────────┐   │
              │  │  Container       │   │
              │  │  ChurnMLP API    │   │
              │  │  FastAPI + uvicorn│  │
              │  │  porta 8000      │   │
              │  └──────────────────┘   │
              │  (auto-scaling)         │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
  ┌──────▼──────┐  ┌───────▼──────┐  ┌──────▼──────┐
  │  Amazon ECR │  │  Amazon S3   │  │  CloudWatch │
  │  (imagens   │  │  (artefatos: │  │  (logs +    │
  │   Docker)   │  │  modelo +    │  │   métricas) │
  └─────────────┘  │  preprocessor│  └─────────────┘
                   └──────────────┘
```

---

## 3. Componentes da Arquitetura

| Componente | Serviço AWS | Função |
|---|---|---|
| Container registry | Amazon ECR | Armazena imagens Docker versionadas |
| Orquestração | Amazon ECS Fargate | Executa containers sem gerenciar servidores |
| Load balancer | AWS ALB | Distribui tráfego e termina TLS |
| Artefatos ML | Amazon S3 | Armazena `mlp_churn.pt` e `preprocessor.joblib` |
| Logs | Amazon CloudWatch Logs | Centraliza logs JSON da API |
| Métricas | Amazon CloudWatch Metrics | Latência, CPU, erros, requisições |
| Alertas | AWS CloudWatch Alarms + SNS | Notificações por e-mail/Slack |
| CI/CD | GitHub Actions | Build, push ECR, deploy ECS automático |

---

## 4. Fluxo de Deploy

```
Desenvolvedor faz push para main
          │
          ▼
  [GitHub Actions — CI]
  1. Checkout
  2. Instala dependências
  3. ruff check (lint)
  4. pytest (testes)
          │ sucesso
          ▼
  [GitHub Actions — CD]
  5. Login AWS (OIDC ou secrets)
  6. docker build
  7. docker push → Amazon ECR
  8. Atualiza ECS task definition
  9. ECS rolling deploy (zero downtime)
 10. Aguarda estabilidade do serviço
```

---

## 5. Configuração do Container (Fargate)

| Parâmetro | Valor recomendado |
|---|---|
| CPU | 512 vCPU (0.5 vCPU) |
| Memória | 1024 MB (1 GB) |
| Réplicas mínimas | 1 |
| Réplicas máximas | 4 (auto-scaling por CPU > 70%) |
| Health check | `GET /health` a cada 30s |
| Porta | 8000 |

---

## 6. Variáveis de Ambiente em Produção

| Variável | Valor | Fonte |
|---|---|---|
| `MODEL_PATH` | `/app/models/mlp_churn.pt` | ECS task definition |
| `PREPROCESSOR_PATH` | `/app/models/preprocessor.joblib` | ECS task definition |
| `CHURN_THRESHOLD` | `0.5` (ajustável) | ECS task definition |
| `INPUT_DIM` | `45` | ECS task definition |

> Em arquiteturas mais avançadas, `MODEL_PATH` e `PREPROCESSOR_PATH` apontariam para um S3 URI e o container faria download no startup via boto3.

---

## 7. Estratégia de Deploy

**Rolling deployment** (padrão ECS):
- Nova versão sobe gradualmente
- Health checks validam antes de derrubar versão antiga
- Zero downtime

**Rollback:**
- Revert do push no git → CI/CD re-deploya versão anterior automaticamente
- Ou: atualizar manualmente a ECS service para task definition anterior via AWS Console

---

## 8. Estimativa de Custo (referência)

| Recurso | Custo estimado |
|---|---|
| ECS Fargate (0.5 vCPU / 1GB, 24/7) | ~$15–20 USD/mês |
| ECR (armazenamento de imagens) | ~$1 USD/mês |
| ALB | ~$16 USD/mês |
| CloudWatch Logs (1 GB/mês) | ~$0,50 USD/mês |
| **Total estimado** | **~$33–38 USD/mês** |

> Valores aproximados baseados na região us-east-1 (mai/2025). Para fins acadêmicos, AWS Free Tier cobre parte desses recursos no primeiro ano.
