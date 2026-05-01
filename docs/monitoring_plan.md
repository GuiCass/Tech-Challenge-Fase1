# Plano de Monitoramento — ChurnMLP em Produção

## 1. Objetivos

Garantir que o modelo continue entregando predições confiáveis, dentro dos SLOs definidos, e que degradações sejam detectadas e tratadas rapidamente.

---

## 2. Camadas de Monitoramento

```
┌─────────────────────────────────────────────────┐
│  Camada 1 — Infraestrutura (API / ECS)          │
│  Camada 2 — Qualidade dos dados de entrada      │
│  Camada 3 — Performance do modelo               │
│  Camada 4 — Impacto de negócio                  │
└─────────────────────────────────────────────────┘
```

---

## 3. Métricas Monitoradas

### 3.1 Infraestrutura da API

| Métrica | SLO mínimo | SLO desejado | Alerta |
|---|---|---|---|
| Latência P95 | ≤ 500 ms | ≤ 200 ms | > 500 ms por 5 min |
| Latência P99 | ≤ 1000 ms | ≤ 400 ms | > 1000 ms |
| Taxa de erros 5xx | < 1% | < 0,1% | > 1% em 10 min |
| Taxa de erros 4xx | < 5% | < 2% | > 10% em 10 min |
| Disponibilidade | ≥ 99,5% | ≥ 99,9% | < 99% no dia |
| Requisições por minuto | — | — | Queda > 80% vs média |

### 3.2 Qualidade dos Dados de Entrada (Data Drift)

| Métrica | Método | Alerta |
|---|---|---|
| Distribuição de `tenure` | KS-test vs distribuição de treino | p-value < 0.05 |
| Distribuição de `MonthlyCharges` | KS-test | p-value < 0.05 |
| Proporção de contratos `Month-to-month` | Desvio % vs baseline | Desvio > 15% |
| Taxa de valores ausentes por coluna | Contagem | > 0% (não esperado) |
| Volume diário de predições | Contagem | Queda > 50% vs média semanal |

### 3.3 Performance do Modelo (Model Drift)

> Requer dados rotulados com lag (churn real confirmado após ~30–90 dias).

| Métrica | SLO mínimo | Alerta |
|---|---|---|
| AUC-ROC (janela mensal) | ≥ 0.75 | < 0.75 por 2 semanas |
| F1-Score (janela mensal) | ≥ 0.55 | < 0.55 por 2 semanas |
| Taxa de predições positivas | 20–35% esperado | < 10% ou > 50% |
| Distribuição do score de probabilidade | Semelhante ao treino | Shift > 10 pontos percentuais |

### 3.4 Impacto de Negócio

| Métrica | Descrição | Frequência |
|---|---|---|
| Taxa de conversão de campanhas de retenção | % de churners identificados que foram retidos | Mensal |
| Custo por retenção | Custo total de ações / churners evitados | Mensal |
| Receita preservada | Valor médio do cliente × churners evitados | Mensal |

---

## 4. Fontes de Dados para Monitoramento

| Fonte | O que contém |
|---|---|
| `logs/api.log` (JSON) | Latência, status HTTP, payloads de request |
| AWS CloudWatch | Métricas de CPU, memória, ECS health |
| AWS ALB Logs | Latência de rede, códigos de resposta |
| CRM / sistema interno | Labels reais de churn para avaliação do modelo |
| MLflow | Baseline de métricas do treino original |

---

## 5. Alertas e Canais

| Severidade | Condição | Canal | Tempo de resposta |
|---|---|---|---|
| 🔴 Crítico | API indisponível ou erro 5xx > 5% | PagerDuty / SMS | Imediato (< 15 min) |
| 🔴 Crítico | Latência P95 > 1000 ms por 10 min | PagerDuty | Imediato |
| 🟡 Alto | AUC-ROC < 0.75 por 2 semanas | Slack #ml-alerts | 24 horas |
| 🟡 Alto | Data drift detectado em feature crítica | Slack #ml-alerts | 24 horas |
| 🟢 Info | Volume de predições fora do padrão | Slack #ml-monitoring | 48 horas |
| 🟢 Info | Taxa de positivos fora do intervalo esperado | Slack #ml-monitoring | 48 horas |

---

## 6. Playbook de Resposta a Incidentes

### 6.1 API Indisponível (5xx crítico)

```
1. Verificar logs do ECS: aws ecs describe-services
2. Verificar health check: GET /health → espera {"status": "ok"}
3. Se modelo não carregado: re-executar export_artifacts.py e redeploy
4. Se container crashando: verificar logs/api.log e CloudWatch
5. Rollback: redirecionar tráfego para task definition anterior no ECS
6. Comunicar stakeholders se > 15 min de indisponibilidade
```

### 6.2 Degradação de Performance (AUC-ROC < 0.75)

```
1. Confirmar degradação com janela mínima de 2 semanas de dados rotulados
2. Verificar data drift: comparar distribuição atual vs baseline de treino
3. Avaliar se houve mudança de produto, campanha ou perfil de clientes
4. Se drift confirmado:
   a. Coletar novos dados rotulados
   b. Retreinar modelo com pipeline notebooks/models_and_mlp.ipynb
   c. Registrar novo run no MLflow
   d. Executar: python scripts/export_artifacts.py
   e. Fazer deploy via pipeline CD (push para main)
5. Comunicar equipe de negócio sobre período de degradação e ações tomadas
```

### 6.3 Data Drift em Feature Crítica

```
1. Identificar qual feature apresenta drift (KS-test p-value < 0.05)
2. Investigar causa: mudança de produto, erro de integração, sazonalidade
3. Se erro de integração: corrigir sistema chamador
4. Se mudança real de comportamento: avaliar retreinamento
5. Documentar no registro de incidentes
```

### 6.4 Latência Alta (P95 > 500 ms)

```
1. Verificar métricas de CPU/memória do container no CloudWatch
2. Se CPU > 80%: escalar horizontalmente o ECS service (aumentar desired count)
3. Verificar se houve pico de requisições simultâneas
4. Avaliar adicionar cache de preprocessor (já em memória, verificar leak)
5. Se persistir: perfilar com py-spy ou adicionar rastreamento de tempo por etapa
```

---

## 7. Frequência de Revisão

| Atividade | Frequência |
|---|---|
| Revisão de alertas ativos | Diária |
| Revisão de métricas de infraestrutura | Semanal |
| Avaliação de data drift | Quinzenal |
| Avaliação de model drift (com labels) | Mensal |
| Revisão completa do plano de monitoramento | Trimestral |
| Avaliação de retreinamento | Trimestral ou sob demanda |

---

## 8. Ferramentas Recomendadas

| Camada | Ferramenta |
|---|---|
| Logs da API | Arquivo `logs/api.log` (JSON) + AWS CloudWatch Logs |
| Métricas de infraestrutura | AWS CloudWatch Metrics |
| Data drift | Evidently AI ou Great Expectations |
| Model drift | MLflow + scripts de avaliação periódica |
| Alertas | AWS CloudWatch Alarms + SNS + Slack webhook |
| Dashboard | AWS CloudWatch Dashboard ou Grafana |
