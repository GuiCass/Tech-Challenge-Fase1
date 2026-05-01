# Model Card — ChurnMLP

## 1. Descrição do Modelo

| Campo | Valor |
|---|---|
| **Nome** | ChurnMLP |
| **Versão** | 1.0.0 |
| **Tipo** | Classificação binária (churn / não churn) |
| **Arquitetura** | MLP — 45 → 128 → 64 → 32 → 1 |
| **Framework** | PyTorch 2.x |
| **Data de treinamento** | 2026 |
| **Responsáveis** | Guilherme Cassiano RM371741 - Felipe Quiroga RM372563 -  |

---

## 2. Uso Pretendido

### Para quem é este modelo

Equipes de Gestão da Empresa de Telecomunicações, retenção, marketing de relacionamento e customer success para a operadora de telco.

### Casos de uso suportados

- Priorização de clientes para campanhas de retenção proativa
- Score de risco de churn para ranqueamento de base
- Apoio à decisão de ofertas e descontos personalizados

### Casos de uso fora do escopo

- Substituição de avaliação humana em decisões críticas
- Predição de churn em setores fora de telecomunicações sem retreinamento
- Decisões automatizadas sem revisão humana sobre contratos ou cobranças

---

## 3. Dataset de Treinamento

| Campo | Valor |
|---|---|
| **Fonte** | Telco Customer Churn — IBM (Kaggle: blastchar/telco-customer-churn) |
| **Registros** | 7.043 clientes |
| **Features** | 19 (4 numéricas, 15 categóricas) |
| **Variável alvo** | `Churn` → `target` (0 = não cancelou, 1 = cancelou) |
| **Desbalanceamento** | ~26,5% positivos (churn) / ~73,5% negativos |
| **Divisão** | 80% treino / 20% teste (estratificado) |

### Features utilizadas

**Numéricas:** `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`

**Categóricas:** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

---

## 4. Pré-processamento

- `TotalCharges`: convertido para numérico, valores ausentes imputados pela mediana
- Features numéricas: `StandardScaler`
- Features categóricas: `OneHotEncoder` (handle_unknown="ignore")
- Dimensão final após encoding: **45 features**

---

## 5. Arquitetura e Treinamento

```
Entrada (45) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
             → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
             → Linear(32)  → ReLU
             → Linear(1)   → [logit]
```

| Hiperparâmetro | Valor |
|---|---|
| Loss | BCEWithLogitsLoss |
| Otimizador | Adam |
| Early stopping | Patience = 10 épocas |
| Seed | 42 (reprodutibilidade) |
| Threshold de classificação | 0.5 (configurável via env) |

---

## 6. Desempenho

### Comparação de modelos (conjunto de teste)

| Modelo | AUC-ROC | F1-Score | Precision | Recall |
|---|---|---|---|---|
| DummyClassifier | 0.50 | — | — | — |
| Regressão Logística | 0.84 | 0.61 | 0.67 | 0.56 |
| **ChurnMLP** | **≥ 0.82** | **≥ 0.65** | — | — |

> Métricas exatas disponíveis no MLflow: `sqlite:///notebooks/mlflow.db`, experimento `churn-model-comparison`.

### SLOs de qualidade

| Métrica | SLO mínimo | SLO desejado | Status |
|---|---|---|---|
| AUC-ROC | ≥ 0.75 | ≥ 0.82 | A verificar no MLflow |
| F1-Score | ≥ 0.55 | ≥ 0.65 | A verificar no MLflow |

---

## 7. Análise de Trade-off de Erros

| Tipo de erro | Impacto no negócio | Custo relativo |
|---|---|---|
| **Falso positivo** | Cliente recebe ação de retenção desnecessária (desconto, contato) | Baixo — custo operacional |
| **Falso negativo** | Cliente churn não identificado, sem ação preventiva | **Alto** — perda de receita |

**Decisão de threshold:** No contexto de churn, falsos negativos são mais caros. O threshold pode ser reduzido abaixo de 0.5 para aumentar recall, aceitando mais falsos positivos, conforme análise de custo-benefício da operadora.

---

## 8. Limitações

- **Dataset histórico e estático:** o modelo não capta mudanças de comportamento em tempo real.
- **Ausência de features temporais ricas:** sem dados de uso mensal, histórico de reclamações ou interações com atendimento.
- **7.043 registros:** base relativamente pequena para generalização ampla.
- **Uma única operadora:** padrões aprendidos podem não generalizar para outras empresas do setor.
- **Desbalanceamento de classes:** ~26,5% de churn pode dificultar recall em populações com taxas muito diferentes.
- **Features do dataset IBM:** variáveis como `SeniorCitizen` são binárias (0/1), não idade real, o que limita granularidade.

---

## 9. Vieses Conhecidos

| Viés | Descrição | Risco |
|---|---|---|
| **Viés de seleção** | Dataset de uma única operadora americana | Pode não representar clientes brasileiros ou de outros mercados |
| **Viés de gênero** | Feature `gender` presente no modelo | Pode gerar predições diferenciadas por gênero sem relação causal com churn |
| **Viés de senioridade** | `SeniorCitizen` como feature preditiva | Pode discriminar clientes idosos nas ações de retenção |
| **Viés temporal** | Dados sem data de referência clara | Não é possível garantir que o padrão aprendido ainda é válido |

**Recomendação:** Auditar regularmente as predições por subgrupos (gênero, senioridade, tipo de contrato) para detectar disparidade nas taxas de falso positivo/negativo.

---

## 10. Cenários de Falha

| Cenário | Causa provável | Impacto | Mitigação |
|---|---|---|---|
| AUC-ROC < 0.75 em produção | Data drift — perfil dos clientes mudou | Predições pouco confiáveis | Retreinar com dados recentes |
| Alta taxa de falsos negativos | Threshold muito alto ou distribuição diferente | Churners não identificados | Reduzir threshold; recalibrar |
| Erro 500 na API | Preprocessor ou modelo corrompido | Serviço indisponível | Restaurar artefatos do MLflow |
| Latência > 500 ms | Sobrecarga de requisições simultâneas | Degradação de SLO | Escalar horizontalmente no ECS |
| Colunas ausentes no payload | Integração incorreta do sistema chamador | Erro 422 Unprocessable Entity | Validação Pydantic retorna detalhes do campo ausente |

---

## 11. Plano de Retreinamento

- **Gatilho:** AUC-ROC em produção cai abaixo de 0.75 por 2 semanas consecutivas, ou F1-Score abaixo de 0.55.
- **Frequência sugerida:** trimestral ou após mudanças significativas de produto/mercado.
- **Processo:** coletar novos dados rotulados → re-executar pipeline de pré-processamento → treinar com mesmos hiperparâmetros → validar SLOs → exportar artefatos → deploy via CD.

---

## 12. Informações de Contato

Para dúvidas sobre o modelo, limitações ou solicitações de auditoria, contatar a equipe de Machine Learning responsável pelo projeto Tech Challenge — Fase 1.
