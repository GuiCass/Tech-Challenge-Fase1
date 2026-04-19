# ML Canvas — Projeto de Previsão de Churn em Telecom

## 1. Problema de Negócio

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado e precisa identificar, com antecedência, quais clientes possuem maior risco de cancelamento (churn), de forma a priorizar ações de retenção e reduzir perdas de receita. O desafio do projeto é construir um pipeline profissional end-to-end para prever churn, tendo como modelo principal uma rede neural MLP, comparada com baselines. :contentReference[oaicite:0]{index=0}

## 2. Objetivo do Modelo

Desenvolver um modelo de classificação binária capaz de estimar a probabilidade de churn por cliente, gerando um score de risco que apoie campanhas e ações de retenção.

## 3. Decisão que o Modelo Apoia

O modelo não toma a decisão final sozinho. Ele apoia decisões como:

- priorizar clientes para campanhas de retenção;
- selecionar clientes para ofertas, descontos ou contato proativo;
- orientar o time de CRM, atendimento e marketing sobre quais perfis exigem maior atenção.

## 4. Stakeholders

### Stakeholders de negócio

- Diretoria / liderança comercial
- Time de CRM / retenção
- Marketing de relacionamento
- Atendimento / customer success

### Stakeholders técnicos

- Time de dados / machine learning
- Engenharia / MLOps
- Time responsável pela API de inferência

### Stakeholders impactados indiretamente

- Clientes da operadora
- Área financeira, devido ao impacto em receita e margem

## 5. Variável Alvo

**Churn**  
Problema de classificação binária:

- `1`: cliente com churn
- `0`: cliente sem churn

## 6. Unidade de Predição

Cada previsão corresponde a **um cliente** da base.

## 7. Horizonte de Uso

O score de churn será utilizado para apoiar ações de retenção em janelas operacionais curtas, como campanhas semanais ou ciclos mensais de priorização.

## 8. Valor de Negócio Esperado

O valor gerado pelo modelo está em:

- reduzir churn;
- preservar receita;
- aumentar eficiência das campanhas de retenção;
- reduzir desperdício de esforço comercial com clientes de baixo risco.

---

# 9. Métrica de Negócio

## Métrica principal

**Custo líquido de churn evitado**

### Definição

Valor preservado com clientes corretamente priorizados para retenção, descontado o custo das ações aplicadas sobre os clientes classificados como risco de churn.

### Fórmula conceitual

**Valor líquido = benefício da retenção − custo da ação**

Uma formulação prática é:

**Valor líquido = (TP × taxa_de_sucesso_da_ação × valor_médio_do_cliente_perdido) − ((TP + FP) × custo_da_ação_de_retenção)**

Onde:

- `TP` = churners corretamente identificados;
- `FP` = clientes classificados como churn, mas que não cancelariam.

### Justificativa

Essa métrica traduz a performance do modelo para impacto real de negócio. Um modelo útil não é apenas o que acerta estatisticamente, mas o que ajuda a gerar valor líquido ao direcionar ações de retenção.

---

# 10. Métricas Técnicas

O projeto pede definição de métricas técnicas para avaliação e comparação de modelos, além da análise do trade-off entre falsos positivos e falsos negativos. :contentReference[oaicite:0]{index=0}

## Métricas

### AUC-ROC

A AUC-ROC foi escolhida como métrica técnica principal por medir a capacidade do modelo de separar clientes com churn e sem churn ao longo de diferentes thresholds. É uma métrica robusta para comparação global entre modelos e adequada para benchmarking inicial entre baselines e MLP.

### PR-AUC

A PR-AUC foi adicionada como métrica técnica complementar por avaliar a relação entre **precision** e **recall** com foco na classe positiva. Isso é especialmente relevante no problema de churn, em que o principal interesse está em identificar corretamente os clientes com maior risco de cancelamento.

### Recall

O Recall foi incluído como métrica complementar por medir a capacidade do modelo de identificar os churners reais. Isso é importante porque recall baixo significa deixar passar clientes que efetivamente cancelariam, reduzindo o potencial de ação preventiva da área de retenção.

### Precision

A Precision foi incluída como métrica complementar por medir a proporção de clientes classificados como churn que realmente apresentam churn. Isso é importante porque precision baixa significa direcionar ações de retenção para muitos clientes que não precisariam delas, aumentando custo e esforço operacional.

### F1-Score

O F1-Score complementa as demais métricas ao resumir o equilíbrio entre precision e recall em um único valor. Isso é relevante no problema de churn porque:

- recall baixo significa deixar passar clientes que iriam cancelar;
- precision baixa significa direcionar ações de retenção para muitos clientes que não precisariam delas.

## Assim, o F1 ajuda a avaliar a utilidade operacional do modelo.

# 11. SLOs (Service Level Objectives)

## SLOs de qualidade do modelo


| Métrica       | SLO mínimo    | SLO desejado                            | Justificativa                                                                                                                      |
| ------------- | ------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **AUC-ROC**   | **≥ 0.75**    | **≥ 0.82**                              | Garante capacidade mínima de separação entre churners e não churners, sendo a principal referência técnica de qualidade do modelo. |
| **F1-Score**  | **≥ 0.55**    | **≥ 0.65**                              | Garante equilíbrio mínimo entre precision e recall, importante para que a predição seja útil em ações de retenção.                 |
| **PR-AUC**    | **monitorar** | **maximizar**                           | Complementa a AUC-ROC ao focar no desempenho sobre a classe positiva, que é a mais relevante no problema de churn.                 |
| **Recall**    | **monitorar** | **maximizar com controle de precision** | Ajuda a reduzir falsos negativos, isto é, clientes que iriam cancelar e não foram identificados.                                   |
| **Precision** | **monitorar** | **manter em equilíbrio com recall**     | Ajuda a controlar falsos positivos e o custo de ações de retenção desnecessárias.                                                  |


## SLO de operação da API


| Métrica             | SLO mínimo   | SLO desejado | Justificativa                                                                                             |
| ------------------- | ------------ | ------------ | --------------------------------------------------------------------------------------------------------- |
| **Latência da API** | **≤ 500 ms** | **≤ 200 ms** | Mantém o serviço responsivo para integrações com aplicações internas, CRM ou fluxos de inferência online. |


### Justificativa geral dos SLOs

Esses SLOs são relevantes porque conectam duas dimensões importantes do projeto:

1. **Qualidade preditiva mínima aceitável**, para que o modelo tenha valor prático;
2. **Qualidade operacional mínima da API**, para que o modelo possa ser consumido de forma confiável em ambiente de uso.

---

# 12. Trade-off Entre Erros

## Falso positivo

O modelo prevê churn, mas o cliente não cancelaria.

### Impacto

- gasto desnecessário com descontos ou campanhas;
- esforço comercial e operacional desperdiçado;
- possível redução de margem.

## Falso negativo

O modelo prevê que o cliente não vai cancelar, mas ele cancela.

### Impacto

- perda de receita;
- perda de cliente;
- perda da oportunidade de retenção.

No contexto de churn, **falsos negativos tendem a ser mais caros que falsos positivos**, desde que o custo da ação de retenção seja controlado. Por isso, o projeto deve buscar bom poder de separação global (AUC-ROC) sem perder equilíbrio operacional entre precisão e cobertura (F1-Score).

---

# 13. Restrições e Premissas

## Restrições

- dataset tabular e histórico;
- ausência de informações temporais ricas;
- variáveis disponíveis podem não capturar todas as causas reais do churn;
- o modelo apoia a decisão, mas não substitui avaliação humana ou estratégia de negócio.

## Premissas

- a empresa consegue executar ações de retenção após a predição;
- existe custo associado a cada ação de retenção;
- o custo de perder um cliente é, em média, maior do que o custo de tentar retê-lo;
- o score gerado pelo modelo pode ser usado para ranquear clientes por risco.

---

# 14. Riscos do Projeto

- threshold mal calibrado gerar excesso de falsos positivos ou falsos negativos;
- drift de dados ao longo do tempo;
- viés em determinados perfis de clientes;
- dependência excessiva de padrões históricos que podem mudar no futuro.

---

# 15. Resumo Executivo

Este projeto busca prever churn de clientes de uma operadora de telecomunicações para apoiar ações de retenção. O modelo será avaliado principalmente por **AUC-ROC**, complementado por **F1-Score**, e seu valor de negócio será medido por **custo líquido de churn evitado**. Como metas de qualidade, o projeto adota **AUC-ROC mínimo de 0.75 e desejado de 0.82**, **F1-Score mínimo de 0.55 e desejado de 0.65**, além de **latência da API de até 500 ms no mínimo e 200 ms como meta desejada**. O desenvolvimento seguirá da formulação do problema e EDA para baselines, MLP, API e documentação final. :contentReference[oaicite:3]{index=3}