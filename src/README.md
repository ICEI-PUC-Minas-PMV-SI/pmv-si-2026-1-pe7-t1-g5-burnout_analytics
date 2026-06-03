# Burnout Analytics

## Descrição

Este projeto tem como objetivo analisar fatores relacionados ao risco de burnout em funcionários, utilizando técnicas de Análise Exploratória de Dados (EDA).

O dataset utilizado é sintético e contém informações sobre:

* características demográficas
* ambiente de trabalho
* hábitos de vida
* produtividade
* nível de estresse

A variável principal analisada é **`Burnout_Risk`**, que indica o risco de esgotamento profissional.

---

## Dataset

O dataset está incluído no link `[Work Productivity & Burnout Risk Dataset](https://www.kaggle.com/datasets/shree0910/work-productivity-and-burnout-risk-dataset)`.

---

## Objetivos

* Explorar a distribuição das variáveis
* Identificar relações entre variáveis e risco de burnout
* Avaliar correlações adequadas por tipo de dado
* Detectar padrões relevantes para saúde ocupacional
* Discutir implicações éticas do uso dos dados

---

## Estrutura do Projeto

```
src/
├── burnout-app/
│   ├── backend/
│   │   ├── app.py
│   │   ├── burnout_predicty.py
│   │   ├── burnout_model_xgboost.pkl
│   │   ├── burnout_model_random_forest.pkl
│   │   └── burnout_model_logistic_regression.pkl
│   └── frontend/
│       ├── index.html
│       └── src/
│           ├── main.js
│           └── style.css
├── Burnout_Analytics.ipynb
├── requirements.txt
└── README.md
```

---

## Como Executar

### 1. Clonar ou baixar o projeto

```bash
git clone https://github.com/ICEI-PUC-Minas-PMV-SI/pmv-si-2026-1-pe7-t1-g5-burnout_analytics.git burnout_analytics
cd burnout_analytics
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o notebook

Abra o arquivo:

```
Burnout_Analytics.ipynb
```

em:

* Jupyter Notebook
* VS Code
* Google Colab

Execute as células em ordem.

---

## Dependências

Principais bibliotecas utilizadas:

* pandas
* numpy
* matplotlib
* seaborn
* scipy
* kagglehub

---

## Metodologia

### 1. EDA (Análise Exploratória)

* Histogramas e boxplots
* Identificação de padrões e dispersão
* Análise de outliers

### 2. Relação com variável alvo

* Proporções para variáveis categóricas
* Boxplots para variáveis numéricas

### 3. Correlação

* Spearman → variáveis numéricas
* Point-biserial → variável binária vs numérica
* Qui-quadrado → variáveis categóricas

### 4. Visualizações

* Heatmaps
* Scatterplots com `hue=Burnout_Risk`

---

## Limitações

* Dataset sintético (não representa comportamento real)
* Possíveis correlações artificiais
* Não permite inferência causal

---

## Considerações Éticas

Este projeto envolve saúde ocupacional, o que exige cautela:

* Não utilizar resultados para punição de funcionários
* Evitar decisões automatizadas sem supervisão humana
* Monitorar possíveis vieses por grupo
* Utilizar a análise como ferramenta preventiva

---

## Arquitetura da Solução

A solução foi organizada em duas camadas principais:

**Frontend**

Responsável pela interface de interação com o usuário, permitindo o preenchimento dos dados utilizados na análise do risco de burnout.

Tecnologias utilizadas:

 * HTML5
 * CSS3
 * JavaScript

O frontend coleta as informações fornecidas pelo usuário e envia uma requisição para a API responsável pela execução do modelo.

**Backend**

Responsável pelo processamento das requisições e pela execução do modelo de Machine Learning.

Tecnologias utilizadas:

 * Python
 * Pandas
 * NumPy
 * Scikit-Learn
 * XGBoost, RandomForest, LogisticRegression
 * Flask

Fluxo de funcionamento:

 * O usuário preenche o formulário na aplicação web;
 * Os dados são enviados para a API;
 * O backend realiza o pré-processamento necessário;
 * O modelo previamente treinado é carregado;
 * A inferência é executada;
 * O resultado é retornado ao usuário.

## Empacotamento da Aplicação

Após o treinamento do modelo, os artefatos necessários para a realização das predições foram armazenados para reutilização em produção.

Arquivos utilizados:

 `burnout_model_xgboost.pkl`
 `burnout_model_random_forest.pkl`
 `burnout_model_logistic_regression.pkl`

Esses arquivos contêm:

 * Modelo treinado;
 * Pipeline de pré-processamento;
 * Configurações necessárias para a inferência.

Além dos artefatos do modelo, o pacote de implantação contém:

 * Código-fonte do backend;
 * Arquivos do frontend;

Durante a inicialização da aplicação, esses componentes são carregados para memória, permitindo a realização das predições sem necessidade de novo treinamento.

---

## Conclusão

A análise permite identificar padrões associados ao burnout, mas os resultados devem ser interpretados com cautela devido à natureza sintética dos dados.

---

## Histórico de versões

### [0.1.0] - 12/03/2026
#### Adicionado
- Implementação dos comandos para explorar os dados do dataset (desenvolvimento da etapa 2)

### [0.1.1] - 23/03/2026
#### Adicionado
- Ajustes dos comandos para exploração dos dados (desenvolvimento da etapa 2)

### [0.1.2] - 14/04/2026
#### Adicionado
- Implementação do modelo de Regressão Logística (desenvolvimento da etapa 3)

### [0.1.3] - 21/03/2026
#### Adicionado
- Ajustes dos comandos criados para exploração dos dados (relacionado a etapa 2)

### [0.1.4] - 27/04/2026
#### Adicionado
- Ajustes da implementação do modelo de Regressão Logística (desenvolvimento da etapa 3)

### [0.1.4] - 03/05/2026
#### Adicionado
- Implementação do modelo XGBoost e Randon Forest (desenvolvimento da etapa 4)

### [0.1.5] - 06/05/2026
#### Adicionado
- Ajustes da implementação do modelo XGBoost e Randon Forest (desenvolvimento da etapa 3)

### [0.1.6] - 07/05/2026
#### Adicionado
- Ajustes da implementação do modelo XGBoost e Randon Forest e comparação entre os três modelos (desenvolvimento da etapa 3)

### [0.1.7] - 20/05/2026
#### Adicionado
- Ajustes da implementação do modelo XGBoost e Randon Forest e comparação entre os três modelos (desenvolvimento da etapa 3)

### [0.1.8] - 31/05/2026
#### Adicionado
- Ajustes nas documentações (desenvolvimento da etapa 3)

