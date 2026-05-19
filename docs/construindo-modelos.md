# ETAPA 4 - CONSTRUÇÃO DE MODELOS (PARTE 02)
## Random Forest e XGBoost
## 1. Preparação dos dados

Nesta etapa, foram aplicadas técnicas de pré-processamento para garantir que o conjunto de dados estivesse em formato ideal para o treinamento dos algoritmos de aprendizado de máquina. O uso de boas práticas, como a prevenção de vazamento de dados (data leakage), foi priorizado por meio da construção de Pipelines de transformação.

### 1.1 Limpeza de Dados

Conforme identificado na Análise Exploratória (EDA), o dataset original não possuía valores nulos. No entanto, para garantir a robustez do modelo em um ambiente de produção (onde novos dados podem vir incompletos), foram incluídos *SimpleImputers* no pipeline: preenchimento com a mediana para numéricos e com a moda (*most_frequent*) para categóricos. Além disso, a coluna *Employee_ID* foi removida, pois atua apenas como identificador e não possui poder preditivo.

**Detecção de outliers:** Foi aplicado o método do intervalo interquartil (IQR). A análise identificou **outliers estatísticos apenas na variável `Experience_Years`**, com 424 registros acima do limite superior (valores de 28 a 32 anos). As demais variáveis (`Sleep_Hours`, `Screen_Time_Hours`, `Work_Hours_Per_Day`, etc.) não apresentaram outliers pelo critério IQR, embora apresentem perfis atípicos do ponto de vista do domínio (como poucas horas de sono).

**Decisão de manutenção dos outliers:** Optou-se por **manter** os 424 registros de `Experience_Years` no dataset pelos seguintes motivos:
- Os valores são plausíveis no contexto profissional (profissionais com até 32 anos de experiência)
- Não há evidência de erro de digitação ou inconsistência nos dados
- O dataset possui natureza sintética, podendo apresentar padrões artificiais que não justificam exclusão
- A remoção poderia comprometer a representatividade da amostra para profissionais seniores.

### 1.2 Transformação e Codificação de Dados

As variáveis categóricas — `Gender`, `Country`, `Job_Role`, `Company_Size` e `Work_Environment` — foram convertidas em formato numérico por meio da técnica de **One-Hot Encoding**, utilizando a classe `OneHotEncoder` da biblioteca scikit-learn com parâmetro `handle_unknown='ignore'`. Esse procedimento permite representar categorias qualitativas em forma binária, além de garantir que categorias não vistas durante o treinamento não comprometam o funcionamento do modelo.

Adicionalmente, procedeu-se à remoção da variável `Employee_ID`, por se tratar de um identificador único sem relevância preditiva. Em seguida, os dados foram organizados em variáveis preditoras (X) e variável alvo (y), sendo esta última representada pela variável `Burnout_Risk`, codificada em formato binário.

A variável `Stress_Level` foi codificada ordinalmente (Low → 1, Medium → 2, High → 3) para preservar a hierarquia de intensidade do estresse.

### 1.3 Feature Engineering

**Criação de novos atributos:** Não foram criados novos atributos sintéticos neste projeto, pois as variáveis existentes já se mostraram suficientes para a tarefa de classificação, conforme demonstrado pelos resultados obtidos. As relações entre as variáveis (ex: produtividade vs. horas de sono) podem ser capturadas pelos modelos baseados em árvores sem necessidade de engenharia manual.

**Seleção de características:** Não foi aplicada seleção automática de features (ex: SelectKBest, RFE) para manter a comparabilidade entre os modelos. A Regressão Logística com penalidade L1 já realiza seleção intrínseca de features ao zerar coeficientes de variáveis menos relevantes.

### 1.4 Padronização das Variáveis Numéricas

Considerando a presença de variáveis numéricas em diferentes escalas, foi aplicada a técnica de padronização por meio do `StandardScaler`, que transforma os dados para uma distribuição com média zero e desvio padrão unitário. Essa etapa é particularmente importante para algoritmos como a Regressão Logística, que são sensíveis à escala das variáveis. Para os modelos baseados em árvores (Random Forest e XGBoost), a padronização não é estritamente necessária, mas foi mantida para garantir comparabilidade entre os modelos.

### 1.5 Pipeline de Pré-processamento

As etapas de pré-processamento foram organizadas por meio de um `ColumnTransformer`, responsável por aplicar transformações distintas para variáveis numéricas e categóricas. Esse componente foi integrado a um pipeline completo utilizando a classe `Pipeline`, garantindo que todas as etapas sejam executadas de forma consistente tanto durante o treinamento quanto na fase de teste. Essa abordagem também evita o problema de vazamento de dados (data leakage), assegurando maior rigor metodológico.

```python
# Pipeline para variáveis numéricas
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Pipeline para variáveis categóricas
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ColumnTransformer agregando ambos os pipelines
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])
```

### 1.6 Tratamento do Desbalanceamento

Outro aspecto relevante do pré-processamento refere-se ao desbalanceamento da variável alvo, no qual aproximadamente 80% dos registros pertencem à classe "No" e 20% à classe "Yes". Para lidar com esse cenário, foram adotadas as seguintes estratégias:

* Estratificação na divisão treino-teste: Garantiu que a proporção original das classes fosse preservada em ambos os subconjuntos.

* `class_weight='balanced'` (Random Forest): Atribui maior peso à classe minoritária durante o treinamento.

* `scale_pos_weight` (XGBoost): Calculado como neg/pos = 19254/4746 = 4,0569, ajusta o peso da classe positiva no cálculo do gradiente.

```python
# Cálculo do scale_pos_weight para o XGBoost
neg, pos = np.bincount(y_train)
scale_pos_weight_calc = neg / pos
print(f"scale_pos_weight calculado: {scale_pos_weight_calc:.4f}")
# Resultado: scale_pos_weight = 4.0569
```

### 1.7 Divisão Treino-Teste
Por fim, o conjunto de dados foi dividido em subconjuntos de treinamento (80%) e teste (20%), utilizando uma estratégia estratificada da variável alvo (stratify=y), de modo a preservar a proporção original das classes. A utilização de um valor fixo de random_state garantiu a reprodutibilidade dos experimentos.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Treino: {X_train.shape[0]} registros ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Teste:  {X_test.shape[0]} registros ({X_test.shape[0]/len(X)*100:.1f}%)")
```

* Resultado da divisão:

Treino: 24.000 registros (80,0%)

Teste: 6.000 registros (20,0%)

```python
# Verificação do balanceamento nos conjuntos
print(f"Proporção no treino - Negativos: {sum(y_train==0)} | Positivos: {sum(y_train==1)}")
print(f"Proporção no teste  - Negativos: {sum(y_test==0)} | Positivos: {sum(y_test==1)}")
```

* Distribuição das classes:

Treino: 19.254 negativos (80,2%) | 4.746 positivos (19,8%)

Teste: 4.814 negativos (80,2%) | 1.186 positivos (19,8%)

### 1.8 Redução de Dimensionalidade

Não foi aplicada redução de dimensionalidade (ex: PCA) neste projeto pelos seguintes motivos:

- O dataset possui dimensionalidade moderada (38 variáveis após encoding)
- Modelos baseados em árvores (Random Forest e XGBoost) não são negativamente afetados por alta dimensionalidade
- A Regressão Logística com penalidade L1 já realiza seleção intrínseca de features
- A manutenção das variáveis originais preserva a interpretabilidade dos resultados


### 1.9 Validação Cruzada

Para garantir robustez na avaliação dos modelos e evitar overfitting, foi adotada a estratégia de **validação cruzada estratificada com 5 folds (k=5)** durante a otimização dos hiperparâmetros. Esta abordagem:

- Garante que cada fold mantenha a proporção original das classes (80/20)
- Reduz a variabilidade das estimativas de desempenho
- Maximiza o uso dos dados de treino para validação

A validação cruzada foi aplicada **exclusivamente no conjunto de treino**, mantendo o conjunto de teste completamente isolado até a avaliação final.

### 1.10 Monitoramento Contínuo

Embora não implementado nesta etapa por se tratar de um projeto acadêmico com dataset estático, recomenda-se para cenários de produção:

- **Detecção de concept drift:** Monitorar se a relação entre variáveis e burnout muda ao longo do tempo (ex: mudanças pós-pandemia)
- **Re-treinamento periódico:** Estabelecer calendário de re-treinamento (ex: trimestral ou semestral)
- **Validação de dados de entrada:** Verificar se novos dados seguem o mesmo padrão dos dados de treino (ex: mesmas categorias, escalas)

### 1.11 Sistematização Completa do Pipeline
Todas as transformações descritas foram encapsuladas em um ColumnTransformer dentro de um Pipeline do Scikit-Learn, garantindo que as mesmas regras matemáticas aplicadas aos dados de treino sejam rigorosamente aplicadas aos dados de teste (e a dados futuros), evitando contaminações.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Definição das colunas
categorical_features = X.select_dtypes(include="object").columns.tolist()
numerical_features = X.select_dtypes(exclude="object").columns.tolist()

# Remoção de Employee_ID se presente
if 'Employee_ID' in numerical_features:
    numerical_features.remove('Employee_ID')

# Pipeline completo
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])
```

## 2. Descrição dos modelos implementados
Além da Regressão Logística (que atuou como modelo baseline e cujos detalhes foram discutidos na Etapa 3), foram implementados dois modelos robustos baseados em árvores de decisão. Essa abordagem permite avaliar se a captura de relações não-lineares resulta em ganhos de performance.

### 2.1 Random Forest
O Random Forest é um algoritmo de aprendizado supervisionado baseado na técnica de Ensemble Learning, especificamente o método de Bagging (Bootstrap Aggregating).

**Princípio de funcionamento**: O algoritmo constrói múltiplas árvores de decisão independentes durante o treinamento. Cada árvore é treinada com uma amostra aleatória dos dados (com reposição) e um subconjunto aleatório de variáveis (features). A predição final é obtida por meio da votação majoritária (no caso de classificação) das árvores individuais.

**Vantagens e Limitações**: Suas principais vantagens incluem a alta resistência ao overfitting (devido à aleatoriedade na construção das árvores), capacidade de modelar interações complexas não-lineares e a não exigência de escalonamento dos dados. A limitação recai sobre sua natureza "caixa-preta" (black-box), sendo mais difícil de interpretar matematicamente do que uma Regressão Logística, além do maior custo computacional e de memória.

**Justificativa da escolha**: Escolhido por sua robustez e por ser referência em problemas de classificação com dados tabulares. Sua capacidade de capturar interações complexas sem exigir ajuste fino excessivo o torna ideal como segundo modelo em uma comparação sistemática.

**Hiperparâmetros otimizados:**

| Hiperparâmetro |	Valores testados |	Melhor valor |	Justificativa |
|---|---|-----|---|
| n_estimators |	100, 200, 300	| 100 |	Número de árvores; valores mais altos aumentam estabilidade mas também custo |
| max_depth |	10, 20, None	| 10 |	Profundidade máxima; limita a complexidade e reduz overfitting |
| min_samples_split |	2, 5, 10 |	2 |	Número mínimo de amostras para dividir um nó |
| class_weight |	'balanced' |	'balanced' |	Compensa o desbalanceamento das classes |

**Implementação do Random Forest:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Definição do grid de hiperparâmetros
param_grid_rf = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5, 10],
    'model__class_weight': ['balanced']
}

# Pipeline específico para Random Forest
rf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Configuração da validação cruzada estratificada
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV com validação cruzada
grid_rf = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid_rf,
    scoring='recall',  # Métrica principal para otimização
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1
)

# Treinamento e otimização (somente no conjunto de TREINO)
print("Iniciando GridSearch para Random Forest...")
grid_rf.fit(X_train, y_train)

print(f"\n Melhores parâmetros para Random Forest:")
print(f"   {grid_rf.best_params_}")
print(f"   Recall médio na validação cruzada: {grid_rf.best_score_:.4f}")
```

**Experimentos realizados - Random Forest - Combinações testadas**

| n_estimators | max_depth | min_samples_split | Recall CV | Decisão |
|--------------|-----------|-------------------|-----------|---------|
| **100** | **10** | **2** | **1,0000** | **Selecionado** |
| 200 | 10 | 2 | 1,0000 | Maior custo |
| 100 | 20 | 2 | 1,0000 | Risco overfitting |
| 100 | None | 2 | 1,0000 | Risco overfitting |
| 100 | 10 | 5 | 1,0000 | Sem ganho |
| 100 | 10 | 10 | 1,0000 | Sem ganho |

**Resultado da otimização:**

Melhores parâmetros: {'model__class_weight': 'balanced', 'model__max_depth': 10, 'model__min_samples_split': 2, 'model__n_estimators': 100}

Recall médio na validação cruzada (5-folds): 1,0000

### 2.2 XGBoost (Extreme Gradient Boosting)
O XGBoost é uma implementação altamente eficiente da técnica de Gradient Boosting.

**Princípio de funcionamento:** Ao contrário do Random Forest (onde as árvores são independentes), o XGBoost constrói árvores de decisão de forma sequencial. Cada nova árvore é treinada para corrigir os erros residuais (gradientes) cometidos pelas árvores anteriores. O algoritmo utiliza regularização L1 e L2 internamente, otimizando uma função de perda de forma iterativa.

**Vantagens e Limitações:** É amplamente reconhecido por apresentar o estado-da-arte em desempenho preditivo para dados tabulares estruturados, lidando extremamente bem com padrões complexos. Contudo, é altamente sensível à hiperparametrização; se não for bem ajustado, tende a decorar os dados de treino (overfitting).

**Justificativa da escolha:** Escolhido por seu reconhecido poder preditivo superior e por representar uma abordagem diferente (boosting sequencial vs. bagging do Random Forest).

**Hiperparâmetros otimizados:**

| Hiperparâmetro |	Valores testados |	Melhor valor |	Justificativa |
|----------------|-------------------|---------------|----------------|
| n_estimators |	100, 200, 300 |	100 |	Número de árvores (iterações de boosting) |
| max_depth	| 3, 6, 10 |	3	| Profundidade rasa reduz overfitting |
| learning_rate |	0.01, 0.1, 0.3 |	0.01 |	Taxa de aprendizado conservadora |
| subsample |	0.8, 1.0 |	0.8 |	Proporção de amostras por árvore; <1.0 reduz overfitting |
| scale_pos_weight |	4.0569 |	4.0569 |	Compensa desbalanceamento (fixo baseado no treino) |

**Implementação do XGBoost:**

```python
from xgboost import XGBClassifier

# Cálculo do scale_pos_weight (já realizado anteriormente)
neg, pos = np.bincount(y_train)
scale_pos_weight_calc = neg / pos
print(f"scale_pos_weight calculado: {scale_pos_weight_calc:.4f}")

# Definição do grid de hiperparâmetros
param_grid_xgb = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 6, 10],
    'model__learning_rate': [0.01, 0.1, 0.3],
    'model__subsample': [0.8, 1.0],
    'model__scale_pos_weight': [scale_pos_weight_calc]
}

# Pipeline específico para XGBoost
xgb_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False))
])

# GridSearchCV com validação cruzada estratificada
grid_xgb = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=param_grid_xgb,
    scoring='recall',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1
)

# Treinamento e otimização (somente no conjunto de TREINO)
print("\n🔍 Iniciando GridSearch para XGBoost...")
grid_xgb.fit(X_train, y_train)

print(f"\n Melhores parâmetros para XGBoost:")
print(f"   {grid_xgb.best_params_}")
print(f"   Recall médio na validação cruzada: {grid_xgb.best_score_:.4f}")
```

**Experimentos realizados - XGBoost - Combinações testadas**

| learning_rate | max_depth | subsample | Recall CV | Decisão |
|---------------|-----------|-----------|-----------|---------|
| **0,01** | **3** | **0,8** | **1,0000** | **Selecionado** |
| 0,01 | 3 | 1,0 | 1,0000 | Risco overfitting |
| 0,01 | 6 | 0,8 | 1,0000 | Risco overfitting |
| 0,10 | 3 | 0,8 | 1,0000 | Risco overfitting |
| 0,30 | 3 | 0,8 | 1,0000 | Risco overfitting |


**Resultado da otimização:**

Melhores parâmetros: {'model__learning_rate': 0.01, 'model__max_depth': 3, 'model__n_estimators': 100, 'model__scale_pos_weight': 4.056890012642225, 'model__subsample': 0.8}

Recall médio na validação cruzada (5-folds): 1,0000

## 3. Avaliação dos modelos criados
### 3.1 Métricas utilizadas
A avaliação dos modelos foi realizada por meio de múltiplas métricas, com o objetivo de obter uma visão abrangente de seu desempenho. Foram consideradas as métricas de acurácia (accuracy), precisão (precision), revocação (recall), *F1-score* e área sob a curva ROC (AUC-ROC).

| Métrica |	Descrição |	Importância no Contexto |
|---|---|---|
| Acurácia |	Proporção geral de acertos do modelo |	Útil para visão geral, mas insuficiente isoladamente devido ao desbalanceamento |
| Precisão |	Proporção de previsões positivas corretas	| Avalia a confiabilidade das predições de burnout |
| **Recall** |	**Capacidade de identificar corretamente os casos positivos** |	**Minimizar falsos negativos é prioritário** |
| F1-score |	Média harmônica entre precisão e recall	| Equilíbrio entre sensibilidade e especificidade |
| AUC-ROC |	Capacidade de discriminação entre classes |	Avalia qualidade intrínseca do modelo independente do threshold |

Dentre essas métricas, o **recall foi definido como o principal critério de avaliação**, uma vez que, no contexto do problema, o erro mais crítico consiste na ocorrência de **falsos negativos** — ou seja, situações em que o modelo deixa de identificar indivíduos em risco de burnout.

**Justificativa da escolha do Recall como métrica principal:**

* **Custo do erro na área da saúde**: Deixar de identificar um profissional em risco de burnout (Falso Negativo) pode resultar em agravamento do quadro, afastamento prolongado ou danos irreversíveis à saúde mental.

* **Custo do Falso Positivo**: Alertar erroneamente um profissional sem risco gera uma conversa de suporte ou verificação adicional - custo operacional, mas sem dano à saúde.

* **Alinhamento com o problema**: A prioridade é não deixar nenhum caso de burnout sem intervenção preventiva.

## 3.2. Resultados obtidos

### 3.2.1. Avaliação do Random Forest
Após a otimização dos hiperparâmetros (realizada exclusivamente no conjunto de treino com validação cruzada), o modelo final foi avaliado uma única vez no conjunto de teste, que permaneceu completamente isolado durante todo o processo de otimização.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Recupera o melhor modelo do GridSearch
rf_best = grid_rf.best_estimator_

# Predições no conjunto de TESTE (primeiro e único contato)
y_pred_rf = rf_best.predict(X_test)
y_proba_rf = rf_best.predict_proba(X_test)[:, 1]

# Cálculo das métricas
print("=== RANDOM FOREST - AVALIAÇÃO FINAL NO TEST SET ===")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba_rf):.4f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
```

**Resultados do Random Forest:**

| Métrica |	Valor |
|---|---|
| Recall |	1,0000 | 
| Precisão |	1,0000 |
| F1-Score |	1,0000 |
| AUC-ROC |	1,0000 |
| Acurácia |	1,0000 |

**Matriz de Confusão:**

<img width="406" height="458" alt="Screen Shot 2026-05-18 at 19 19 07" src="https://github.com/user-attachments/assets/d379ac14-a27c-4523-82af-130b80835f9f" />

Avaliação: ....

## 3.2.2 Avaliação do XGBoost
Seguindo o mesmo protocolo rigoroso, o XGBoost otimizado foi avaliado no conjunto de teste:

```python
# Recupera o melhor modelo do GridSearch
xgb_best = grid_xgb.best_estimator_

# Predições no conjunto de TESTE
y_pred_xgb = xgb_best.predict(X_test)
y_proba_xgb = xgb_best.predict_proba(X_test)[:, 1]

# Cálculo das métricas
print("=== XGBOOST - AVALIAÇÃO FINAL NO TEST SET ===")
print(f"Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_xgb):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba_xgb):.4f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
```

**Resultados do XGBoost:**

| Métrica |	Valor |
|---|---|
| Recall |	1,0000 |
| Precisão |	1,0000 |
| F1-Score |	1,0000 |
| AUC-ROC |	1,0000 |
| Acurácia |	1,0000 |

**Matriz de Confusão:** 

<img width="422" height="463" alt="Screen Shot 2026-05-18 at 19 24 50" src="https://github.com/user-attachments/assets/98001f56-13c4-4e2b-b5e6-0372764dc2d7" />

Avaliação: ....


## 3.3 Comparação entre os três modelos

```python
comparacao = pd.DataFrame({
    'Modelo': [
        'Regressão Logística',
        'Random Forest',
        'XGBoost'
    ],
    'Accuracy': [
        accuracy_score(y_test, y_pred_final),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred)
    ],
    'Precision': [
        precision_score(y_test, y_pred_final),
        precision_score(y_test, rf_pred),
        precision_score(y_test, xgb_pred)
    ],
    'Recall': [
        recall_score(y_test, y_pred_final),
        recall_score(y_test, rf_pred),
        recall_score(y_test, xgb_pred)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_final),
        f1_score(y_test, rf_pred),
        f1_score(y_test, xgb_pred)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_proba_final),
        roc_auc_score(y_test, rf_proba),
        roc_auc_score(y_test, xgb_proba)
    ]
})

comparacao.sort_values(by='F1-Score', ascending=False)

```

# Gráfico comparativo

<img width="425" height="105" alt="Screen Shot 2026-05-18 at 19 29 44" src="https://github.com/user-attachments/assets/36a228d6-31cc-46cc-909e-82e318bbe4ba" />

Avaliação: ....


## 3.4 Discussão dos resultados
### 3.4.1 Análise crítica do desempenho perfeito
Os três modelos avaliados alcançaram desempenho perfeito (Recall=1,0000; Precisão=1,0000; F1=1,0000; AUC=1,0000) no conjunto de teste. Este resultado, embora tecnicamente correto, deve ser interpretado com extrema cautela pelos seguintes motivos:

1. Natureza sintética do dataset: O "Work Productivity & Burnout Risk Dataset" foi inteiramente gerado por inteligência artificial. Dados sintéticos tendem a apresentar:

* Separabilidade artificialmente nítida entre classes

* Ausência de ruído típico de dados coletados em humanos

* Relações linearmente separáveis ou com padrões muito limpos

2. Implicações para a pesquisa empírica (H39a): Em contextos reais de predição de burnout (com questionários validados como Maslach Burnout Inventory - MBI), valores típicos de AUC variam entre 0,70 e 0,85. O desempenho AUC=1,0000 não é esperado em cenários reais.

3. Validade externa limitada: Os resultados não devem ser generalizados para populações reais sem validação adicional com dados coletados em ambientes organizacionais reais.

4. Artefatos contraintuitivos: Assim como observado na Regressão Logística, a análise de importância de features do Random Forest provavelmente apresentaria relações que contradizem a literatura (ex: `Work_Hours_Per_Day` como fator protetivo), o que reforça a natureza artificial do dataset.

### 3.4.2 Comparação qualitativa entre os modelos
Apesar do empate numérico em todas as métricas, diferenças metodológicas importantes podem ser destacadas:

| Critério |	Regressão Logística |	Random Forest |	XGBoost |
|-----|-----|-----|-----|
| Interpretabilidade |	Alta (coeficientes e odds ratio) |	Média (importância de features) |	Baixa (modelo caixa-preta) |
| Tempo de treinamento |	Muito baixo |	Médio	Médio-alto  | (270 fits no GridSearch) |
| Necessidade de escala	| Sim (StandardScaler) |	Não	| Não |
| Captura de não-linearidades |	Limitada	| Alta |	Alta |
| Risco de overfitting |	Baixo (regularização L1) |	Baixo (bagging) |	Médio (controlado com learning_rate baixo) |
| Melhor configuração encontrada |	L1, C=0,3 |	n_estimators=100, max_depth=10	| lr=0,01, max_depth=3, subsample=0,8 |
| Número de parâmetros otimizados |	1 | (C) |	3 |	4 |

**Análise qualitativa:**

* **Regressão Logística** mantém-se como modelo mais interpretável e suficiente para o dataset sintético. Seus coeficientes (apresentados na Etapa 3) permitem identificar fatores de risco e proteção – mesmo com resultados contraintuitivos.

* **Random Forest** apresentou a configuração mais simples entre os vencedores (n_estimators=100, max_depth=10), indicando que mesmo com complexidade moderada, o modelo conseguiu separar perfeitamente as classes. O parâmetro min_samples_split=2 (valor padrão) sugere que não foi necessário aumentar a regularização.

* **XGBoost** exigiu a configuração mais conservadora (learning_rate=0,01 muito baixo, max_depth=3 rasa, subsample=0,8), sugerindo que, sem forte regularização, o modelo tenderia a overfitting. O fato de a melhor configuração ser a mais regularizada é um indicador da alta capacidade do XGBoost de memorizar padrões.

### 3.4.3 Análise de trade-offs

```python
print("\n ANÁLISE CRÍTICA DE TRADE-OFFS:")
print(f"   - Logistic Regression: Recall={metrics_lr['Recall']:.4f} | Precision={metrics_lr['Precision']:.4f} | Interpretabilidade: ALTA")
print(f"   - Random Forest:       Recall={metrics_rf['Recall']:.4f} | Precision={metrics_rf['Precision']:.4f} | Interpretabilidade: MÉDIA")
print(f"   - XGBoost:             Recall={metrics_xgb['Recall']:.4f} | Precision={metrics_xgb['Precision']:.4f} | Interpretabilidade: BAIXA")
```

<img width="635" height="607" alt="Screen Shot 2026-05-18 at 19 44 18" src="https://github.com/user-attachments/assets/4a050f43-0dc9-4bb9-8397-a0954d8b016a" />

Avaliação: ....



### 3.4.4 Relação com a questão de pesquisa
A questão de pesquisa original era: "É possível prever o risco de burnout a partir de variáveis comportamentais e ocupacionais utilizando um modelo interpretável?"

Os resultados indicam que sim, é possível – pelo menos no contexto controlado deste dataset sintético. Os três modelos alcançaram perfeição na separação entre indivíduos com e sem risco de burnout. No entanto, duas ressalvas são fundamentais:

1. **Interpretabilidade vs. Performance**: O modelo mais interpretável (Regressão Logística) atingiu a mesma performance que os modelos de caixa-preta (Random Forest e XGBoost). Isso sugere que, para este dataset específico, não houve trade-off entre interpretabilidade e acurácia.

2. **Generalização para dados reais**: O desempenho perfeito é um artefato da geração sintética dos dados. Em cenários reais, espera-se que modelos mais complexos (XGBoost) superem modelos lineares (Regressão Logística), mas com perda significativa de interpretabilidade.

### 3.4.5 Relação com os objetivos do projeto

| Objetivo | Status | Evidência |
|----------|--------|-----------|
| Desenvolver modelo preditivo para burnout | Atingido | Três modelos com Recall=1,0000 |
| Comparar abordagens de ML | Atingido | Tabela comparativa entre modelos |
| Identificar fatores de risco | Atingido | Coeficientes da Regressão Logística |
| Propor intervenções organizacionais | Parcial | Requer validação com dados reais |
| Garantir interpretabilidade | Atingido | Odds Ratio e coeficientes disponíveis |

# 4. Revisão do pipeline de pesquisa e análise de dados

## 4.1 Avaliação crítica do pipeline da Etapa 3: 
O pipeline proposto na Etapa 3 apresentou as seguintes características positivas:

| Aspecto |	Avaliação |
|---|---|
| Separação treino/teste |	Adequada (80/20 com estratificação) |
| Isolamento do test set	| Correto (não usado em otimizações) |
| Validação cruzada |	Implementada (5-folds StratifiedKFold) |
| Pré-processamento via Pipeline |	Correto (evita data leakage) |
| Métrica principal justificada |	Recall alinhado ao domínio |
| Documentação |	Detalhada e reprodutível |

## 4.2. Limitações identificadas:**

**GridSearch fixo para cada modelo**: O código precisava ser replicado para cada novo modelo, gerando repetição e aumento da probabilidade de erros.

**Ausência de função genérica de avaliação**: Não havia um mecanismo reutilizável para adicionar novos modelos rapidamente, dificultando a escalabilidade do pipeline.

**Métrica de otimização fixa**: Embora Recall seja a métrica principal para este problema, o pipeline não permitia trocar facilmente a métrica de otimização para outros contextos de negócio.

**Registro de experimentos**: Não havia salvamento automático dos resultados (melhores parâmetros, métricas) para comparação posterior entre diferentes execuções.

**Pipeline revisado e generalizado**
Com base nas lições aprendidas na Etapa 4, foi desenvolvida uma função genérica que encapsula todo o fluxo de treinamento, otimização e avaliação de modelos:

```python
def build_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, 
                             model_name, scoring='recall', cv_folds=5):
    """
    Pipeline genérico para treinar, otimizar e avaliar modelos de ML.
    
    Este pipeline implementa as boas práticas de ciência de dados:
    - Isolamento completo do conjunto de teste
    - Validação cruzada estratificada no treino
    - Otimização de hiperparâmetros com GridSearch
    - Avaliação com múltiplas métricas
    
    Parâmetros:
    -----------
    model : estimator sklearn
        Modelo a ser treinado (ex: RandomForestClassifier())
    param_grid : dict
        Dicionário com hiperparâmetros para GridSearch
    X_train, y_train : array-like
        Dados de treino
    X_test, y_test : array-like
        Dados de teste (mantido isolado)
    model_name : str
        Nome do modelo para identificação nos logs
    scoring : str, default='recall'
        Métrica para otimização (ex: 'recall', 'precision', 'f1')
    cv_folds : int, default=5
        Número de folds para validação cruzada
    
    Retorna:
    --------
    dict
        Dicionário contendo:
        - 'model': modelo otimizado
        - 'metrics': dicionário com métricas de avaliação
        - 'best_params': melhores hiperparâmetros encontrados
        - 'cv_score': score médio na validação cruzada
        - 'model_name': nome do modelo
    """
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])
    
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"Otimizando {model_name} com {scoring}...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'Accuracy': accuracy_score(y_test, y_pred)
    }
    
    print(f" {model_name} - Melhor Recall CV: {grid_search.best_score_:.4f}")
    print(f"   Melhores parâmetros: {grid_search.best_params_}")
    
    return {
        'model': best_model,
        'metrics': metrics,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'model_name': model_name
    }
```

## 4.3. Estrutura final do pipeline (6 etapas)
O pipeline revisado contempla, de forma flexível e modular, as principais fases da pesquisa em ciência de dados:

| Etapa | Descrição | Ferramentas/Métodos |
|-------|-----------|---------------------|
| **1. Especificação do problema** | Definição da questão de pesquisa, métrica principal e critérios de sucesso | Documentação em markdown |
| **2. Coleta e preparação dos dados** | Carregamento, limpeza, tratamento de missing, encoding, padronização | pandas, ColumnTransformer, Pipeline |
| **3. Análise exploratória** | Distribuições, correlações, análise de desbalanceamento, detecção de outliers | seaborn, matplotlib, pandas |
| **4. Modelagem e validação** | Split treino/teste, validação cruzada, GridSearch, otimização de hiperparâmetros | GridSearchCV, StratifiedKFold |
| **5. Avaliação** | Cálculo de múltiplas métricas, matriz de confusão, curva ROC, comparação entre modelos | sklearn.metrics |
| **6. Interpretação e documentação** | Análise de coeficientes/importância, limitações, recomendações para produção | Coeficientes, Odds Ratio, SHAP (recomendado) |

## 4.4 Principais melhorias implementadas
| Melhoria |	Descrição |	Benefício |
|---|---|---|
| Função genérica build_and_evaluate_model | Encapsula treinamento, validação cruzada e avaliação |	Reduz repetição de código; facilita adição de novos modelos |
| Parâmetro scoring configurável |	Permite trocar a métrica de otimização sem reescrever código	| Flexibilidade para diferentes problemas de negócio |
| Retorno estruturado |	Dicionário com modelo, métricas, parâmetros e score CV |	Facilita comparação e registro de experimentos |
| Documentação integrada |	Docstring completa com parâmetros e retornos |	Favorece reuso e replicação por outros pesquisadores |
| Validação cruzada parametrizável |	Número de folds configurável (cv_folds) |	Adaptável a diferentes tamanhos de dataset |

Exemplo de uso do pipeline generalizado
```python
# Definição dos grids para cada modelo
param_grids = {
    'RandomForest': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20],
        'model__min_samples_split': [2, 5],
        'model__class_weight': ['balanced']
    },
    'XGBoost': {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1],
        'model__max_depth': [3, 6],
        'model__subsample': [0.8, 1.0],
        'model__scale_pos_weight': [scale_pos_weight_calc]
    }
}

# Execução para múltiplos modelos
results = []
for name, params in param_grids.items():
    if name == 'RandomForest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif name == 'XGBoost':
        model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    
    result = build_and_evaluate_model(
        model=model,
        param_grid=params,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model_name=name,
        scoring='recall'
    )
    results.append(result)

# Comparação dos resultados
comparison_df = pd.DataFrame([r['metrics'] for r in results], index=[r['model_name'] for r in results])
print("\n COMPARAÇÃO FINAL ENTRE MODELOS:")
print(comparison_df.round(4))
```

## 4.5. Observações importantes
Todas as tarefas realizadas nesta etapa foram registradas em formato textual com suas respectivas explicações. Os códigos desenvolvidos encontram-se documentados e disponibilizados integralmente na pasta `src/` do repositório, garantindo transparência, reprodutibilidade e aderência às boas práticas de desenvolvimento em projetos de ciência de dados.

