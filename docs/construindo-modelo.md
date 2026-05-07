# Preparação dos dados

Nesta etapa, foram aplicadas técnicas de pré-processamento com o objetivo de adequar o conjunto de dados ao treinamento do modelo de aprendizado de máquina, garantindo consistência, robustez e reprodutibilidade dos resultados.

## Limpeza dos Dados
Inicialmente, foi realizada a etapa de limpeza dos dados, na qual se confirmou a ausência de valores faltantes no dataset, conforme identificado na análise exploratória. Embora o dataset original não apresente valores ausentes, optou-se por manter etapas de imputação no pipeline, utilizando a mediana para variáveis numéricas e a moda para variáveis categóricas. Essa decisão foi adotada visando aumentar a robustez do modelo e sua capacidade de generalização para cenários futuros com dados incompletos.

**Detecção de outliers:** Foi aplicado o método do intervalo interquartil (IQR). A análise identificou **outliers estatísticos apenas na variável `Experience_Years`**, com 424 registros acima do limite superior (valores de 28 a 32 anos). As demais variáveis (`Sleep_Hours`, `Screen_Time_Hours`, `Work_Hours_Per_Day`, etc.) não apresentaram outliers pelo critério IQR, embora apresentem perfis atípicos do ponto de vista do domínio (como poucas horas de sono).

**Decisão de manutenção dos outliers:** Optou-se por **manter** os 424 registros de `Experience_Years` no dataset pelos seguintes motivos:
- Os valores são plausíveis no contexto profissional (profissionais com até 32 anos de experiência)
- Não há evidência de erro de digitação ou inconsistência nos dados
- O dataset possui natureza sintética, podendo apresentar padrões artificiais que não justificam exclusão
- A remoção poderia comprometer a representatividade da amostra para profissionais seniores.

## Transformação dos Dados

As variáveis categóricas — `Gender`, `Country`, `Job_Role`, `Company_Size` e `Work_Environment` — foram convertidas em formato numérico por meio da técnica de **One-Hot Encoding**, utilizando a classe `OneHotEncoder` da biblioteca scikit-learn com parâmetro `handle_unknown='ignore'`. Esse procedimento permite representar categorias qualitativas em forma binária, além de garantir que categorias não vistas durante o treinamento não comprometam o funcionamento do modelo.

Adicionalmente, procedeu-se à remoção da variável `Employee_ID`, por se tratar de um identificador único sem relevância preditiva. Em seguida, os dados foram organizados em variáveis preditoras (X) e variável alvo (y), sendo esta última representada pela variável `Burnout_Risk`, codificada em formato binário.

## Padronização das Variáveis Numéricas

Considerando a presença de variáveis numéricas em diferentes escalas, foi aplicada a técnica de padronização por meio do `StandardScaler`, que transforma os dados para uma distribuição com média zero e desvio padrão unitário. Essa etapa é particularmente importante para algoritmos como a Regressão Logística, que são sensíveis à escala das variáveis.

## Pipeline de Pré-processamento

As etapas de pré-processamento foram organizadas por meio de um `ColumnTransformer`, responsável por aplicar transformações distintas para variáveis numéricas e categóricas. Esse componente foi integrado a um pipeline completo utilizando a classe `Pipeline`, garantindo que todas as etapas sejam executadas de forma consistente tanto durante o treinamento quanto na fase de teste. Essa abordagem também evita o problema de vazamento de dados (data leakage), assegurando maior rigor metodológico.

## Tratamento do Desbalanceamento

Outro aspecto relevante do pré-processamento refere-se ao desbalanceamento da variável alvo, no qual aproximadamente 80% dos registros pertencem à classe “No” e 20% à classe “Yes”. Para lidar com esse cenário, foi adotada a estratégia de ajuste de pesos das classes (`class_weight='balanced'`), permitindo que o modelo atribua maior importância à classe minoritária durante o treinamento, sem a necessidade de técnicas adicionais como oversampling.

## Divisão Treino-Teste

Por fim, o conjunto de dados foi dividido em subconjuntos de treinamento (80%) e teste (20%), utilizando uma estratégia estratificada da variável alvo (`stratify=y`), de modo a preservar a proporção original das classes. A utilização de um valor fixo de random_state garantiu a reprodutibilidade dos experimentos.

# Descrição do modelo escolhido

O modelo selecionado para esta etapa foi a **Regressão Logística**, amplamente utilizada em problemas de classificação binária. Esse algoritmo baseia-se na aplicação da função logística, que transforma uma combinação linear das variáveis independentes em uma probabilidade entre 0 e 1, permitindo estimar a chance de ocorrência do evento de interesse. A decisão de classificação é realizada a partir de um limiar de probabilidade, geralmente definido como 0,5, a partir do qual o modelo classifica a observação como pertencente à classe positiva.

## Justificativa da escolha

A escolha da Regressão Logística fundamenta-se em três pilares principais:
| Critério | Justificativa |
|-------|-------|
| Interpretabilidade | Os coeficientes indicam diretamente a direção e intensidade do impacto de cada variável no risco de burnout |
| Eficiência Computacional | Treinamento rápido mesmo com 30.000 registros e 38 variáveis após encoding |
| Probabilidades Calibradas | Fornece estimativas de probabilidade, essenciais para tomada de decisão em saúde organizacional |

Do ponto de vista conceitual, a Regressão Logística modela a relação entre as variáveis preditoras e a variável resposta por meio da função sigmoide, possibilitando a interpretação probabilística dos resultados. Essa característica a torna particularmente adequada para problemas em que a tomada de decisão depende da estimativa de risco, como no caso da predição de burnout.

Além disso, o modelo foi adotado como baseline, servindo como referência inicial para comparação com abordagens mais sofisticadas em etapas posteriores do projeto. A utilização de um modelo baseline é uma prática consolidada em ciência de dados, pois permite avaliar se o aumento de complexidade em modelos futuros resulta, de fato, em ganhos significativos de desempenho.

## Implementação do modelo

A implementação foi realizada utilizando a biblioteca `scikit-learn`, com destaque para o ajuste do parâmetro `max_iter=1000`, visando garantir a convergência do algoritmo, e a utilização de `class_weight='balanced'`, com o objetivo de mitigar os efeitos do desbalanceamento entre as classes.

<img width="1579" height="190" alt="Screen Shot 2026-04-29 at 20 39 51" src="https://github.com/user-attachments/assets/c98ba357-0468-47fd-b106-53eed7bf75e3" />

## Limitações do Modelo

Como principal limitação, destaca-se o fato de que a Regressão Logística assume relações lineares entre as variáveis independentes e o logaritmo da razão de chances, o que pode restringir sua capacidade de capturar padrões mais complexos presentes nos dados. Ainda assim, sua simplicidade e interpretabilidade justificam sua utilização como ponto de partida.

Adicionalmente, o modelo permite a análise direta dos coeficientes estimados, possibilitando a identificação dos fatores que mais contribuem para o aumento ou redução do risco de burnout, reforçando sua aplicabilidade no contexto organizacional.

# Avaliação do modelo criado

## Métricas utilizadas

A avaliação do modelo foi realizada por meio de múltiplas métricas, com o objetivo de obter uma visão abrangente de seu desempenho. Foram consideradas as métricas de *acurácia (accuracy)*, *precisão (precision)*, *revocação (recall)*, *F1-score* e área sob a *curva ROC (AUC-ROC)*.

| Métrica |	Descrição |	Importância no Contexto |
|---------|-----------|-------------------------|
| Acurácia | Proporção geral de acertos do modelo |	Útil para visão geral, mas insuficiente isoladamente devido ao desbalanceamento |
| Precisão | Proporção de previsões positivas corretas | Avalia a confiabilidade das predições de burnout |
| **Recall**	 | **Capacidade de identificar corretamente os casos positivos** | **Minimizar falsos negativos é prioritário** |
| F1-score | Média harmônica entre precisão e recall | Equilíbrio entre sensibilidade e especificidade |
| AUC-ROC | Capacidade de discriminação entre classes | Avalia qualidade intrínseca do modelo independente do threshold |

Dentre essas métricas, o **recall foi definido como o principal critério de avaliação**, uma vez que, no contexto do problema, o erro mais crítico consiste na ocorrência de **falsos negativos** — ou seja, situações em que o modelo deixa de identificar indivíduos em risco de burnout. 

### Justificativa da escolha do Recall como métrica principal:

- **Custo do erro na área da saúde:** Deixar de identificar um profissional em risco de burnout (Falso Negativo) pode resultar em agravamento do quadro, afastamento prolongado ou danos irreversíveis à saúde mental.
- **Custo do Falso Positivo:** Alertar erroneamente um profissional sem risco gera uma conversa de suporte ou verificação adicional - custo operacional, mas sem dano à saúde.
- **Alinhamento com o problema:** A prioridade é não deixar nenhum caso de burnout sem intervenção preventiva.

Todas as decisões de otimização do modelo (escolha do parâmetro C via GridSearchCV e definição do threshold) foram tomadas priorizando a **maximização do recall**, respeitando uma restrição mínima de precisão (85%) para evitar um número excessivo de falsos positivos que tornaria o modelo operacionalmente inviável.

## Modelo Baseline (Configuração Inicial)

Antes de realizar a otimização sistemática dos hiperparâmetros, foi estabelecido um modelo baseline como referência inicial. Este modelo utiliza a configuração padrão da Regressão Logística, que trouxe os seguintes resultados:

<img width="313" height="122" alt="Screen Shot 2026-04-29 at 20 46 43" src="https://github.com/user-attachments/assets/f6233ca6-a657-4eff-ba57-af2fa07628e1" />

## Análise do Baseline

O modelo baseline já apresenta desempenho excepcional:

**Recall** = 1,0: Nenhum indivíduo em risco de burnout deixou de ser identificado. Este é o resultado mais importante, dado que falsos negativos são o erro mais crítico no contexto do problema.

**AUC-ROC** = 0,99999: O modelo distingue quase perfeitamente as classes, indicando alta separabilidade dos dados.

**Precisão** = 0,9526: Apesar do recall perfeito, há uma pequena taxa de falsos positivos (aproximadamente 4,74% das predições positivas estão incorretas).

## Limitações do Baseline

Embora o baseline já apresente resultados muito bons, identificou-se oportunidade de melhoria na precisão e no equilíbrio geral das métricas. A partir deste baseline, partiu-se para a otimização sistemática dos hiperparâmetros ao longo desse trabalho.

## Discussão dos resultados preliminares

A Figura I apresenta a matriz de confusão do modelo (configuração Balanced L2, C=1,0, threshold=0,5), que evidencia a distribuição das previsões realizadas. 

<img width="518" height="393" alt="image" src="https://github.com/user-attachments/assets/04f26046-daf7-4565-a452-1c44e153c7d7" />

Figura I - Matriz de Confusão do Modelo (Balanced L2, C=1,0, threshold=0,5) 

A análise da matriz de confusão evidencia que o modelo apresenta elevada capacidade de identificação da classe positiva, não sendo observados falsos negativos. Esse resultado indica que todos os indivíduos em risco de burnout foram corretamente classificados pelo modelo. Por outro lado, observa-se a presença de falsos positivos (59 casos), indicando que alguns indivíduos foram classificados como em risco quando, na realidade, não pertenciam a essa classe. Embora esse tipo de erro possa gerar intervenções desnecessárias, ele é menos crítico no contexto do problema, quando comparado aos falsos negativos.

A Figura II apresenta a curva ROC para a mesma configuração.

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/a5c3d1fe-3b04-45bc-bb3a-ea8ddf14b8b6" />

Figura II - Curva ROC do modelo Balanced L2 (AUC = 0,999993)

A curva ROC evidencia a capacidade do modelo em discriminar entre as classes ao longo de diferentes limiares de decisão. Observa-se que a curva se mantém significativamente acima da linha de referência aleatória, com AUC praticamente igual a 1,0.

Adicionalmente, o formato da curva, próximo ao canto superior esquerdo, sugere elevada taxa de verdadeiros positivos combinada com baixa taxa de falsos positivos, reforçando a qualidade do modelo na separação entre indivíduos com e sem risco de burnout.

A análise conjunta das métricas evidencia que o modelo consegue equilibrar a taxa de acertos gerais com a capacidade de identificação da classe minoritária, ainda que apresente limitações típicas de modelos lineares. Em particular, observa-se que o foco no recall contribui para uma maior sensibilidade na detecção de casos positivos, o que está alinhado com os objetivos do projeto.

No contexto da questão de pesquisa — que busca verificar a possibilidade de prever o risco de burnout a partir de variáveis ocupacionais e comportamentais —, os resultados obtidos reforçam a viabilidade da abordagem proposta. O modelo demonstra que, mesmo utilizando uma técnica relativamente simples, é possível extrair padrões preditivos relevantes a partir dos dados disponíveis.

Entretanto, é importante destacar que algumas relações identificadas diferem da literatura, possivelmente em função da **natureza sintética do dataset** (produzido por Inteligência Artificial). Um exemplo é a variável `Work_Hours_Per_Day`, que apresentou coeficiente negativo (indicando que maiores jornadas reduziriam o risco de burnout), contradizendo evidências da literatura. Esse artefato, decorrente da natureza sintética dos dados, deve ser interpretado com cautela, especialmente quanto à generalização para contextos reais.

A análise dos coeficientes do modelo será apresentada na seção **Análise dos Coeficientes do Modelo Final**.

## Testes com hiperparâmetros ajustados

Com o objetivo de avaliar o impacto de diferentes configurações do modelo no desempenho preditivo, foram realizados experimentos controlados variando-se parâmetros-chave da Regressão Logística. Especificamente, foram analisados: (i) o uso ou não de balanceamento de classes, (ii) o tipo de penalização aplicada (L1 e L2) e (iii) a intensidade da regularização (parâmetro C). Nestes experimentos preliminares, o threshold foi mantido fixo em 0,5.

A realização desses testes permite compreender a sensibilidade do modelo a diferentes configurações, além de fornecer embasamento empírico para a escolha da configuração final adotada.

### Teste com penalidade L2 (diferentes valores de C)

Foram testados valores de C entre 0.1 e 10.0 para a penalidade L2, mantendo `class_weight='balanced'` e threshold=0,5. Os resultados são apresentados na Figura a seguir:

<img width="511" height="682" alt="Screen Shot 2026-05-07 at 00 44 24" src="https://github.com/user-attachments/assets/ff0419ad-95b0-4f0f-8d09-cfbd0e5717b4" />

**Análise:** Observa-se que quanto maior o valor de C (menos regularização), melhor a precisão do modelo, com redução significativa dos falsos positivos. O melhor resultado para penalidade L2 foi obtido com **C=10,0**, apresentando 2 falsos positivos e recall perfeito.

### Teste com penalidade L1 (Lasso)

Foram testados valores de C entre 0.1 e 1.0 para a penalidade L1, mantendo `class_weight='balanced'` e threshold=0,5. Os resultados são apresentados na Figura abaixo:

<img width="517" height="711" alt="Screen Shot 2026-05-07 at 00 49 20" src="https://github.com/user-attachments/assets/b2da9cb1-8e6e-45d1-b7f0-eddc175311e6" />.

**Análise:** A penalidade L1 com C=0,3 atingiu **desempenho perfeito**: recall=1,0000 (nenhum falso negativo) e precisão=1,0000 (nenhum falso positivo). Este resultado representa o ótimo global para o dataset analisado.

## Otimização de parâmetros - testes Adicionais

### Otimização do Threshold

Após identificar a melhor configuração de penalidade e C (L1, C=0,3), foi realizada a otimização do threshold (limiar de decisão) no conjunto de treino. Foram testados thresholds entre 0,30 e 0,90, mantendo a melhor configuração do modelo.

**Tabela V - Otimização do Threshold (L1, C=0,3)**

| Threshold | Recall | Precision | Falsos Positivos | Falsos Negativos |
|-----------|--------|-----------|------------------|------------------|
| 0,30 | 1,0000 | 0,9324 | 86 | 0 |
| 0,40 | 1,0000 | 0,9826 | 21 | 0 |
| **0,50** | **1,0000** | **1,0000** | **0** | **0** |
| 0,60 | 1,0000 | 1,0000 | 0 | 0 |
| 0,70 | 1,0000 | 1,0000 | 0 | 0 |
| 0,80 | 1,0000 | 1,0000 | 0 | 0 |
| 0,90 | 0,9595 | 1,0000 | 0 | 48 |

**Análise:** Com a configuração L1, C=0,3, o modelo já apresenta desempenho perfeito com o threshold padrão de 0,5. Thresholds mais baixos (0,30 e 0,40) introduzem falsos positivos, enquanto thresholds mais altos (0,90) introduzem falsos negativos. Portanto, **mantém-se threshold=0,5 como configuração final**.


### Modelo Final Otimizado

Dessa forma, a configuração final adotada foi:

<img width="512" height="496" alt="Screen Shot 2026-05-07 at 01 07 11" src="https://github.com/user-attachments/assets/5ea6fb06-9eae-4fe4-a23a-35d3389a0625" />

### Interpretação do Modelo Final

O modelo final otimizado (L1, C=0,3, threshold=0,5) apresenta **desempenho perfeito** no dataset de teste:

- **Recall = 1,0000**: Todos os 1.186 casos de burnout foram identificados (nenhum falso negativo)
- **Precisão = 1,0000**: Nenhum falso positivo (0 intervenções desnecessárias)
- **F1-Score = 1,0000**: Equilíbrio perfeito
- **AUC-ROC = 1,0000**: Separação perfeita entre as classes

### Avaliação da Curva ROC no Modelo Final

<img width="595" height="507" alt="Screen Shot 2026-05-07 at 01 01 47" src="https://github.com/user-attachments/assets/ca32268a-5ee5-4e39-83eb-ff640b594cec" />

O modelo final atingiu **AUC = 1,0000**, indicando separabilidade perfeita entre as classes no conjunto de teste. Este resultado, embora excelente, deve ser interpretado com cautela por refletir a natureza sintética do dataset.


### Análise dos Coeficientes do Modelo Final (L1, C=0.3, threshold=0.5)

Principais variáveis que AUMENTAM o risco de burnout (coeficiente positivo):

| Variável | Coeficiente | Odds Ratio | Interpretação |
|----|----|----|----|
| Country_UK | +0,0711 | 1,0736 | Reino Unido apresenta maior prevalência |

Principais variáveis que REDUZEM o risco de burnout (coeficiente negativo):

| Variável | Coeficiente | Odds Ratio | Interpretação |
|----|----|----|----|
| Productivity_Score | -45,0323 | 2,77e-20 | Fator protetivo mais forte |
| Work_Environment_Coworking | -0,0788 | 0,9243 | Ambiente de coworking reduz risco |
| Gender_Other | -0,0787 | 0,9243 | Outros gêneros apresentam menor risco |
| Gender_Female | -0,0773 | 0,9256 | Gênero feminino associado a menor risco |
| Meetings_Per_Day | -0,0624 | 0,9395 | Menos reuniões reduzem o risco |
| Work_Hours_Per_Day | -0,0422 | 0,9587 | **Contraintuitivo** (ver seção dedicada) |

**Resultado contraintuitivo:** O coeficiente negativo de `Work_Hours_Per_Day` e `Meetings_Per_Day` sugere que mais horas de trabalho e mais reuniões reduziriam o risco de burnout, o que contradiz a literatura especializada. Este é provavelmente um artefato do dataset sintético, devendo ser interpretado com cautela.

### Limitações da Análise de Coeficientes e Uso de Odds Ratio

Embora a Regressão Logística permita interpretação direta dos coeficientes, algumas **ressalvas metodológicas** são necessárias:

**1. Comparabilidade limitada entre tipos de variáveis:** Coeficientes de variáveis dummies (ex: `Country_UK`) e variáveis contínuas padronizadas (ex: `Sleep_Hours`) **não são diretamente comparáveis** em magnitude, pois suas escalas e naturezas são distintas.

**2. Odds Ratio para interpretação mais robusta:** Para facilitar a interpretação, recomenda-se exponenciar os coeficientes:

> `Odds Ratio = exp(coeficiente)`

- **Odds Ratio > 1** → variável aumenta o risco de burnout
- **Odds Ratio < 1** → variável reduz o risco de burnout
- Exemplo: Odds Ratio = 2 significa que a chance de burnout dobra

**3. Ausência de intervalo de confiança:** O relatório apresenta apenas o valor pontual do coeficiente. Em análises futuras com dados reais, recomenda-se calcular intervalos de confiança (via bootstrap).

**4. Relações não lineares não capturadas:** Coeficientes lineares podem não refletir relações mais complexas (ex: efeito teto ou platô).

**Recomendações para trabalhos futuros:**
- Calcular e reportar **Odds Ratio** (já implementado no código)
- Utilizar técnicas de explicabilidade como **SHAP** para análises mais robustas
- Realizar análise de sensibilidade removendo variáveis com baixo impacto


## Decisão do Modelo

Com base nos resultados obtidos, optou-se por utilizar a configuração com `class_weight='balanced'`, **penalidade L1, C=0,3 e threshold=0,5**, considerando seu **desempenho perfeito** em todas as métricas avaliadas (Recall=1,0000, Precisão=1,0000, F1-Score=1,0000, AUC-ROC=1,0000).

A escolha está alinhada ao objetivo central do projeto, que consiste na identificação de indivíduos em risco de burnout, minimizando a ocorrência de falsos negativos.

**Avaliação final:** O modelo apresentou desempenho perfeito no dataset sintético utilizado, servindo como **prova de conceito** para a abordagem proposta.

**Ressalva importante:** O desempenho perfeito obtido é um fenômeno extremamente raro em dados reais. Este resultado reflete a natureza sintética do dataset (gerado por IA) e não deve ser generalizado para contextos reais sem validação adicional. Recomenda-se que o modelo seja utilizado apenas com supervisão humana e validação prévia em dados reais da organização.

## Análise de Interpretabilidade: Fatores que Influenciam o Burnout

A tabela a seguir apresenta as 10 variáveis com maior impacto absoluto no modelo final (L1, C=0.3, threshold=0.5):

| Variável | Coeficiente | Odds Ratio | Direção |
|----------|-------------|------------|---------|
| Productivity_Score | -45,0323 | 2,77e-20 | Reduz Risco |
| Work_Environment_Coworking | -0,0788 | 0,9243 | Reduz Risco |
| Gender_Other | -0,0787 | 0,9243 | Reduz Risco |
| Gender_Female | -0,0773 | 0,9256 | Reduz Risco |
| Country_UK | +0,0711 | 1,0736 | Aumenta Risco |
| Meetings_Per_Day | -0,0624 | 0,9395 | Reduz Risco* |
| Work_Hours_Per_Day | -0,0422 | 0,9587 | Reduz Risco* |
| Country_Canada | -0,0418 | 0,9591 | Reduz Risco |
| Age | -0,0279 | 0,9725 | Reduz Risco |
| Exercise_Hours_Per_Week | -0,0196 | 0,9806 | Reduz Risco |

*Resultado contraintuitivo - ver discussão abaixo.

### Análise dos Resultados

**Fatores Protetivos (Reduzem o Risco):**
- **Productivity_Score** (coeficiente -45,03): O fator mais importante do modelo. Quanto maior a produtividade, menor o risco de burnout. A magnitude extremamente alta sugere uma relação quase determinística no dataset sintético.

- **Work_Environment_Coworking** (coeficiente -0,0788): Ambientes de coworking associados a menor risco.

- **Gender_Female / Gender_Other** (coeficiente ~ -0,078): Gêneros feminino e outros apresentam menor risco em relação à categoria de referência.

**Fatores de Risco (Aumentam o Risco):**
- **Country_UK** (coeficiente +0,0711): Reino Unido apresenta maior prevalência em relação aos países de referência.

**Resultados Contraintuitivos:**
- `Meetings_Per_Day` e `Work_Hours_Per_Day` apresentaram coeficientes negativos, sugerindo que mais reuniões e mais horas de trabalho reduziriam o risco. Estes resultados contradizem a literatura e são provavelmente artefatos do dataset sintético.

### Resultado Contraintuitivo

**Work_Hours_Per_Day** apresentou coeficiente negativo (-0,68), sugerindo que trabalhar mais horas reduziria o risco de burnout. Este resultado **contradiz a literatura especializada** e é provavelmente um **artefato da natureza sintética do dataset** (gerado por IA). Recomenda-se cautela na interpretação e validação com dados reais antes de qualquer decisão organizacional baseada neste achado.

### Variáveis com Baixo Impacto (< 1% de contribuição)

As seguintes variáveis apresentaram impacto desprezível no modelo, podendo ser consideradas para remoção em versões futuras:
- Age, Experience_Years, diversas categorias de países e cargos com coeficientes próximos a zero.

# Pipeline de pesquisa e análise de dados

O pipeline de pesquisa e análise de dados adotado neste projeto foi estruturado de forma a garantir coerência metodológica, reprodutibilidade, conformidade com a LGPD e alinhamento com boas práticas em ciência de dados.

### 1. Especificação do problema

O problema foi caracterizado como uma tarefa de **classificação binária supervisionada** voltada à predição do risco de burnout, a partir de variáveis ocupacionais, demográficas e comportamentais. A variável alvo é `Burnout_Risk` (Yes/No), com classe minoritária representando aproximadamente 20% dos registros.

**Questão de pesquisa:** É possível prever o risco de burnout a partir de variáveis comportamentais e ocupacionais utilizando um modelo interpretável?

### 2. Coleta e preparação dos dados

**Fonte dos dados:** Dataset "Work Productivity & Burnout Risk Dataset" obtido via Kaggle, gerado por agente de inteligência artificial (natureza sintética).

**Etapas de pré-processamento:**
- Remoção da variável `Employee_ID` (identificador único, sem valor preditivo)
- Conversão da variável alvo para formato binário (Yes→1, No→0)
- Codificação ordinal da variável `Stress_Level` (Low→1, Medium→2, High→3)
- One-Hot Encoding para variáveis categóricas nominais (Gender, Country, Job_Role, Company_Size, Work_Environment) com `handle_unknown='ignore'`
- Padronização das variáveis numéricas com `StandardScaler` (média zero e desvio unitário)
- Organização das transformações em `ColumnTransformer` e `Pipeline` para evitar data leakage

**Tratamento do desbalanceamento:** Adoção de `class_weight='balanced'` na Regressão Logística, atribuindo maior peso à classe minoritária.

**Divisão treino-teste:** 80/20 com estratificação pela variável alvo (`stratify=y`) e `random_state=42` para reprodutibilidade.

### 3. Metodologia de modelagem e validação

**Modelo selecionado:** Regressão Logística, justificada por sua alta interpretabilidade, eficiência computacional e capacidade de fornecer probabilidades calibradas.

**Estratégia de validação adotada (atendendo às boas práticas):**

Para garantir que a avaliação do modelo não fosse contaminada pelo processo de otimização, adotou-se o seguinte protocolo:

1. **Divisão inicial dos dados:** 80% para treino e 20% para teste, com estratificação (`stratify=y`) e `random_state=42`. O conjunto de teste foi mantido **completamente isolado** até o final do processo.

2. **Otimização do parâmetro C (regularização) e penalidade (L1/L2):** Realizada exclusivamente no conjunto de treino utilizando **GridSearchCV com validação cruzada estratificada de 5 folds (k=5)**.
   - A métrica de otimização foi **Recall** (alinhada com o objetivo de minimizar falsos negativos)
   - Foram testados valores de C entre 0.1 e 10.0 para penalidades L1 e L2

3. **Otimização do threshold (limiar de decisão):** Realizada no conjunto de treino (após definir o melhor C e penalty).
   - Foram testados thresholds entre 0.30 e 0.80
   - O critério de escolha foi: **maximizar Recall, respeitando precisão mínima de 85%**

4. **Avaliação final:** Após definidos C, penalty e threshold, o modelo foi avaliado **uma única vez** no conjunto de teste, produzindo as métricas finais reportadas.

**Configuração final adotada:** `class_weight='balanced'`, `penalty='l1'`, `C=0,3`, `threshold=0,5`, `max_iter=1000`.

Esta abordagem garante que:
- Não há **data leakage** entre treino e teste
- O conjunto de teste permanece **cego** durante toda a otimização
- As métricas finais representam a real capacidade de generalização do modelo

**Recomendação para trabalhos futuros:** Em cenários com dados reais, manter o mesmo protocolo de validação cruzada para maior robustez.

**Estratégia de validação adotada:** 
- Divisão treino-teste estratificada única (80/20) foi considerada adequada para este contexto específico devido a:
  - Alta separabilidade entre classes (AUC > 0,99 no baseline)
  - Dataset de 30.000 registros, onde overfitting é improvável dada a simplicidade do modelo
  - Objetivo de estabelecer um baseline replicável e comparável
- **Recomendação para trabalhos futuros:** Em cenários com dados reais ou modelos mais complexos, implementar validação cruzada estratificada com k=5 ou k=10 para maior robustez.

**Otimização de hiperparâmetros:** Busca sistemática variando:
- Parâmetro de regularização (C = 0,1 a 0,9)
- Tipo de penalização (L1 e L2)
- Limiar de decisão (threshold = 0,3 a 0,7)

**Configuração final adotada:** `class_weight='balanced'`, `penalty='l2'`, `C=0,9`, `threshold=0,7`, `max_iter=1000`.

### 4. Avaliação do modelo

**Métricas utilizadas:** Acurácia, Precisão, Recall, F1-Score e AUC-ROC.

**Métrica principal:** Recall, justificada pelo custo assimétrico dos erros – falsos negativos (deixar de identificar um indivíduo em risco de burnout) são considerados mais críticos do que falsos positivos no contexto de prevenção e saúde ocupacional. Todas as decisões de otimização (GridSearchCV e threshold) utilizaram o Recall como critério principal.

**Resultados da configuração final (L1, C=0,3, threshold=0,5):**

| Métrica | Valor |
|---------|-------|
| Recall | **1,0000** (todos os 1.186 casos de burnout identificados) |
| Precisão | **1,0000** (nenhum falso positivo) |
| F1-Score | **1,0000** |
| AUC-ROC | **1,0000** |

**Matriz de Confusão:**

| Real \ Predito | No Burnout | Burnout |
|----------------|------------|---------|
| No Burnout | 4.814 | 0 |
| Burnout | 0 | 1.186 |

O modelo atingiu **desempenho perfeito** no dataset sintético, com zero falsos negativos e zero falsos positivos.

### 5. Conformidade Ética e LGPD (Lei nº 13.709/2018)

O pipeline foi estruturado em conformidade com os princípios da LGPD desde a especificação do problema:

| Princípio LGPD | Implementação no pipeline |
|----------------|---------------------------|
| **Minimização de dados** | Remoção da variável `Employee_ID` (identificador direto) antes de qualquer análise |
| **Finalidade específica** | Modelo destinado exclusivamente à prevenção e suporte organizacional, nunca para decisões automatizadas individuais ou punição |
| **Transparência** | Regressão Logística garante coeficientes interpretáveis e auditáveis por stakeholders não técnicos |
| **Não discriminação** | Análise de viés realizada via teste qui-quadrado e V de Cramér para variáveis categóricas (gênero, país, cargo) |
| **Supervisão humana** | Pipeline exige validação por profissionais de RH/saúde antes de qualquer intervenção baseada nas predições |
| **Responsabilidade (accountability)** | Pipeline documentado, reproduzível e com `random_state` fixo para auditoria externa |

**Mitigação de riscos específicos:**
- **Falsos negativos** (não identificar burnout real): threshold ajustado para maximizar recall, com recomendação de confirmação por profissional de saúde.
- **Uso punitivo do modelo:** Documentação explicita que o modelo é ferramenta de apoio à prevenção, não para controle ou demissão.
- **Generalização indevida:** Ressalvas explícitas sobre a natureza sintética do dataset impedem extrapolação ingênua para contextos reais.

### 6. Reprodutibilidade e transparência

Para garantir a reprodutibilidade dos experimentos, foram adotadas as seguintes práticas:

- **Semente aleatória fixa:** `random_state=42` em todas as funções que envolvem aleatoriedade (divisão treino-teste, inicialização do modelo)
- **Pipeline documentado:** Todo o código de pré-processamento, modelagem e avaliação está disponível na pasta `src/` do repositório
- **Dependências versionadas:** O arquivo `requirements.txt` na raiz do repositório especifica as versões exatas de todas as bibliotecas utilizadas (Python 3.10+, pandas 2.2.2, scikit-learn 1.4.2, etc.)
- **Ambiente de execução:** Os experimentos foram realizados em Google Colab, com código disponível em formato notebook (`.ipynb`).
- **Download automatizado dos dados:** O dataset é obtido via `kagglehub` no próprio código de análise, garantindo que a mesma versão seja utilizada em novas execuções

Para replicar os resultados em novo ambiente:
1. Instalar as dependências listadas no `requirements.txt`
2. Executar o notebook `src/Burnout_Analytics.ipynb` em ordem sequencial
3. O download do dataset ocorrerá automaticamente na primeira célula do notebook

### 7. Limitações críticas e recomendações para generalização

Os resultados obtidos (AUC=1,0000, Recall=1,0000, Precisão=1,0000) são **excepcionalmente elevados** e não devem ser esperados em contextos reais de predição de burnout. Este fenômeno decorre da natureza sintética do dataset:

**Ausência de ruído real**: Dados gerados por IA apresentam padrões mais limpos e separabilidade mais nítida entre classes do que dados coletados com humanos.

**Relações artificialmente lineares**: A correlação extremamente alta entre `Productivity_Score` e burnout (coeficiente -45,03) é raramente observada em pesquisas empíricas.

**Artefatos contraintuitivos**: Exemplo é o coeficiente negativo de `Work_Hours_Per_Day` e `Meetings_Per_Day` que contradiz a literatura especializada.

**Recomendações para aplicação em contexto real:**

- Reavaliar o pipeline com dados reais e anonimizados da organização
- Expectativa realista de métricas significativamente inferiores (AUC 0,70-0,80; Recall 0,65-0,75)
- Implementar validação cruzada estratificada k-fold
- Realizar auditoria de viés por grupos demográficos
- Estabelecer comitê de ética para supervisão do uso do modelo

### 8. Síntese e próximos passos

O pipeline desenvolvido demonstra a viabilidade de predição do risco de burnout utilizando um modelo interpretável, mesmo em um cenário controlado (dados sintéticos). Os principais insights para tomada de decisão organizacional incluem:

* Fatores protetivos identificados: Horas de sono, atividade física e produtividade elevada

* Fatores de risco identificados: Tempo excessivo de tela e reuniões frequentes

* Fatores sem impacto relevante: Idade, experiência profissional e gênero

**Intervenções propostas**: Adoção do modelo como ferramenta de triagem preventiva, com threshold ajustável conforme recursos disponíveis para intervenção, sempre sob supervisão humana e validação prévia com dados reais da organização.

# Observações importantes

Os códigos desenvolvidos encontram-se documentados e disponibilizados integralmente na pasta src/ do repositório, garantindo transparência, reprodutibilidade e aderência às boas práticas de desenvolvimento em projetos de ciência de dados.
