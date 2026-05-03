# Preparação dos dados

Nesta etapa, foram aplicadas técnicas de pré-processamento com o objetivo de adequar o conjunto de dados ao treinamento do modelo de aprendizado de máquina, garantindo consistência, robustez e reprodutibilidade dos resultados.

## Limpeza dos Dados
Inicialmente, foi realizada a etapa de limpeza dos dados, na qual se confirmou a ausência de valores faltantes no dataset, conforme identificado na análise exploratória. Embora o dataset original não apresente valores ausentes, optou-se por manter etapas de imputação no pipeline, utilizando a mediana para variáveis numéricas e a moda para variáveis categóricas. Essa decisão foi adotada visando aumentar a robustez do modelo e sua capacidade de generalização para cenários futuros com dados incompletos. 

Em relação à detecção de valores extremos, embora tenham sido observados alguns perfis atípicos — como indivíduos com baixa quantidade de horas de sono ou elevado tempo de exposição a telas —, a análise pelo método do intervalo interquartil (IQR) indicou que tais valores não configuram outliers estatísticos. Assim, optou-se por manter todos os registros, considerando sua relevância potencial para o fenômeno estudado.

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
| Recall	 | Capacidade de identificar corretamente os casos positivos | **Métrica principal** - minimizar falsos negativos é prioritário |
| F1-score | Média harmônica entre precisão e recall | Equilíbrio entre sensibilidade e especificidade |
| AUC-ROC | Capacidade de discriminação entre classes | Avalia qualidade intrínseca do modelo independente do threshold |

Dentre essas métricas, o **recall foi definido como o principal critério de avaliação**, uma vez que, no contexto do problema, o erro mais crítico consiste na ocorrência de falsos negativos — ou seja, situações em que o modelo deixa de identificar indivíduos em risco de burnout. Tal falha pode comprometer ações preventivas e impactar negativamente a saúde dos trabalhadores e o desempenho organizacional.

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

Tabela I – Resultados dos testes preliminares (Threshold fixo = 0,5)

| Teste           | Class Weight | Penalização | C  | Threshold | Accuracy | Precision | Recall  | F1-Score | AUC-ROC  |
|-----------------|-------------|-------------|-----|----------|-----------|---------|----------|----------|----------|
| No Balanced     | None        | L2          | 1.0 |      0.5 | 0.999000 | 0.997470  | 0.997470| 0.997470 | 0.999990 |
| Balanced L2     | Balanced    | L2          | 1.0 |      0.5 | 0.990167 | 0.952610  | 1.000000| 0.975730 | 0.999993 |
| L2 C=0.1        | Balanced    | L2          | 0.1 |      0.5 | 0.978333 | 0.901216  | 1.000000| 0.948042 | 0.999651 |
| L2 C=10         | Balanced    | L2          | 10.0|      0.5 | 0.999667 | 0.998316  | 1.000000| 0.999158 | 1.000000 |
| L1 C=1.0        | Balanced    | L1          | 1.0 |      0.5 | 1.000000 | 1.000000  | 1.000000| 1.000000 | 1.000000 |

### Análise dos resultados preliminares

O modelo apresenta desempenho extremamente elevado em todas as configurações testadas, com valores de recall próximos ou iguais a 1.0 na maioria dos experimentos. Esse comportamento indica que o modelo já possui alta capacidade de identificação da classe positiva, independentemente da configuração adotada.

A utilização de `class_weight='balanced'` não resultou em ganhos expressivos de recall, mas impactou negativamente a precisão, indicando aumento na ocorrência de falsos positivos. Esse comportamento é esperado, uma vez que o balanceamento torna o modelo mais sensível à classe minoritária.

A variação do parâmetro de regularização (C) e o tipo de penalização (L1 e L2) não resultou em mudanças drásticas do desempenho global, indicando que o problema apresenta alta separabilidade entre as classes. Este resultado sugere que o dataset, possivelmente por sua natureza sintética, apresenta padrões facilmente capturáveis pelo modelo.

## Otimização de parâmetros - testes Adicionais

### Teste de Thresholds (0,3 a 0,7)

Para avaliar o impacto do limiar de decisão no desempenho do modelo, foram testados diferentes thresholds, mantendo-se `class_weight='balanced'`, `C=1.0` e `penalty='l2'`.

<img width="689" height="652" alt="Screen Shot 2026-04-29 at 16 30 40" src="https://github.com/user-attachments/assets/be70ff0c-4c01-49d9-863c-ae8eb8c7901f" />

**Análise**: O threshold de 0,70 apresentou o melhor F1-Score (0,9975) e será utilizado na configuração final do modelo.

### Teste de Regularização (C de 0,1 a 0,9)

Para avaliar o impacto do parâmetro de regularização C, foram testados diferentes valores, mantendo-se `class_weight='balanced'`, `threshold=0,5` e `penalty='l2'`.

<img width="695" height="620" alt="Screen Shot 2026-04-29 at 22 51 16" src="https://github.com/user-attachments/assets/574de72c-8012-404a-b9ef-38c225e262d4" />

**Análise**: Observa-se uma tendência clara: **quanto maior o valor de C (menos regularização), melhor o desempenho do modelo**. O valor testado mais alto, **C=0,9**, apresentou o melhor F1-Score (0,9749) e será utilizado como referência para os experimentos combinados com threshold.

### Experimentos Combinados ( C + Threshold )

Para identificar a melhor combinação possível, foi realizada uma busca sistemática variando C e threshold simultaneamente:

<img width="353" height="335" alt="Screen Shot 2026-04-29 at 23 01 26" src="https://github.com/user-attachments/assets/4594f4f2-b224-4504-a0ae-3e5d536e6eae" />

**Melhor combinação encontrada**: C = 0,9 e Threshold = 0,7 (F1-Score ≈ 0,9971).

### Modelo Final Otimizado

Dessa forma, a configuração final adotada foi:

<img width="435" height="466" alt="Screen Shot 2026-04-29 at 23 00 37" src="https://github.com/user-attachments/assets/86fcebe6-f7bb-4c9e-adc5-0a6e1891f184" />

### Interpretação do Modelo Final

O modelo final otimizado (C=0,9, threshold=0,7) apresenta desempenho excepcional no dataset de teste, com recall de 0,9975 (apenas 3 casos de burnout não detectados em 1.200) e precisão de 0,9966 (apenas 0,34% de falsos positivos). 

A escolha do threshold 0,7 (acima do padrão 0,5) reflete uma decisão consciente de priorizar a **redução de falsos positivos** em detrimento de uma pequena perda de sensibilidade (de 100% para 99,75%). Esta configuração é mais adequada para cenários com recursos limitados para intervenção.

O AUC-ROC de 1,0 indica separabilidade perfeita entre as classes, mas este resultado deve ser interpretado com cautela, pois reflete a **natureza sintética do dataset**. Em contextos reais, o desempenho tende a ser inferior.

Todos os experimentos foram executados utilizando o mesmo pipeline de pré-processamento e divisão de dados, garantindo comparabilidade entre os testes. As variações foram aplicadas exclusivamente nos parâmetros do modelo, mantendo-se constantes os demais elementos do processo. Essa abordagem assegura que as diferenças observadas nos resultados sejam decorrentes apenas das configurações testadas.

### Avaliação da Curva ROC no Modelo Final

<img width="564" height="439" alt="Screen Shot 2026-04-29 at 23 10 11" src="https://github.com/user-attachments/assets/40164178-a799-4b42-9da9-448599f09bc8" />

O modelo final atingiu AUC = 1,0000, indicando separabilidade perfeita entre as classes no conjunto de teste.

### Análise dos Coeficientes do Modelo Final

Principais variáveis que AUMENTAM o risco de burnout (coeficiente positivo):

| Variável | Coeficiente | Interpretação |
|----|----|----|
| Screen_Time_Hours	| +0,2710 |	Mais tempo de tela aumenta o risco |
| Meetings_Per_Day	| +0,2314 |	Reuniões excessivas são fator de risco |
| Country_UK | +0,2252 |	Reino Unido apresenta maior prevalência |

Principais variáveis que REDUZEM o risco de burnout (coeficiente negativo):

| Variável	| Coeficiente |	Interpretação |
|----|----|----|
| Productivity_Score |	-22,6676 |	Fator protetivo mais forte |
| Sleep_Hours	| -0,8385	| Dormir bem reduz significativamente o risco |
| Work_Hours_Per_Day |	-0,6795	| **Contraintuitivo** (ver seção dedicada) |
| Exercise_Hours_Per_Week |	-0,5282 |	Atividade física protege contra burnout |
| Internet_Speed_Mbps	| -0,3811 |	Boa conexão reduz estresse |

**Resultado contraintuitivo**: O coeficiente negativo de `Work_Hours_Per_Day` sugere que trabalhar mais horas reduz o risco de burnout, o que contradiz a literatura especializada. Este é provavelmente um artefato do *dataset sintético*, devendo ser interpretado com cautela.

## Decisão do Modelo

Com base nos resultados obtidos, optou-se por utilizar a configuração com class_weight='balanced', penalização L2, C=0,9 e threshold=0,7, considerando seu desempenho superior nas métricas avaliadas (acima de 99,6% em todas as métricas, com AUC-ROC perfeito de 1,0000).

A escolha está alinhada ao objetivo central do projeto, que consiste na identificação de indivíduos em risco de burnout, minimizando a ocorrência de falsos negativos.

**Avaliação final:** O modelo apresentou uma performance excelente como ferramenta de triagem e prevenção, desde que utilizado com supervisão humana e validação prévia em dados reais da organização.

Ressalva importante: O desempenho quase perfeito obtido (acima de 99,6% em todas as métricas) é um fenômeno extremamente raro em dados reais. Este resultado reflete a natureza sintética do dataset (gerado por IA) e não deve ser generalizado para contextos reais sem validação adicional.

## Análise de Interpretabilidade: Fatores que Influenciam o Burnout

Um dos principais diferenciais da Regressão Logística é sua capacidade de fornecer coeficientes interpretáveis, permitindo identificar quais variáveis mais contribuem para o aumento ou redução do risco de burnout. A tabela a seguir apresenta as 10 variáveis com maior impacto absoluto no modelo final (C=0,9, threshold=0,7).

<img width="545" height="202" alt="Screen Shot 2026-04-29 at 20 01 21" src="https://github.com/user-attachments/assets/cf4f584d-e3ad-4993-8ecc-39ad05c407e0" />

### Análise dos Resultados

**Fatores Protetivos (Reduzem o Risco):**
- **Productivity_Score** (coeficiente -22,67): O fator mais importante do modelo. Quanto maior a produtividade, menor o risco de burnout. Este achado sugere que profissionais engajados e produtivos apresentam maior resiliência ao estresse ocupacional.

- **Sleep_Hours** (coeficiente -0,84): Dormir bem é o segundo fator protetivo mais relevante. Cada hora adicional de sono está associada à redução significativa do risco, corroborando a literatura sobre a importância do descanso para a saúde mental.

- **Exercise_Hours_Per_Week** (coeficiente -0,53): A prática regular de atividades físicas demonstra efeito protetivo, reforçando a importância de hábitos saudáveis.

- **Internet_Speed_Mbps** (coeficiente -0,38): Velocidades mais altas de internet associam-se a menor risco, possivelmente refletindo melhores condições de infraestrutura e menor estresse tecnológico.

**Fatores de Risco (Aumentam o Risco):**
- **Screen_Time_Hours** (coeficiente +0,27): Maior tempo de exposição a telas está associado ao aumento do risco, sugerindo a necessidade de pausas regulares.

- **Meetings_Per_Day** (coeficiente +0,23): Reuniões excessivas contribuem para o burnout, indicando que a otimização do tempo coletivo é uma estratégia de prevenção.

- **Países (UK, USA, China)**: Fatores geográficos apresentam associação com o risco, possivelmente refletindo diferenças culturais ou organizacionais.

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

**Modelo selecionado:** Regressão Logística com penalização L2, justificada por sua alta interpretabilidade, eficiência computacional e capacidade de fornecer probabilidades calibradas.

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

**Métrica principal:** Recall, justificada pelo custo assimétrico dos erros – falsos negativos (deixar de identificar um indivíduo em risco de burnout) são considerados mais críticos do que falsos positivos no contexto de prevenção e saúde ocupacional.

**Resultados da configuração final:**
- Recall: 0,9975 (apenas 3 casos de burnout não detectados em 1.200)
- Precisão: 0,9966
- F1-Score: 0,9971
- AUC-ROC: 1,0000

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
2. Executar o notebook `src/Colab_Burnout_2.ipynb` em ordem sequencial
3. O download do dataset ocorrerá automaticamente na primeira célula do notebook

### 7. Limitações críticas e recomendações para generalização

Os resultados obtidos (AUC=1,0000, recall=0,9975) são excepcionalmente elevados e não devem ser esperados em contextos reais de predição de burnout. Este fenômeno decorre da natureza sintética do dataset:

**Ausência de ruído real**: Dados gerados por IA apresentam padrões mais limpos e separabilidade mais nítida entre classes do que dados coletados com humanos.

**Relações artificialmente lineares**: A correlação de -0,69 entre `Productivity_Score` e `Burnout` é raramente observada em pesquisas empíricas.

**Artefatos contraintuitivos**: Exemplo é o coeficiente negativo de `Work_Hours_Per_Day` que contradiz a literatura especializada.

Recomendações para aplicação em contexto real:

- Reavaliar o pipeline com dados reais e anonimizados da organização

- Expectativa realista de métricas significativamente inferiores (AUC 0,70-0,80; recall 0,65-0,75)

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
