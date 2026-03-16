# Conhecendo os dados

Nesta seção, deverá ser registrada uma detalhada análise descritiva e exploratória sobre a base de dados selecionada na Etapa 1 com o objetivo de compreender a estrutura dos dados, detectar eventuais _outliers_ e também, avaliar/detectar as relações existentes entre as variáveis analisadas.

Para isso, sugere-se que sejam utilizados cálculos de medidas de tendência central, como média, mediana e moda, para entender a centralidade dos dados; sejam exploradas medidas de dispersão como desvio padrão e intervalos interquartil para avaliar a variabilidade dos dados; sejam utilizados gráficos descritivos como histogramas e box plots, para representar visualmente as características essenciais dos dados, pois essas visualizações podem facilitar a identificação de padrões e anomalias; sejam analisadas as relações entre as variáveis por meio de análise de correlação, gráficos de dispersões, mapas de calor, entre outras técnicas. 

Inclua nesta seção, gráficos, tabelas, trechos de código e demais artefatos que você considere relevantes para entender os dados com os quais você irá trabalhar.  Além disso, inclua e comente os trechos de código mais relevantes desenvolvidos para realizar suas análises. Na pasta "src", inclua o código fonte completo.

# Conhecendo os dados

Nesta seção é apresentada uma análise descritiva e exploratória da base de dados **Work Productivity & Burnout Risk Dataset**. O objetivo dessa análise é compreender a estrutura dos dados, identificar possíveis outliers e investigar relações existentes entre as variáveis presentes no dataset.

A análise exploratória de dados (*Exploratory Data Analysis – EDA*) é uma etapa fundamental em projetos de ciência de dados e aprendizado de máquina, pois permite entender o comportamento das variáveis antes da construção de modelos preditivos. A partir dessa etapa é possível identificar padrões relevantes, inconsistências nos dados e possíveis relações entre fatores que influenciam o fenômeno estudado.

O dataset analisado contém **30.000 registros e 17 variáveis**, incluindo informações demográficas, ocupacionais e relacionadas ao estilo de vida dos profissionais analisados. A variável alvo do estudo é **Burnout_Risk**, que indica se o indivíduo apresenta ou não risco de burnout.

Para realizar a análise exploratória foram utilizadas medidas estatísticas de tendência central (média, mediana e moda), medidas de dispersão (desvio padrão e intervalo interquartil), além de técnicas de visualização de dados, como histogramas, box plots e mapas de calor de correlação.

---

## Descrição dos achados

A partir da análise descrita e exploratória realizada, descreva todos os achados considerados relevantes para o contexto em que o trabalho se insere. Por exemplo: com relação à centralidade dos dados algo chamou a atenção? Foi possível identificar correlação entre os atributos? Que tipo de correlação (forte, fraca, moderada)? 

## Carregamento e inspeção inicial dos dados

A primeira etapa da análise consistiu em carregar o dataset e inspecionar sua estrutura.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("Work_Productivity.csv")

df.head()
```

Em seguida, foram verificadas informações gerais sobre o dataset.

```
df.info()
```
Essa inspeção inicial permitiu identificar o tipo de cada variável e confirmar a estrutura do dataset.

Foi observado que o conjunto de dados possui 30.000 registros e 17 colunas, contendo variáveis numéricas e categóricas.

## Verificação de valores ausentes

Uma etapa importante da análise exploratória consiste em verificar a presença de valores ausentes.

```
df.isnull().sum()
```

O resultado mostrou que não há valores ausentes no dataset, indicando boa qualidade estrutural dos dados e reduzindo a necessidade de etapas adicionais de tratamento.

## Medidas de tendência central

Para compreender o comportamento das variáveis numéricas foram calculadas medidas de tendência central.

```
df.describe()
```

Algumas estatísticas relevantes observadas foram:

| Variável | Média |	Mediana | Mínimo |	Máximo |
|---|----|----|----|---|
| ``Age`` |~38 anos |	38 | 22 |	54 |
| ``Work_Hours_Per_Day`` |	~7.1 horas	| 7 |	4 |	10 |
| ``Meetings_Per_Day``|	~3.2 |	3 |	0 |	7 |
| ``Sleep_Hours``|	~6.5 horas |	6 | 4 |	9 |
| ``Screen_Time_Hours``|	~9.3 horas |	9	| 5 |	14 |
| ``Exercise_Hours_Per_Week``|	~2.4 horas |	2 |	0 |	6 |
| ``Productivity_Score``|	~71	| 72 |	30	| 100 |

### Interpretação

Observa-se que a média de horas de sono (6,5) está abaixo da recomendação média para adultos, que geralmente varia entre 7 e 8 horas por noite. Além disso, o tempo médio de exposição a telas (9,3 horas) é relativamente elevado, refletindo a realidade de ambientes profissionais altamente digitalizados.

Esses fatores podem contribuir para níveis mais elevados de estresse ocupacional e estão frequentemente associados ao risco de burnout.

## Medidas de dispersão

Para avaliar a variabilidade dos dados foram analisadas medidas de dispersão, como desvio padrão e quartis.

Essas medidas permitem compreender o grau de heterogeneidade entre os indivíduos analisados.

Variáveis como ``Screen_Time_Hours`` e ``Productivity_Score`` apresentaram maior dispersão, indicando que os indivíduos da amostra apresentam comportamentos bastante variados em relação ao tempo de exposição digital e produtividade.

## Distribuição das variáveis numéricas

Histogramas foram utilizados para visualizar a distribuição das variáveis numéricas.

``
df.hist(figsize=(12,10))
plt.show()
``

A análise das distribuições revelou alguns padrões importantes:

``Work_Hours_Per_Day`` apresenta concentração entre 6 e 9 horas de trabalho diário.

``Sleep_Hours`` apresenta distribuição levemente assimétrica, com menor frequência de indivíduos que dormem mais de 8 horas.

``Screen_Time_Hours`` apresenta valores elevados, indicando alta exposição digital.

Esses resultados refletem características comuns do ambiente de trabalho contemporâneo, marcado pela intensa utilização de dispositivos digitais.

## Detecção de outliers

Para identificar possíveis valores extremos foram utilizados gráficos de boxplot.

``
plt.figure(figsize=(12,6))
sns.boxplot(data=df[['Work_Hours_Per_Day','Sleep_Hours','Screen_Time_Hours']])
plt.show()
``

A análise visual indica alguns valores extremos, como indivíduos que apresentam menos de 5 horas de sono por noite ou mais de 13 horas de exposição a telas.

Esses casos podem representar perfis com maior vulnerabilidade ao estresse ocupacional.



## Ferramentas utilizadas

Existem muitas ferramentas diferentes que podem ser utilizadas para fazer a análise dos dados. Nesta seção, descreva as ferramentas/tecnologias utilizadas e sua aplicação. Vale destacar que, preferencialmente, as análises deverão ser realizadas utilizando a linguagem de programação Python.


