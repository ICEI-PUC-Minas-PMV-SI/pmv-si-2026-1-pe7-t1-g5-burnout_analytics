# Conhecendo os dados

Nesta seção é apresentada uma análise descritiva e exploratória da base de dados **Work Productivity & Burnout Risk Dataset**. O objetivo dessa análise é compreender a estrutura dos dados, identificar possíveis outliers e investigar relações existentes entre as variáveis presentes no dataset.

A análise exploratória de dados (*Exploratory Data Analysis – EDA*) é uma etapa fundamental em projetos de ciência de dados e aprendizado de máquina, pois permite entender o comportamento das variáveis antes da construção de modelos preditivos. A partir dessa etapa é possível identificar padrões relevantes, inconsistências nos dados e possíveis relações entre fatores que influenciam o fenômeno estudado.

O dataset analisado contém **30.000 registros e 17 variáveis**, incluindo informações demográficas, ocupacionais e relacionadas ao estilo de vida dos profissionais analisados. A variável alvo do estudo é **Burnout_Risk**, que indica se o indivíduo apresenta ou não risco de burnout.

O conjunto de dados foi obtido por meio da plataforma Kaggle e, conforme a documentação disponibilizada, trata-se de um dataset gerado por um agente de inteligência artificial. Dessa forma, os dados não refletem percepções subjetivas de respondentes humanos nem estão sujeitos a vieses decorrentes de instrumentos de coleta, interpretação de especialistas ou inconsistências comuns em levantamentos técnicos reais. Por outro lado, essa característica implica que o dataset representa um cenário simulado, o que deve ser considerado na interpretação dos resultados e na generalização das conclusões.

Para realizar a análise exploratória foram utilizadas medidas estatísticas de tendência central (média, mediana e moda), medidas de dispersão (desvio padrão e intervalo interquartil), além de técnicas de visualização de dados, como histogramas, box plots e mapas de calor de correlação.

## Carregamento e inspeção inicial dos dados

A primeira etapa da análise consistiu em carregar o dataset e inspecionar sua estrutura.

```python

# IMPORTAÇÃO DAS BIBLIOTECAS
import pandas as pd                        # Manipulação de dados em formato de tabelas (DataFrames)
import os                                  # Operações com caminhos de arquivos (path, diretórios)
import kagglehub                           # Download de datasets diretamente do Kaggle
import matplotlib.pyplot as plt            # Criação de gráficos básicos
import seaborn as sns                      # Visualização estatística mais avançada (baseado em matplotlib)

# CARREGAMENTO DO DATA SET
path = kagglehub.dataset_download("shree0910/work-productivity-and-burnout-risk-dataset")
csv_file_path = os.path.join(path, "Work Productivity.csv")

# Lê o arquivo CSV e carrega os dados em um DataFrame do pandas
df = pd.read_csv(csv_file_path)
```

Em seguida, foram verificadas informações gerais sobre o dataset.

```
df.info()
```
Essa inspeção inicial permitiu identificar o tipo de cada variável e confirmar a estrutura do dataset.

Foi observado que o conjunto de dados possui 30.000 registros e 17 colunas, contendo variáveis numéricas e categóricas.

## Estrutura dos Dados

A verificação inicial do tipo de cada variável mostrou que o dataset contém tanto variáveis numéricas (Sleep_Hours, Screen_Time_Hours, Age, etc.) quanto categóricas (Gender, Country, Job_Role, etc.)


<img width="507" height="383" alt="Screen Shot 2026-03-23 at 20 53 43" src="https://github.com/user-attachments/assets/3f552957-cea1-4c37-8250-7e8cedf7efe4" />


## Verificação de valores ausentes

Uma etapa importante da análise exploratória consiste em verificar a presença de valores ausentes.

```
df.isnull().sum()
```

<img width="167" height="459" alt="Screen Shot 2026-03-23 at 21 04 40" src="https://github.com/user-attachments/assets/14497db2-278f-4374-b42a-79a81d31af14" />

O resultado mostrou que não há valores ausentes no dataset, indicando boa qualidade estrutural dos dados e reduzindo a necessidade de etapas adicionais de tratamento.

## Tratamento das variáveis e preparação dos dados

Antes da realização das análises estatísticas, foi necessário realizar algumas transformações nas variáveis, com o objetivo de adequar o dataset para análise exploratória e posterior modelagem.

Inicialmente, a variável Employee_ID foi removida do conjunto de dados, por se tratar de um identificador único para cada registro. Esse tipo de variável não possui significado analítico e não contribui para a identificação de padrões ou relações entre variáveis.

```
# TRATAMENTO DAS VARIÁVEIS
df = df.drop(columns=['Employee_ID']) # Employee_ID foi removido por ser identificador
```

Em seguida, foram realizadas transformações em variáveis categóricas ordinais e na variável alvo, de modo a convertê-las para formato numérico, permitindo sua utilização em análises estatísticas.

A variável Burnout_Risk, originalmente categórica (“Yes”/“No”), foi convertida para valores binários (1 e 0). Essa transformação é necessária para viabilizar análises quantitativas, como correlação, além de ser uma etapa padrão em problemas de classificação supervisionada.

```
# Trata a variável Burnout_Risk para que os valores sejam numéricos
df["Burnout_Risk"] = df["Burnout_Risk"].map({
    "Yes": 1,
    "No": 0
})
```

A variável Stress_Level, que representa níveis ordinais de estresse (“Low”, “Medium” e “High”), foi convertida para uma escala numérica (1, 2 e 3, respectivamente). Essa transformação preserva a ordem natural entre as categorias, permitindo sua utilização em análises que consideram intensidade ou progressão.

```
# Trata a variável Stress_Level para que os valores sejam numéricos
df["Stress_Level"] = df["Stress_Level"].map({
    "Low": 1,
    "Medium": 2,
    "High": 3
})
```

É importante destacar que, apesar dessa codificação numérica, Stress_Level continua sendo uma variável ordinal, e sua interpretação deve respeitar essa característica. Dessa forma, análises que assumem relações lineares estritas devem ser interpretadas com cautela.

Após o tratamento das variáveis, foi realizada a separação entre variáveis numéricas e categóricas:

```
# VARIÁVEIS GERAIS
colsNum = df.select_dtypes(include='number').columns # Variável com todas as colunas numéricas
colsCat = df.select_dtypes(include='object').columns # Variável com todas as colunas categóricas
```

Variáveis numéricas: utilizadas em análises estatísticas, como medidas de tendência central, dispersão e correlação.
Variáveis categóricas: utilizadas em análises de frequência e proporção, especialmente em relação à variável alvo.

Essa separação foi realizada automaticamente com base nos tipos de dados do DataFrame, garantindo maior consistência e escalabilidade da análise.

## Estatísticas descritivas iniciais

Como etapa inicial da análise exploratória, foram calculadas estatísticas descritivas para todas as variáveis numéricas do dataset.
```
df[colsNum].describe()
```

<img width="1503" height="266" alt="Screen Shot 2026-04-21 at 21 42 50" src="https://github.com/user-attachments/assets/39c00223-246e-4a86-93bc-bb01a1cd0bca" />

Essas estatísticas incluem, média (mean), desvio padrão (std), valores mínimos e máximos, quartis (25%, 50% e 75%). Essas medidas permitem uma visão geral da distribuição dos dados, auxiliando na identificação de padrões, dispersão e possíveis valores extremos. A análise das estatísticas descritivas revelou que:

* A idade média dos indivíduos é de aproximadamente 38 anos, com distribuição relativamente equilibrada entre 22 e 54 anos.
* O tempo médio de experiência profissional é de 8 anos, apresentando alta variabilidade, o que indica a presença de indivíduos tanto iniciantes quanto experientes.
* A média de horas de trabalho diárias é de aproximadamente 7 horas, com baixa dispersão, sugerindo uma distribuição relativamente homogênea.
* O tempo médio de sono é de 6,5 horas, abaixo da recomendação usual para adultos, o que pode estar associado a fatores de estresse e saúde ocupacional.
* O tempo médio de exposição a telas é elevado (aproximadamente 9,5 horas), refletindo ambientes de trabalho altamente digitalizados.
* A variável Burnout_Risk apresenta média de 0,197, indicando que cerca de 19,7% dos indivíduos estão classificados com risco de burnout, o que confirma o desbalanceamento da variável alvo.

De forma geral, observa-se que variáveis relacionadas ao estilo de vida (sono, exercício, tempo de tela) apresentam maior dispersão em comparação com variáveis estruturais (idade, horas de trabalho), o que sugere maior heterogeneidade comportamental entre os indivíduos analisados.

## Medidas de tendência central e dispersão

Para compreender o comportamento das variáveis numéricas, foram calculadas medidas de tendência central (média, mediana e moda) e medidas de dispersão (desvio padrão e intervalo interquartil).

<img width="771" height="352" alt="Screen Shot 2026-04-21 at 21 47 41" src="https://github.com/user-attachments/assets/8517fb05-df84-4d49-a304-6c7970f3de59" />

### Interpretação

A análise das medidas descritivas permite identificar padrões importantes na distribuição das variáveis do dataset.

A variável Age apresenta média e mediana praticamente iguais (≈ 38 anos), indicando uma distribuição aproximadamente simétrica. O intervalo interquartil (30 a 46 anos) sugere concentração dos indivíduos em idade adulta intermediária.

A variável Experience_Years apresenta média (8,0) superior à mediana (6,0), indicando assimetria à direita, ou seja, a presença de indivíduos com muitos anos de experiência que elevam a média. Isso é consistente com a cauda longa observada na distribuição.

A variável Work_Hours_Per_Day apresenta baixa dispersão (desvio padrão ≈ 1,73), indicando que a maioria dos indivíduos trabalha em uma faixa relativamente homogênea entre 5,5 e 8,5 horas diárias.

Já a variável Meetings_Per_Day apresenta maior variabilidade relativa, com valores distribuídos entre 0 e 7 reuniões diárias, indicando diferentes perfis de carga de reuniões entre os indivíduos.

A variável Sleep_Hours apresenta média de aproximadamente 6,5 horas, abaixo da recomendação média para adultos (7 a 8 horas). Esse resultado pode indicar um padrão de privação de sono, frequentemente associado a fatores de estresse ocupacional.

A variável Screen_Time_Hours apresenta média elevada (≈ 9,5 horas) e alta dispersão, refletindo a forte presença de atividades digitais no ambiente de trabalho analisado.

As variáveis Exercise_Hours_Per_Week e Sleep_Hours apresentam dispersão moderada, sugerindo heterogeneidade nos hábitos de vida dos indivíduos.

A variável Productivity_Score apresenta ampla variação (30 a 100), com desvio padrão elevado, indicando grande diversidade de níveis de produtividade na amostra.

A variável Internet_Speed_Mbps também apresenta alta dispersão, o que pode refletir diferentes contextos de infraestrutura tecnológica entre os indivíduos analisados.

A variável Stress_Level, embora representada numericamente, é de natureza ordinal, assumindo valores de 1 (baixo), 2 (médio) e 3 (alto). Sua média próxima de 2 indica predominância de níveis moderados de estresse na amostra.

Por fim, a variável Burnout_Risk é binária e sua média (0,1977) deve ser interpretada como a proporção de indivíduos com risco de burnout, indicando que aproximadamente 19,7% da amostra pertence à classe positiva. Esse resultado confirma o desbalanceamento da variável alvo.

De forma geral, observa-se que variáveis comportamentais, como sono, exercício físico e tempo de exposição a telas, apresentam maior variabilidade em comparação com variáveis estruturais, como idade e horas de trabalho. Esse padrão sugere que fatores relacionados ao estilo de vida podem ter maior heterogeneidade e potencial influência no risco de burnout.

## Detecção de outliers

Para identificar possíveis valores extremos, foram utilizados duas abordagens complementares: 
* Análise visual por meio de gráficos de boxplot
* Análise estatística utilizando o método do intervalo interquartil (IQR)

### Análise visual (Boxplot)

Inicialmente, foi gerado um boxplot considerando todas as variáveis numéricas, com exceção da variável Burnout_Risk, por se tratar de uma variável binária.

```
plt.figure(figsize=(20, 10))
sns.boxplot(data=df[colsOutliers],
    flierprops={
        'marker': 'o',               # formato do marcador
        'markerfacecolor': 'red',    # cor dos outliers
        'markersize': 6              # tamanho dos pontos
    }
)
plt.xticks(rotation=45)
plt.show()
```

<img width="1210" height="703" alt="Screen Shot 2026-04-21 at 21 58 23" src="https://github.com/user-attachments/assets/8c571aa6-7bf0-497d-88a6-abfcd40295e0" />

A análise visual sugere a presença de alguns valores extremos em determinadas variáveis, indicados pelos pontos destacados fora das caixas. No entanto, essa abordagem não permite, por si só, confirmar se esses valores são outliers estatísticos.

### Análise estatística (método IQR - Tukey)

Para complementar a análise visual, foi aplicada uma função baseada no método do intervalo interquartil (IQR), que calcula limites formais para detecção de outliers:

* Limite inferior: Q1 − 1,5 × IQR
* Limite superior: Q3 + 1,5 × IQR

A imagem a seguir apresenta os resultados obtidos para cada variável analisada:

<img width="256" height="605" alt="Screen Shot 2026-04-21 at 22 22 13" src="https://github.com/user-attachments/assets/316dd164-e966-46ea-9f32-7bf0cd4be451" />
<img width="202" height="579" alt="Screen Shot 2026-04-21 at 22 22 45" src="https://github.com/user-attachments/assets/02ba3f6e-d2ee-479a-8570-05d0a25ffc0f" />

Os resultados indicam que a maioria das variáveis não apresenta outliers estatísticos, pois seus valores mínimos e máximos permanecem dentro dos limites calculados.

### Caso específico: Experience_Years

A variável Experience_Years foi a única que apresentou outliers estatísticos.

De acordo com a análise:

Q1 = 2,0
Q3 = 12,0
IQR = 10,0
Limite superior = 27,0

Foram identificados 424 registros acima desse limite, correspondendo aos valores:

28, 29, 30, 31 e 32 anos

Observa-se que esses valores estão concentrados em poucos níveis discretos, o que sugere um padrão específico de geração dos dados, possivelmente relacionado à natureza sintética do dataset.

### Interpretação dos resultados

Apesar de serem classificados como outliers pelo método estatístico, esses valores são plausíveis no contexto profissional; não representam erros ou inconsistências; podem ser relevantes para a análise do fenômeno de burnout.

Além disso, algumas variáveis — como **Sleep_Hours** e **Screen_Time_Hours** — apresentam valores extremos do ponto de vista do domínio (por exemplo, poucas horas de sono ou alto tempo de exposição a telas), mas não são identificadas como outliers pelo método IQR. Isso evidencia a diferença entre:

* Outliers estatísticos, definidos por critérios matemáticos
* Valores críticos de domínio, relevantes para interpretação do problema

### Decisão de tratamento

Diante dos resultados, optou-se por não remover os outliers identificados, uma vez que:

* Não há evidência de erro nos dados;
* Os valores extremos são plausíveis e interpretáveis;
* A remoção poderia comprometer a representatividade da amostra;
* O dataset possui natureza sintética, podendo apresentar padrões artificiais.

A utilização combinada de análise visual e métodos estatísticos permitiu uma avaliação mais robusta dos dados, evitando decisões baseadas exclusivamente em critérios automáticos.

## Distribuição das variáveis numéricas

Histogramas foram utilizados para visualizar a distribuição das variáveis numéricas.

```
df[colsNum].hist(figsize=(12,10))
plt.show()
```
<img width="994" height="836" alt="image" src="https://github.com/user-attachments/assets/8abf35e8-f5a2-461e-acc0-13cbff371a48" />

A análise das distribuições revelou alguns padrões importantes:

``Work_Hours_Per_Day`` apresenta concentração entre 6 e 10 horas de trabalho diário.

``Sleep_Hours`` apresenta distribuição levemente assimétrica, com menor frequência de indivíduos que dormem mais de 8 horas.

``Screen_Time_Hours`` apresenta valores elevados, indicando alta exposição digital.

Esses resultados refletem características comuns do ambiente de trabalho contemporâneo, marcado pela intensa utilização de dispositivos digitais.

## Distribuição das variáveis categóricas

A distribuição das variáveis categóricas também foi analisada utilizando gráficos de barras.

```
sns.countplot(x="Gender", data=df)
plt.show()
```

Observou-se uma distribuição relativamente equilibrada entre os gêneros, o que contribui para reduzir possíveis vieses relacionados a essa variável.

Outras variáveis categóricas analisadas incluem:

``Gender``

<img width="589" height="455" alt="image" src="https://github.com/user-attachments/assets/5493e9a5-9d41-4074-9095-ee40d46721f4" />

``Country``

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/ab005dec-ef25-41a9-9533-68a389546e28" />

``Job_Role``

<img width="598" height="455" alt="image" src="https://github.com/user-attachments/assets/d05383dd-adf7-4c33-a7e9-0aad6d85d100" />

``Company_Size``

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/340fcf7d-7aeb-4b10-8062-265e90402489" />

``Work_Environment``

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/de4a5778-d5d0-4ad2-a58a-595a096dfcde" />

Essas variáveis permitem analisar possíveis diferenças no risco de burnout entre diferentes contextos organizacionais.

## Distribuição da variável alvo

A variável alvo ``Burnout_Risk`` foi analisada para verificar o balanceamento das classes.

```
counts = df["Burnout_Risk"].value_counts()
percent = df["Burnout_Risk"].value_counts(normalize=True) * 100

ax = sns.barplot(x=counts.index, y=counts.values)
ax.set_xticks([0,1])
ax.set_xticklabels(["No", "Yes"])

for i, p in enumerate(ax.patches):
    ax.text(p.get_x() + p.get_width()/2,
            p.get_height(),
            f'{percent[i]:.1f}%',
            ha="center")

plt.show()
```

<img width="569" height="434" alt="image" src="https://github.com/user-attachments/assets/ff7bf00c-6a4e-4d5a-8eef-29391d1863fa" />

Foi observado que aproximadamente **80% dos registros correspondem à classe "No" e 20% à classe "Yes"**.

Essa distribuição caracteriza um desbalanceamento significativo, podendo impactar o desempenho de modelos de classificação, especialmente na identificação da classe minoritária.

## Análise de correlação

Para investigar relações entre as variáveis numéricas foi utilizada a correlação de Pearson.

```
cols = ["Burnout_Risk", "Work_Hours_Per_Day", "Stress_Level", "Sleep_Hours", "Exercise_Hours_Per_Week", "Productivity_Score", "Screen_Time_Hours"]
corr = df[cols].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
```

<img width="692" height="584" alt="image" src="https://github.com/user-attachments/assets/c874b8b2-a0e7-4c30-a7c2-b43b5f0edada" />

A análise do mapa de calor permitiu identificar algumas relações relevantes entre as variáveis do dataset.

Entre as correlações mais relevantes destacam-se:

``Stress_Level`` e ``Burnout_Risk`` apresentam correlação praticamente nula (-0,0042), indicando ausência de relação linear significativa entre as variáveis.

``Work_Hours_Per_Day`` e ``Burnout_Risk`` apresentam correlação negativa moderada (-0,34), sugerindo que o aumento das horas de trabalho está associado a uma leve redução no risco de burnout.

``Sleep_Hours`` e ``Burnout_Risk`` apresentam correlação negativa moderada (-0,44), indicando que maiores horas de sono estão relacionadas à diminuição do risco de burnout.

``Exercise_Hours_Per_Week`` e ``Burnout_Risk`` apresentam correlação negativa fraca (-0,27), sugerindo uma leve tendência de redução do burnout com o aumento da prática de exercícios.

Esses resultados sugerem que maiores horas de sono estão associadas à redução do risco de burnout, enquanto outras variáveis apresentaram relações fracas ou inexistentes. No entanto, essas associações devem ser interpretadas com cautela, uma vez que não indicam causalidade.

---

## Descrição dos achados

A análise exploratória realizada permitiu identificar diversos padrões relevantes no dataset analisado.

Entre os principais achados destacam-se:

- A média de horas de sono observada no dataset é inferior à recomendação média para adultos, o que pode estar associado ao aumento do estresse ocupacional.

- O tempo médio de exposição a telas é elevado, refletindo a forte digitalização do ambiente de trabalho moderno.

- A análise de correlação indicou que não há associação linear significativa entre os níveis de estresse e o risco de burnout.

- Variáveis relacionadas à carga de trabalho, como número de horas trabalhadas, apresentaram correlação negativa moderada com o risco de burnout.

- Variáveis relacionadas ao estilo de vida saudável, como horas de sono e prática de exercícios físicos, apresentaram correlação negativa, com destaque para o sono, que demonstrou associação moderada com a redução do risco de burnout.

Esses resultados indicam a presença de algumas associações entre as variáveis analisadas, especialmente relacionadas a hábitos de vida, porém devem ser interpretados com cautela, uma vez que não implicam relações de causa e efeito.

A análise exploratória também mostrou que o dataset possui boa qualidade estrutural, sem valores ausentes e com distribuição desbalanceada da variável alvo, na qual aproximadamente 80% dos registros pertencem à classe “No” e 20% à classe “Yes”.

Essas características tornam o conjunto de dados adequado para o treinamento de modelos de aprendizado de máquina supervisionado, embora o desbalanceamento entre as classes deva ser considerado nas etapas posteriores do projeto.

---

## Ferramentas utilizadas

A análise exploratória foi realizada utilizando a linguagem de programação Python, executada no ambiente Google Colab, que oferece suporte para análise de dados e desenvolvimento de modelos de aprendizado de máquina.

As principais bibliotecas utilizadas foram:

| Ferramenta |	Aplicação |
|---|---|
| **Python** |	Linguagem de programação utilizada para análise dos dados |
| **pandas** |	Manipulação e estruturação de dados em DataFrames |
| **matplotlib** |	Criação de gráficos e visualizações |
| **seaborn** | Visualização estatística avançada |

Essas ferramentas são amplamente utilizadas em projetos de ciência de dados por permitirem análises eficientes, reprodutíveis e escaláveis.
