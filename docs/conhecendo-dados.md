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

## Estatísticas descritivas

Como etapa inicial da análise exploratória, foram calculadas estatísticas descritivas para todas as variáveis numéricas do dataset.

```
df[colsNum].describe()
```

<img width="1503" height="266" alt="Screen Shot 2026-04-21 at 21 42 50" src="https://github.com/user-attachments/assets/39c00223-246e-4a86-93bc-bb01a1cd0bca" />

**Principais achados:**

Idade (`Age`)
- Média e mediana de aproximadamente 38 anos, indicando distribuição simétrica
- Variação entre 22 e 54 anos, com desvio padrão de cerca de 8 anos
- A população analisada é majoritariamente adulta em idade produtiva

Experiência profissional (`Experience_Years`)
- Média de 8 anos, mas mediana de apenas 6 anos (assimetria à direita)
- Profissionais com muitos anos de experiência (até 32 anos) elevam a média
- Desvio padrão elevado (~7,5 anos) indica grande heterogeneidade

Horas de trabalho (`Work_Hours_Per_Day`)
- Média de aproximadamente 7 horas diárias
- Baixa dispersão (desvio padrão ~1,7 horas)
- Maioria concentrada entre 5,5 e 8,5 horas (intervalo interquartil)

Horas de sono (`Sleep_Hours`)
- Média de 6,5 horas, **abaixo da recomendação** (7-8 horas para adultos)
- Mediana também em 6,5 horas (distribuição simétrica)
- Variação entre 4 e 9 horas, com desvio padrão ~1,4 horas

Tempo de tela (`Screen_Time_Hours`)
- Média elevada de aproximadamente 9,5 horas diárias
- Alta dispersão (desvio padrão ~2,9 horas), variando de 5 a 15 horas
- Reflete ambientes de trabalho altamente digitalizados

Produtividade (`Productivity_Score`)
- Média em torno de 65 pontos (escala 0-100)
- Ampla variação (30 a 100 pontos) com desvio padrão elevado (~20 pontos)
- Grande diversidade de níveis de produtividade na amostra

Risco de burnout (`Burnout_Risk`)
- Média de 0,197, significando que **19,7%** dos indivíduos apresentam risco
- Confirma o **desbalanceamento** da variável alvo (classe minoritária em ~20%)
- Mediana igual a 0, reforçando a predominância da classe "sem burnout"

**Padrões gerais observados**: Variáveis comportamentais (sono, tempo de tela, produtividade) apresentam maior dispersão em comparação com variáveis estruturais (idade, horas de trabalho). Isso sugere maior heterogeneidade nos hábitos de vida dos indivíduos, indicando que fatores comportamentais podem ter maior potencial de influência sobre o risco de burnout do que características demográficas.

## Detecção de outliers

Para identificar possíveis valores extremos, foram utilizados duas abordagens complementares: 
* Análise visual por meio de gráficos de boxplot
* Análise estatística utilizando o método do intervalo interquartil (IQR)

### Análise visual (Boxplot)

Inicialmente, foi gerado um boxplot considerando todas as variáveis numéricas, com exceção da variável Burnout_Risk, por se tratar de uma variável binária.

```
colsOutliers = colsNum.drop("Burnout_Risk")
plt.figure(figsize=(20, 10))
sns.boxplot(data=df[colsOutliers],
    flierprops={
        'marker': 'o',               
        'markerfacecolor': 'red',    
        'markersize': 6              
    }
)
plt.xticks(rotation=45)
plt.show()
```

<img width="1210" height="703" alt="Screen Shot 2026-04-21 at 21 58 23" src="https://github.com/user-attachments/assets/8c571aa6-7bf0-497d-88a6-abfcd40295e0" />

A análise visual sugere a presença de alguns valores extremos em determinadas variáveis, indicados pelos pontos destacados fora das caixas. Para confirmar se esses pontos são de fato outliers estatísticos, aplicou-se o método do intervalo interquartil (IQR).

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

## Análise bivariada: variáveis numéricas vs Burnout_Risk

Com o objetivo de investigar a relação entre as variáveis numéricas e a variável alvo (Burnout_Risk), foram utilizados gráficos de boxplot, que permitem comparar a distribuição dos dados entre os grupos “sem burnout = 0” e “com burnout = 1”.

```
for col in colsNum:
    sns.boxplot(x='Burnout_Risk', y=col, data=df)
    plt.show()
```

<img width="562" height="433" alt="image" src="https://github.com/user-attachments/assets/f41083fb-51b8-4e0b-9896-92ee948df606" />
<img width="563" height="433" alt="image" src="https://github.com/user-attachments/assets/fbf3ee4d-e85a-445a-9f6e-2cb204a929d9" />
<img width="563" height="433" alt="image" src="https://github.com/user-attachments/assets/c044bc7b-f3c8-4730-84bd-71f99bc8e824" />
<img width="554" height="433" alt="image" src="https://github.com/user-attachments/assets/3001a247-39c6-4848-a3cd-04143e663a65" />
<img width="571" height="433" alt="image" src="https://github.com/user-attachments/assets/ca62aeb1-a66a-495f-b0e4-5b31d78bdc76" />
<img width="554" height="433" alt="image" src="https://github.com/user-attachments/assets/e16da984-0a28-4a9d-bfab-a384adbf2a83" />
<img width="554" height="433" alt="image" src="https://github.com/user-attachments/assets/c3998832-8b2c-419a-8939-0b698041131e" />
<img width="563" height="433" alt="image" src="https://github.com/user-attachments/assets/6bb7833c-adc7-443d-9084-05dbc7e0d135" />
<img width="576" height="433" alt="image" src="https://github.com/user-attachments/assets/62ccdcaf-9a0e-4b8e-84ab-272af49cbf57" />
<img width="571" height="433" alt="image" src="https://github.com/user-attachments/assets/e86a49f7-49e7-4e30-a9d8-14a8894c1e03" />

### Interpretação dos resultados

A análise conjunta dos boxplots permite identificar padrões relevantes na relação entre as variáveis numéricas e o risco de burnout.

As variáveis podem ser agrupadas em três categorias principais:

1. Variáveis com diferença clara entre os grupos. Algumas variáveis apresentam distinção visível entre indivíduos com e sem burnout:

* Sleep_Hours: indivíduos com burnout apresentam menor mediana de sono
* Stress_Level: distribuição mais elevada no grupo com burnout
* Screen_Time_Hours: maior concentração de valores altos no grupo com burnout

Essas variáveis indicam associação mais evidente com o fenômeno estudado.

2. Variáveis com diferença moderada. Outras variáveis apresentam alguma diferença, mas com sobreposição significativa:

* Productivity_Score
* Exercise_Hours_Per_Week
* Meetings_Per_Day

Esses resultados sugerem que essas variáveis podem ter influência indireta ou combinada com outros fatores.

3. Variáveis com pouca ou nenhuma diferenciação. Algumas variáveis não apresentam diferenças relevantes entre os grupos:

* Age
* Work_Hours_Per_Day
* Internet_Speed_Mbps
* Experience_Years

Em especial, a variável **Work_Hours_Per_Day** apresentou *comportamento contraintuitivo*, sem evidência clara de aumento de burnout com maior carga de trabalho. Esse resultado deve ser interpretado com cautela, podendo estar relacionado à *natureza sintética do dataset*.

De forma geral, observa-se que variáveis comportamentais (sono, estresse, tempo de tela) apresentam maior capacidade de discriminação. As variáveis estruturais (idade, experiência, carga de trabalho) apresentam menor influência isolada. Nota-se uma sobreposição significativa entre os grupos, indicando que o burnout é um fenômeno multifatorial.

Vale ressaltar as **limitações dessa análise**, já que ela é baseada em comparações visuais e não implica causalidade. A variável alvo é binária, limitando a granularidade da análise e o dataset é sintético, podendo apresentar padrões artificiais.

## Distribuição das variáveis numéricas

Para analisar o comportamento das variáveis numéricas, foram utilizados histogramas, que permitem visualizar a distribuição dos dados, identificar padrões de concentração, assimetrias e possíveis irregularidades.

```
df[colsNum].hist(figsize=(16,12))
plt.show()
```
<img width="894" height="665" alt="Screen Shot 2026-04-21 at 22 57 45" src="https://github.com/user-attachments/assets/e7236d86-2e2b-4898-bd7f-d17af63fa546" />

A análise dos histogramas revela diferentes padrões de distribuição entre as variáveis do dataset.

As variáveis podem ser agrupadas em três comportamentos principais:

1. Variáveis com distribuição aproximadamente simétrica

Algumas variáveis apresentam distribuição relativamente equilibrada em torno da média, indicando ausência de forte assimetria:

``Age``
``Work_Hours_Per_Day``
``Sleep_Hours``

Essas variáveis apresentam concentração de valores próximos à mediana, sugerindo maior homogeneidade entre os indivíduos.

2. Variáveis com assimetria (distribuição não uniforme)

Algumas variáveis apresentam assimetria, especialmente à direita, indicando presença de valores mais elevados menos frequentes:

``Experience_Years``: concentração em valores baixos e cauda longa à direita
``Screen_Time_Hours``: maior concentração em valores elevados
``Internet_Speed_Mbps``: ampla dispersão e distribuição assimétrica

Esses padrões indicam maior heterogeneidade entre os indivíduos, especialmente em aspectos relacionados à experiência profissional e infraestrutura tecnológica.

3. Variáveis discretas ou com valores limitados

Algumas variáveis apresentam distribuições com poucos valores possíveis, caracterizando comportamento discreto:

``Meetings_Per_Day``
``Stress_Level``
``Exercise_Hours_Per_Week``

Nesses casos, os histogramas mostram picos em valores específicos, refletindo a natureza categórica ou ordinal dessas variáveis.

### Destaques relevantes
``Sleep_Hours`` apresenta concentração em torno de 6 a 7 horas, abaixo da recomendação média para adultos, o que pode estar associado a fatores de estresse ocupacional.
``Screen_Time_Hours`` apresenta valores elevados (em torno de 9 a 10 horas), indicando alta exposição digital.
``Work_Hours_Per_Day`` concentra-se entre 6 e 9 horas, com baixa variabilidade.
``Productivity_Score`` apresenta ampla distribuição, indicando grande diversidade de níveis de produtividade.

De forma geral, observa-se que variáveis comportamentais apresentam maior dispersão e variabilidade. Variáveis estruturais tendem a ser mais concentradas e algumas distribuições apresentam assimetria, o que deve ser considerado em análises estatísticas posteriores. Além disso, a presença de variáveis discretas e distribuições não normais indica que nem todos os métodos estatísticos paramétricos são adequados, sendo necessário cautela na escolha das técnicas analíticas.

## Distribuição das variáveis numéricas por classe (Burnout_Risk)

Com o objetivo de aprofundar a análise exploratória, foram construídos histogramas condicionados à variável alvo (``Burnout_Risk``), permitindo comparar a distribuição das variáveis numéricas entre os grupos com e sem risco de burnout.

```
colsHist = colsNum.drop("Burnout_Risk")

for col in colsHist:
    plt.figure(figsize=(6, 4))

    sns.histplot(
        data=df,
        x=col,                  # variável numérica que está sendo analisada
        hue="Burnout_Risk",     # separa as distribuições por classe (0 e 1)
        bins=20,                # divide os dados em 20 intervalos (bins)
        element="step",         # evita sobreposição sólida
        stat="density",         # normaliza o histograma
        common_norm=False       # evita que a classe majoritária "domine" o gráfico
    )

    plt.title(f"Distribuição de {col} por Burnout_Risk")
    plt.show()
```

Nessa abordagem, cada histograma apresenta duas distribuições sobrepostas:

* Indivíduos sem burnout (0)
* Indivíduos com burnout (1)

A utilização de densidade (stat="density") e normalização independente (common_norm=False) permite comparar as formas das distribuições sem influência do desbalanceamento da variável alvo.

<img width="554" height="393" alt="image" src="https://github.com/user-attachments/assets/fafab252-442a-4b7b-abf9-a988af61db31" />
<img width="545" height="393" alt="image" src="https://github.com/user-attachments/assets/c2befc14-1c99-4aff-b97e-563b7e94ac63" />
<img width="545" height="393" alt="image" src="https://github.com/user-attachments/assets/43e3d1b6-d713-46ec-a526-cee362211c47" />
<img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/361250dd-25f3-410b-87dd-d321574a0a9b" />
<img width="553" height="393" alt="image" src="https://github.com/user-attachments/assets/f47ae2df-2805-4c09-9985-56d1fd67f88a" />
<img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/bfddc12c-6977-4dc1-aa08-1a9ce6903e2a" />
<img width="559" height="393" alt="image" src="https://github.com/user-attachments/assets/78772e44-ba28-428a-bbf6-89aa4c5b1ff2" />
<img width="553" height="393" alt="image" src="https://github.com/user-attachments/assets/734ec57f-3625-48fb-b774-291ab05f1e4c" />
<img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/f3d36301-bc32-48c2-8caf-c0bb8dbc7f75" />
<img width="545" height="393" alt="image" src="https://github.com/user-attachments/assets/962a306a-e29d-4ccc-ac23-e7045df1c0f2" />

### Análise dos resultados

A comparação entre as distribuições revelou diferentes comportamentos entre as variáveis:

1. Variáveis com separação mais evidente entre as classes. Algumas variáveis apresentam diferenças claras entre os grupos:

``Sleep_Hours``: indivíduos com burnout tendem a apresentar distribuição deslocada para valores menores
``Stress_Level``: maior concentração de valores elevados no grupo com burnout
``Screen_Time_Hours``: maior densidade em valores mais altos para indivíduos com burnout

Esses padrões indicam maior capacidade dessas variáveis em discriminar os grupos.

2. Variáveis com sobreposição parcial

Outras variáveis apresentam diferenças menos acentuadas, com sobreposição significativa entre as distribuições:

``Productivity_Score``
``Exercise_Hours_Per_Week``
``Meetings_Per_Day``

Esses resultados sugerem que essas variáveis podem ter influência indireta ou combinada com outros fatores.

3. Variáveis com pouca diferenciação entre as classes. Algumas variáveis apresentam distribuições muito semelhantes entre os grupos:

``Age``
``Experience_Years``
``Work_Hours_Per_Day``
``Internet_Speed_Mbps``

Em especial, a variável ``Work_Hours_Per_Day`` não apresenta o comportamento esperado, não evidenciando aumento do risco de burnout com maior carga de trabalho. Esse resultado deve ser interpretado com cautela, podendo estar relacionado à *natureza sintética do dataset*.

De forma geral, observa-se que variáveis comportamentais apresentam maior poder de discriminação entre os grupos. Variáveis estruturais apresentam menor impacto isolado. Nota-se que existe forte sobreposição entre as distribuições, indicando que o fenômeno de burnout é multifatorial.

Além disso, a análise evidencia que nenhuma variável isoladamente é suficiente para explicar o risco de burnout, reforçando a necessidade de abordagens multivariadas na etapa de modelagem.

Novamente, há de se ressaltar as limitações da análise:
* A análise é baseada em distribuição e não implica causalidade;
* O desbalanceamento da variável alvo pode influenciar a interpretação visual;
* O dataset é sintético, podendo apresentar padrões artificiais.

## Análise das variáveis categóricas vs Burnout_Risk

Com o objetivo de investigar a relação entre as variáveis categóricas e o risco de burnout, foram construídas tabelas de contingência normalizadas e gráficos de barras empilhadas.

```
for col in colsCat:
    ct = pd.crosstab(df[col], df["Burnout_Risk"], normalize='index')    
    ct.columns = ["Sem_Burnout", "Com_Burnout"]
    ct = ct * 100
    ct.plot(
        kind='bar',
        stacked=True
    )
    plt.title(f'{col} vs Burnout_Risk (proporção)')
    plt.ylabel('Percentual (%)')
    plt.show()
```

A normalização por linha (normalize='index') permite analisar a proporção de indivíduos com e sem burnout dentro de cada categoria, eliminando o efeito do tamanho absoluto dos grupos.

<img width="571" height="491" alt="image" src="https://github.com/user-attachments/assets/b39e654e-bd9c-4ca7-a5b1-fc050c112fca" />
<img width="571" height="525" alt="image" src="https://github.com/user-attachments/assets/c5896c22-8b10-4022-951d-6dfa5d2c0d11" />
<img width="571" height="578" alt="image" src="https://github.com/user-attachments/assets/982a01b4-f5aa-4390-9906-7ed591f8d423" />
<img width="571" height="496" alt="image" src="https://github.com/user-attachments/assets/b0a6583c-e346-4eb0-a1db-00fd1387d44e" />
<img width="571" height="514" alt="image" src="https://github.com/user-attachments/assets/8e8ed472-65c8-434d-975e-234551b26482" />

### Interpretação dos resultados

A análise visual dos gráficos de barras empilhadas mostra que as proporções de burnout entre as diferentes categorias de cada variável são **muito semelhantes**, com diferenças percentuais pequenas e aparentemente não relevantes.

Não se observam variações expressivas em nenhuma das variáveis categóricas analisadas (Gender, Country, Job_Role, Company_Size, Work_Environment). As proporções de indivíduos com burnout permanecem estáveis em torno de 20% em praticamente todas as categorias.

Esta análise visual sugere que, isoladamente, as variáveis categóricas não apresentam forte discriminação do risco de burnout. No entanto, para uma conclusão rigorosa, serão aplicados testes estatísticos formais (qui-quadrado e V de Cramér) nas seções seguintes.

A normalização por linha (normalize='index') foi utilizada para eliminar o efeito do desbalanceamento da variável alvo, permitindo comparar proporções entre categorias de tamanhos diferentes.

## Distribuição da variável alvo (Burnout_Risk)

Para analisar o balanceamento da variável alvo, foi construída uma visualização da distribuição das classes de ``Burnout_Risk``, considerando valores absolutos e percentuais.

```
counts = df["Burnout_Risk"].value_counts()
percent = df["Burnout_Risk"].value_counts(normalize=True) * 100

x = sns.barplot(x=counts.index, y=counts.values)
x.set_xticks([0,1])
x.set_xticklabels(["No", "Yes"])

for i, p in enumerate(ax.patches):
    x.text(p.get_x() + p.get_width()/2,
            p.get_height(),
            f'{percent[i]:.1f}%',
            ha="center")

plt.show()
```

<img width="569" height="434" alt="image" src="https://github.com/user-attachments/assets/ff7bf00c-6a4e-4d5a-8eef-29391d1863fa" />

Foi observado que aproximadamente **80% dos registros correspondem à classe "No" e 20% à classe "Yes"**.

Essa distribuição caracteriza um desbalanceamento significativo, podendo impactar o desempenho de modelos de classificação, especialmente na identificação da classe minoritária.

Entre os principais pontos de atenção, destacam-se:

* Modelos podem apresentar alta acurácia apenas prevendo a classe majoritária;
* Métricas como acurácia podem ser inadequadas para avaliação de desempenho;
* A identificação correta da classe minoritária (burnout) torna-se mais desafiadora.

Dessa forma, será necessário considerar técnicas específicas para lidar com o desbalanceamento, tais como:

* uso de métricas apropriadas (precision, recall, F1-score, ROC-AUC);
* técnicas de reamostragem (oversampling ou undersampling);
* ajuste de pesos de classe em algoritmos de classificação.
* Considerações adicionais

A análise do balanceamento também é importante do ponto de vista aplicado, uma vez que a classe minoritária representa indivíduos em potencial situação de risco. Assim, erros de classificação (especialmente falsos negativos) podem ter implicações relevantes no contexto de uso do modelo.

## Análise de correlação

Foram aplicados três métodos complementares: Point-Biserial (variável alvo binária vs. contínuas), Spearman (associações monotônicas para todas as variáveis numéricas) e Pearson (relações lineares entre variáveis contínuas). A tabela abaixo resume os principais achados para a variável alvo.

| Variável | Coeficiente | Método | Interpretação |
| :--- | :--- | :--- | :--- |
| Productivity_Score | **-0.69** | Point-Biserial | Forte (negativa) |
| Sleep_Hours | **-0.44** | Point-Biserial | Moderada (negativa) |
| Work_Hours_Per_Day | **-0.34** | Point-Biserial | Fraca a moderada (contraintuitivo*) |
| Exercise_Hours_Per_Week | **-0.27** | Point-Biserial | Fraca (negativa) |
| Meetings_Per_Day | **0.18** | Point-Biserial | Fraca (positiva) |
| Screen_Time_Hours | **0.14** | Point-Biserial | Fraca (positiva) |

*O resultado contraintuitivo de `Work_Hours_Per_Day` (mais horas = menos burnout) pode ser um artefato do dataset sintético e será investigado na modelagem.

As demais variáveis (Age, Experience_Years, Internet_Speed_Mbps, Stress_Level) apresentam correlação desprezível (< |0.05|).

### Correlação de Spearman

Para investigar as relações entre as variáveis numéricas, foi utilizada a correlação de Spearman, que mede associações monotônicas entre variáveis, sendo mais adequada em contextos onde não há garantia de linearidade ou normalidade dos dados.

```
corr_spearman = df[colsNum].corr(method='spearman')
sns.heatmap(
    corr_spearman,          # matriz de correlação
    annot=True,             # exibe os valores numéricos dentro dos quadrados
    cmap='coolwarm',        # paleta: azul (negativo) → vermelho (positivo)
    annot_kws={"size": 6}   # tamanho dos valores
)
plt.title("Correlação de Spearman - Todas as Variáveis Numéricas")
plt.show()
```
<img width="692" height="601" alt="image" src="https://github.com/user-attachments/assets/152fa16a-d4b9-43e2-abce-b67110b37b59" />

**Interpretação dos resultados**

A análise do mapa de calor permite identificar padrões de associação entre as variáveis, com destaque para aquelas relacionadas à variável alvo (Burnout_Risk):

- **Productivity_Score** apresenta correlação negativa forte (-0.69) com Burnout_Risk, indicando que menores níveis de produtividade estão fortemente associados a maior risco de burnout.
- **Sleep_Hours** apresenta correlação negativa moderada (-0.44), indicando que menos horas de sono estão associadas a maior risco.
- **Work_Hours_Per_Day** apresenta correlação negativa fraca a moderada (-0.34), um resultado contraintuitivo que será investigado na modelagem.
- **Exercise_Hours_Per_Week** apresenta correlação negativa fraca (-0.27), sugerindo que a prática de exercícios pode estar associada à redução do risco.
- **Meetings_Per_Day** e **Screen_Time_Hours** apresentam correlações positivas fracas (0.18 e 0.14, respectivamente).
- **Stress_Level** apresenta correlação desprezível (-0.0042) com Burnout_Risk na análise de Spearman.

### Correlação de Pearson (variáveis contínuas)

Com o objetivo de analisar relações lineares entre variáveis numéricas contínuas, foi utilizada a correlação de Pearson.

```
colsNumCont = [
    "Age",
    "Work_Hours_Per_Day",
    "Sleep_Hours",
    "Experience_Years",
    "Screen_Time_Hours",
    "Exercise_Hours_Per_Week",
    "Internet_Speed_Mbps",
    "Meetings_Per_Day",
    "Burnout_Risk"
]

corr_pearson = df[colsNumCont].corr(method='pearson')

sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', annot_kws={"size": 6})
plt.title("Correlação de Pearson - Variáveis Contínuas")
plt.show()
```
<img width="692" height="601" alt="image" src="https://github.com/user-attachments/assets/63e76804-4d87-459a-8b30-0f7e7bb19f77" />

**Interpretação dos resultados**

A matriz de Pearson confirma os padrões observados na análise de Spearman para as variáveis contínuas, com destaque para:

- **Sleep_Hours** (-0.44) apresenta a maior correlação linear negativa com burnout, indicando que menos horas de sono estão associadas a maior risco.
- **Work_Hours_Per_Day** (-0.34) apresenta correlação negativa fraca a moderada, confirmando o comportamento contraintuitivo observado na análise de Spearman.
- **Exercise_Hours_Per_Week** (-0.27) e **Internet_Speed_Mbps** (-0.20) apresentam correlações negativas fracas.
- **Meetings_Per_Day** (0.18) e **Screen_Time_Hours** (0.14) apresentam correlações positivas fracas.
- **Age** e **Experience_Years** apresentam correlações desprezíveis com burnout.

A variável **Productivity_Score**, por ser discreta (escala 0-100), não foi incluída nesta matriz de correlação de Pearson, que é específica para relações lineares entre variáveis contínuas. Sua correlação com burnout (-0.69) foi analisada via Point-Biserial e Spearman, conforme apresentado anteriormente.

## Discussão de resultados contraintuitivos

O comportamento observado em variáveis como ``Work_Hours_Per_Day`` deve ser interpretado com cautela, pois contraria expectativas teóricas.

Esse resultado pode estar relacionado a características artificiais do dataset sintético. Ausência de variáveis mediadoras relevantes. Possíveis relações indiretas não capturadas pela correlação bivariada e relações entre variáveis explicativas.

Além da variável alvo, observa-se também, possíveis associações entre variáveis comportamentais (sono, exercício, tempo de tela). Verifica-se baixa correlação entre variáveis demográficas e comportamentais.

Esses padrões sugerem que diferentes grupos de variáveis capturam aspectos distintos do fenômeno analisado.

Ressalta-se as limitações da análise:
* A correlação de Spearman mede associação monotônica, não implicando causalidade;
* Relações não monotônicas não são capturadas por esse método;
* A análise é bivariada e não considera interações entre múltiplas variáveis;
* O dataset é sintético, podendo apresentar padrões artificiais.

## Análise do Point-Biserial para Burnout

```
from scipy.stats import pointbiserialr
for col in colsNum:
    corr, _ = pointbiserialr(df['Burnout_Risk'], df[col])
    print(col, corr)
```

<img width="379" height="205" alt="Screen Shot 2026-04-22 at 00 17 17" src="https://github.com/user-attachments/assets/01904cb5-0272-42fe-8e8a-1fc9c788f79d" />

O risco de burnout está fortemente associado a queda de produtividade e piora de hábitos de recuperação (sono e exercício). Fatores demográficos como idade e experiência são irrelevantes. O modelo sugere que burnout é um fenômeno predominantemente comportamental e operacional.

## Análise do teste Qui-quadrado para variáveis categóricas

```
from scipy.stats import chi2_contingency
for col in colsCat:
    table = pd.crosstab(df[col], df['Burnout_Risk'])
    chi2, p, _, _ = chi2_contingency(table)
    print(col, "p-value:", p)
```


<img width="396" height="93" alt="Screen Shot 2026-04-22 at 00 19 35" src="https://github.com/user-attachments/assets/f24b3af4-4d15-4e2a-a79a-6845b04eec0b" />

As variáveis categóricas analisadas não apresentam associação estatisticamente significativa com o risco de burnout, indicando que o fenômeno observado no dataset é predominantemente explicado por fatores comportamentais contínuos, e não por características demográficas ou estruturais.

## Força da associação (V de Cramér)

```
for col in colsCat:
    table = pd.crosstab(df[col], df['Burnout_Risk'])
    v_cramer = cramers_v(table.values)
    
    # Interpretação
    if v_cramer < 0.05:
        interp = "Desprezível"
    elif v_cramer < 0.10:
        interp = "Fraca"
    else:
        interp = "Moderada/Forte"
    
    print(f"{col}: V de Cramér = {v_cramer:.4f} ({interp})")
```

Complementando a análise do qui-quadrado, o coeficiente V de Cramér mede a intensidade da associação entre variáveis categóricas e o risco de burnout:

<img width="461" height="136" alt="Screen Shot 2026-04-22 at 15 52 06" src="https://github.com/user-attachments/assets/45aeaf70-b6d3-41b0-9221-1588a57d6195" />

Todas as variáveis categóricas apresentaram associação desprezível com o risco de burnout (V de Cramér < 0.05), confirmando que características demográficas e estruturais não são preditoras relevantes neste dataset. O fenômeno do burnout é explicado predominantemente por fatores comportamentais contínuos.

## Análise de Scatterplots

```
cols_Scatterplots = colsNum.drop("Burnout_Risk")

for col in cols_Scatterplots:
    if col != "Burnout_Risk":
        sns.scatterplot(
            x=col,
            y="Burnout_Risk",
            data=df
        )
        plt.title(f"{col} vs Burnout_Risk")
        plt.show()
```

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/88d7e66e-7a40-436f-a78b-6f3415f21104" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/d8feaac3-8d23-4df1-af1d-968337687af1" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/b14147c5-1a83-4b00-bacd-82ccab0ff49c" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/5ffb3304-091e-422c-96ed-df64c08db15b" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/ea456e4d-bd00-4f66-8d99-6567e1dede3c" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/84952a6c-298b-4efe-8de1-17ce53091be2" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/22205b89-3857-4702-80dd-2d0f4019980a" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/bb841689-ede5-44d3-b2c3-df6b94a83b87" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/3ef3bea1-3105-4840-bd4c-83976629b455" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/d6719e77-2073-42a4-a263-bed475613805" />

O burnout no conjunto analisado está fortemente associado a um padrão de piora em indicadores de recuperação e desempenho, especialmente:

- redução expressiva da produtividade
- menor duração do sono
- menor atividade física
- aumento de reuniões e tempo de tela

Em contraste, características demográficas não apresentam poder explicativo relevante. Isso sugere que o fenômeno é predominantemente comportamental e operacional, não estrutural.

## Considerações éticas e adequação à LGPD

Embora o dataset utilizado seja sintético, a análise envolve variáveis relacionadas a comportamento, saúde e contexto profissional, o que exige atenção a aspectos éticos e às diretrizes da Lei Geral de Proteção de Dados (LGPD).

### Sensibilidade dos dados

Algumas variáveis presentes no dataset podem ser consideradas sensíveis ou potencialmente sensíveis no contexto real, tais como:

- Stress_Level (nível de estresse)
- Sleep_Hours (hábitos de sono)
- Exercise_Hours_Per_Week (hábitos de saúde)
- Burnout_Risk (indicador de saúde mental)

Em um cenário real, essas informações poderiam ser classificadas como dados pessoais sensíveis, pois estão relacionadas à saúde e ao bem-estar do indivíduo.

### Riscos no uso dos dados e do modelo

Caso esse tipo de análise fosse aplicada em dados reais, alguns riscos importantes deveriam ser considerados:

- Uso punitivo do modelo: identificação de indivíduos com alto risco de burnout poderia ser utilizada para penalização ou discriminação no ambiente de trabalho;
- Estigmatização: funcionários classificados como “alto risco” poderiam ser tratados de forma desigual;
- Privacidade: exposição indevida de informações relacionadas à saúde e comportamento;
- Decisões automatizadas: uso do modelo sem supervisão humana pode gerar interpretações incorretas.
- Falsos positivos e falsos negativos

### Em modelos preditivos de burnout:

- Falsos positivos (indicar burnout quando não existe) podem gerar preocupação desnecessária ou impacto na reputação profissional;
- Falsos negativos (não identificar burnout quando existe) podem impedir intervenções preventivas importantes.

Ambos os casos podem causar impactos negativos, reforçando a necessidade de uso responsável.

### Viés e equidade

A análise exploratória indicou que variáveis categóricas como gênero, país e cargo apresentam distribuições semelhantes em relação ao burnout.

No entanto, em um cenário real, seria fundamental avaliar:

- possíveis viéses discriminatórios;
- diferenças de desempenho do modelo entre grupos;
- impacto desigual das decisões automatizadas.
- Limitações do dataset

O dataset utilizado é sintético, o que implica:

- ausência de dados pessoais reais;
- redução de riscos diretos à privacidade;
- possível presença de padrões artificiais.

Dessa forma, os resultados obtidos não devem ser generalizados diretamente para contextos reais sem validação adicional.

### Uso responsável da solução

A aplicação prática de um modelo de predição de burnout deve ser orientada por princípios éticos, sendo utilizada como:

- ferramenta de apoio à prevenção;
- instrumento para melhoria do bem-estar organizacional;
- base para ações de suporte, e não para controle ou punição.

### Mitigação de Riscos

| Risco | Mitigador Proposto |
|-------|-------|
|Uso punitivo do modelo |Modelo deve ser usado apenas por RH/gestão com supervisão humana, nunca como ferramenta automatizada de demissão.|
|Falsos negativos (não identificar burnout real)|Implementar limiar de decisão ajustado para maximizar recall, com confirmação por profissional de saúde.|
|Viés de gênero/raça/cargo|Auditoria periódica de métricas de desempenho por grupo.|


## Ferramentas utilizadas

A análise exploratória foi realizada utilizando a linguagem de programação Python, executada no ambiente Google Colab, que oferece suporte para análise de dados e desenvolvimento de modelos de aprendizado de máquina.

As principais bibliotecas utilizadas foram:

| Ferramenta | Aplicação |
| :--- | :--- |
| Python 3.10+ | Linguagem de programação |
| pandas 2.2.2 | Manipulação e estruturação de dados |
| matplotlib 3.8.4 | Criação de gráficos e visualizações |
| seaborn 0.13.2 | Visualização estatística avançada |
| scipy 1.13.1 | Testes estatísticos (point-biserial, qui-quadrado) |
| kagglehub 0.3.10 | Download do dataset do Kaggle |

> **Nota sobre reprodutibilidade**: O arquivo `requirements.txt` com as versões exatas de todas as dependências será disponibilizado na raiz do repositório GitHub na entrega final. A opção por não incluí-lo durante o desenvolvimento deve-se à natureza efêmera do ambiente Colab, que gerencia dinamicamente as bibliotecas a cada sessão. A versão consolidada do projeto conterá todos os artefatos necessários para reprodução integral da análise.
