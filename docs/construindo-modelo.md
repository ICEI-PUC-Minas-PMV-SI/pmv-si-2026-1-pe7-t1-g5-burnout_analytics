# Preparação dos dados

Nesta etapa, deverão ser descritas todas as técnicas utilizadas para pré-processamento/tratamento dos dados.

Algumas das etapas podem estar relacionadas à:

* Limpeza de Dados: trate valores ausentes: decida como lidar com dados faltantes, seja removendo linhas, preenchendo com médias, medianas ou usando métodos mais avançados; remova _outliers_: identifique e trate valores que se desviam significativamente da maioria dos dados.

* Transformação de Dados: normalize/padronize: torne os dados comparáveis, normalizando ou padronizando os valores para uma escala específica; codifique variáveis categóricas: converta variáveis categóricas em uma forma numérica, usando técnicas como _one-hot encoding_.

* _Feature Engineering_: crie novos atributos que possam ser mais informativos para o modelo; selecione características relevantes e descarte as menos importantes.

* Tratamento de dados desbalanceados: se as classes de interesse forem desbalanceadas, considere técnicas como _oversampling_, _undersampling_ ou o uso de algoritmos que lidam naturalmente com desbalanceamento.

* Separação de dados: divida os dados em conjuntos de treinamento, validação e teste para avaliar o desempenho do modelo de maneira adequada.
  
* Manuseio de Dados Temporais: se lidar com dados temporais, considere a ordenação adequada e técnicas específicas para esse tipo de dado.
  
* Redução de Dimensionalidade: aplique técnicas como PCA (Análise de Componentes Principais) se a dimensionalidade dos dados for muito alta.

* Validação Cruzada: utilize validação cruzada para avaliar o desempenho do modelo de forma mais robusta.

* Monitoramento Contínuo: atualize e adapte o pré-processamento conforme necessário ao longo do tempo, especialmente se os dados ou as condições do problema mudarem.

* Entre outras....

Avalie quais etapas são importantes para o contexto dos dados que você está trabalhando, pois a qualidade dos dados e a eficácia do pré-processamento desempenham um papel fundamental no sucesso de modelo(s) de aprendizado de máquina. É importante entender o contexto do problema e ajustar as etapas de preparação de dados de acordo com as necessidades específicas de cada projeto.

# Descrição do modelo

Nesta seção, conhecendo os dados e de posse dos dados preparados, é hora de descrever o algoritmo de aprendizado de máquina selecionado para a construção do modelo proposto. Inclua informações abrangentes sobre o algoritmo implementado, aborde conceitos fundamentais, princípios de funcionamento, vantagens/limitações e justifique a escolha do algoritmo utilizado. 

Explore aspectos específicos, como o ajuste dos parâmetros livres do algoritmo. Lembre-se de experimentar parâmetros diferentes e principalmente, de registrar os testes realizados com diferentes parâmetros que servirão para justificar as escolhas realizadas.

# Avaliação dos modelos criados

## Métricas utilizadas

Nesta seção, as métricas utilizadas para avaliar os modelos desenvolvidos deverão ser apresentadas (p. ex.: acurácia, precisão, recall, F1-Score, MSE etc.). A escolha de cada métrica deverá ser justificada, pois esta escolha é essencial para avaliar de forma mais assertiva a qualidade do modelo construído. 

## Discussão dos resultados obtidos

Nesta seção, discuta os resultados obtidos pelo modelo construído, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade dele. Lembre-se de relacionar os resultados obtidos por cada uma das métricas ao problema identificado, a questão de pesquisa levantada e estabelecer relação com os objetivos previamente propostos. 
É fundamental compreender o que cada uma das métricas "conta" sobre a qualidade do modelo desenvolvido.

# Pipeline de pesquisa e análise de dados

Em pesquisa e experimentação em sistemas de informação, um pipeline de pesquisa e análise de dados refere-se a um conjunto organizado de processos e etapas que um profissional segue para realizar a coleta, preparação, análise e interpretação de dados durante a fase de pesquisa e desenvolvimento de modelos. Esse pipeline é essencial para extrair _insights_ significativos, entender a natureza dos dados e, construir modelos de aprendizado de máquina eficazes. 

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".


-----////------

# Etapa 3: Preparação dos dados

Nesta etapa, foram aplicadas técnicas de pré-processamento com o objetivo de adequar o conjunto de dados ao treinamento do modelo de aprendizado de máquina, garantindo consistência, robustez e reprodutibilidade dos resultados.

Inicialmente, foi realizada a etapa de limpeza dos dados, na qual se confirmou a ausência de valores faltantes no dataset, conforme identificado na análise exploratória. Embora o dataset original não apresente valores ausentes, optou-se por manter etapas de imputação no pipeline, utilizando a mediana para variáveis numéricas e a moda para variáveis categóricas. Essa decisão foi adotada visando aumentar a robustez do modelo e sua capacidade de generalização para cenários futuros com dados incompletos. Em relação à detecção de valores extremos, embora tenham sido observados alguns perfis atípicos — como indivíduos com baixa quantidade de horas de sono ou elevado tempo de exposição a telas —, a análise pelo método do intervalo interquartil (IQR) indicou que tais valores não configuram outliers estatísticos. Assim, optou-se por manter todos os registros, considerando sua relevância potencial para o fenômeno estudado.

Na sequência, foi realizada a transformação dos dados, com foco na preparação das variáveis para uso em algoritmos de aprendizado supervisionado. As variáveis categóricas — `Gender`, `Country`, `Job_Role`, `Company_Size` e `Work_Environment` — foram convertidas em formato numérico por meio da técnica de **One-Hot Encoding**, utilizando a classe `OneHotEncoder` da biblioteca scikit-learn com parâmetro `handle_unknown='ignore'`. Esse procedimento permite representar categorias qualitativas em forma binária, além de garantir que categorias não vistas durante o treinamento não comprometam o funcionamento do modelo.

Adicionalmente, procedeu-se à remoção da variável `Employee_ID`, por se tratar de um identificador único sem relevância preditiva. Em seguida, os dados foram organizados em variáveis preditoras (X) e variável alvo (y), sendo esta última representada pela variável `Burnout_Risk`, codificada em formato binário.

Considerando a presença de variáveis numéricas em diferentes escalas, foi aplicada a técnica de padronização por meio do `StandardScaler`, que transforma os dados para uma distribuição com média zero e desvio padrão unitário. Essa etapa é particularmente importante para algoritmos como a Regressão Logística, que são sensíveis à escala das variáveis.

As etapas de pré-processamento foram organizadas por meio de um `ColumnTransformer`, responsável por aplicar transformações distintas para variáveis numéricas e categóricas. Esse componente foi integrado a um pipeline completo utilizando a classe `Pipeline`, garantindo que todas as etapas sejam executadas de forma consistente tanto durante o treinamento quanto na fase de teste. Essa abordagem também evita o problema de vazamento de dados (data leakage), assegurando maior rigor metodológico.

Outro aspecto relevante do pré-processamento refere-se ao desbalanceamento da variável alvo, no qual aproximadamente 80% dos registros pertencem à classe “No” e 20% à classe “Yes”. Para lidar com esse cenário, foi adotada a estratégia de ajuste de pesos das classes (`class_weight='balanced'`), permitindo que o modelo atribua maior importância à classe minoritária durante o treinamento, sem a necessidade de técnicas adicionais como oversampling.

Por fim, o conjunto de dados foi dividido em subconjuntos de treinamento (80%) e teste (20%), utilizando uma estratégia estratificada da variável alvo (`stratify=y`), de modo a preservar a proporção original das classes. A utilização de um valor fixo de random_state garantiu a reprodutibilidade dos experimentos.

# Descrição do modelo

O modelo selecionado para esta etapa foi a **Regressão Logística**, amplamente utilizada em problemas de classificação binária. Esse algoritmo baseia-se na aplicação da função logística, que transforma uma combinação linear das variáveis independentes em uma probabilidade entre 0 e 1, permitindo estimar a chance de ocorrência do evento de interesse. A decisão de classificação é realizada a partir de um limiar de probabilidade, geralmente definido como 0,5, a partir do qual o modelo classifica a observação como pertencente à classe positiva.

Do ponto de vista conceitual, a Regressão Logística modela a relação entre as variáveis preditoras e a variável resposta por meio da função sigmoide, possibilitando a interpretação probabilística dos resultados. Essa característica a torna particularmente adequada para problemas em que a tomada de decisão depende da estimativa de risco, como no caso da predição de burnout.

A escolha desse modelo fundamenta-se, sobretudo, em sua elevada interpretabilidade. Diferentemente de algoritmos mais complexos, a Regressão Logística permite analisar diretamente os coeficientes associados a cada variável, indicando tanto a direção quanto a intensidade de sua influência sobre o risco de burnout. Essa propriedade é especialmente relevante no contexto organizacional, no qual a transparência e a explicabilidade dos modelos são fundamentais para sua adoção por profissionais de recursos humanos e gestores.

Além disso, o modelo foi adotado como baseline, servindo como referência inicial para comparação com abordagens mais sofisticadas em etapas posteriores do projeto. A utilização de um modelo baseline é uma prática consolidada em ciência de dados, pois permite avaliar se o aumento de complexidade em modelos futuros resulta, de fato, em ganhos significativos de desempenho.

A implementação foi realizada utilizando a biblioteca `scikit-learn`, com destaque para o ajuste do parâmetro `max_iter=1000`, visando garantir a convergência do algoritmo, e a utilização de `class_weight='balanced'`, com o objetivo de mitigar os efeitos do desbalanceamento entre as classes.

Como principal limitação, destaca-se o fato de que a Regressão Logística assume relações lineares entre as variáveis independentes e o logaritmo da razão de chances (`log-odds`), o que pode restringir sua capacidade de capturar padrões mais complexos presentes nos dados. Ainda assim, sua simplicidade e interpretabilidade justificam sua utilização como ponto de partida.

Adicionalmente, o modelo permite a análise direta dos coeficientes estimados, possibilitando a identificação dos fatores que mais contribuem para o aumento ou redução do risco de burnout, reforçando sua aplicabilidade no contexto organizacional.

# Avaliação do modelo criado
## Métricas utilizadas

A avaliação do modelo foi realizada por meio de múltiplas métricas, com o objetivo de obter uma visão abrangente de seu desempenho. Foram consideradas as métricas de *acurácia (accuracy)*, *precisão (precision)*, *revocação (recall)*, *F1-score* e área sob a *curva ROC (AUC-ROC)*.

A **acurácia** foi utilizada para medir a proporção geral de acertos do modelo. No entanto, devido ao desbalanceamento das classes, essa métrica isoladamente não é suficiente para avaliar adequadamente a performance.

A **precisão** indica a proporção de previsões positivas que estão corretas, sendo relevante para avaliar a confiabilidade das predições. Já o recall mede a capacidade do modelo de identificar corretamente os casos positivos, sendo particularmente importante em cenários nos quais o custo de não detectar um evento é elevado.

O **F1-score** foi utilizado como medida de equilíbrio entre precisão e recall, enquanto a AUC-ROC permitiu avaliar a capacidade do modelo de discriminar entre as classes ao longo de diferentes limiares de decisão.

Dentre essas métricas, o **recall** foi definido como o principal critério de avaliação, uma vez que, no contexto do problema, o erro mais crítico consiste na ocorrência de falsos negativos — ou seja, situações em que o modelo deixa de identificar indivíduos em risco de burnout. Tal falha pode comprometer ações preventivas e impactar negativamente a saúde dos trabalhadores e o desempenho organizacional.

## Resultados dos experimentos com diferentes configurações

Com o objetivo de avaliar o impacto de diferentes configurações do modelo no desempenho preditivo, foram realizados experimentos controlados variando-se parâmetros-chave da Regressão Logística. Especificamente, foram analisados: (i) o uso ou não de balanceamento de classes, (ii) o tipo de penalização aplicada (L1 e L2) e (iii) a intensidade da regularização, controlada pelo parâmetro C.

A realização desses testes permite compreender a sensibilidade do modelo a diferentes configurações, além de fornecer embasamento empírico para a escolha da configuração final adotada.

Tabela I – Resultados dos testes experimentais da Regressão Logística

| Teste           | Class Weight | Penalização | C    | Accuracy | Precision | Recall  | F1-Score | AUC-ROC  |
|-----------------|-------------|-------------|------|----------|-----------|---------|----------|----------|
| Baseline        | None        | L2          | 1.0  | 0.999000 | 0.997470  | 0.997470| 0.997470 | 0.999990 |
| Balanced L2     | Balanced    | L2          | 1.0  | 0.990167 | 0.952610  | 1.000000| 0.975730 | 0.999993 |
| L2 C=0.1        | Balanced    | L2          | 0.1  | 0.978333 | 0.901216  | 1.000000| 0.948042 | 0.999651 |
| L2 C=10         | Balanced    | L2          | 10.0 | 0.999667 | 0.998316  | 1.000000| 0.999158 | 1.000000 |
| L1 C=1.0        | Balanced    | L1          | 1.0  | 1.000000 | 1.000000  | 1.000000| 1.000000 | 1.000000 |
| Threshold 0.3   | Balanced    | L2          | 1.0  | 0.981000 | 0.912308  | 1.000000| 0.954143 | 0.999993 |

A análise dos resultados evidencia que o modelo apresenta desempenho extremamente elevado em todas as configurações testadas, com valores de recall próximos ou iguais a 1.0 na maioria dos experimentos. Esse comportamento indica que o modelo já possui alta capacidade de identificação da classe positiva, independentemente da configuração adotada.

Nesse contexto, a utilização de `class_weight='balanced'` não resultou em ganhos expressivos de recall, mas impactou negativamente a precisão, indicando aumento na ocorrência de falsos positivos. Esse comportamento é esperado, uma vez que o balanceamento torna o modelo mais sensível à classe minoritária.

Observa-se que as principais variações entre os testes ocorreram na precisão, e não no recall, evidenciando um trade-off entre sensibilidade e especificidade do modelo. Configurações mais sensíveis, como o uso de balanceamento ou redução do threshold, tendem a aumentar a identificação de casos positivos ao custo de maior número de classificações incorretas.

Adicionalmente, a variação do parâmetro de regularização (C) e do tipo de penalização (L1 e L2) não resultou em mudanças significativas no desempenho global, indicando que o problema apresenta alta separabilidade entre as classes. Esse resultado sugere que o dataset, possivelmente por sua natureza sintética, apresenta padrões facilmente capturáveis pelo modelo.

Todos os experimentos foram executados utilizando o mesmo pipeline de pré-processamento e divisão de dados, garantindo comparabilidade entre os testes. As variações foram aplicadas exclusivamente nos parâmetros do modelo, mantendo-se constantes os demais elementos do processo. Essa abordagem assegura que as diferenças observadas nos resultados sejam decorrentes apenas das configurações testadas.

## Discussão dos resultados obtidos

Os resultados obtidos indicam que o modelo de Regressão Logística apresenta desempenho satisfatório, servindo como uma base sólida para a modelagem do problema.

A Figura I apresenta a matriz de confusão do modelo, que evidencia a distribuição das previsões realizadas pelo modelo, permitindo observar a quantidade de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos. 

<img width="518" height="393" alt="image" src="https://github.com/user-attachments/assets/a11bb841-da76-44a9-a15a-3baa938f9d7c" />

Figura I - Matriz de Confusão do Modelo de Regressão Logística 

A análise da matriz de confusão evidencia que o modelo apresenta elevada capacidade de identificação da classe positiva, não sendo observados falsos negativos. Esse resultado indica que todos os indivíduos em risco de burnout foram corretamente classificados pelo modelo.

Por outro lado, observa-se a presença de falsos positivos (64 casos), indicando que alguns indivíduos foram classificados como em risco quando, na realidade, não pertenciam a essa classe. Embora esse tipo de erro possa gerar intervenções desnecessárias, ele é menos crítico no contexto do problema, quando comparado aos falsos negativos.

A Figura II apresenta a curva ROC.

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/a5c3d1fe-3b04-45bc-bb3a-ea8ddf14b8b6" />

Figura II - Curva ROC 

A curva ROC apresentada na Figura II evidencia a capacidade do modelo em discriminar entre as classes ao longo de diferentes limiares de decisão. Observa-se que a curva se mantém significativamente acima da linha de referência aleatória, indicando desempenho superior ao acaso.

Adicionalmente, o formato da curva, próximo ao canto superior esquerdo, sugere elevada taxa de verdadeiros positivos combinada com baixa taxa de falsos positivos, reforçando a qualidade do modelo na separação entre indivíduos com e sem risco de burnout.

A análise conjunta das métricas evidencia que o modelo consegue equilibrar a taxa de acertos gerais com a capacidade de identificação da classe minoritária, ainda que apresente limitações típicas de modelos lineares. Em particular, observa-se que o foco no recall contribui para uma maior sensibilidade na detecção de casos positivos, o que está alinhado com os objetivos do projeto.

No contexto da questão de pesquisa — que busca verificar a possibilidade de prever o risco de burnout a partir de variáveis ocupacionais e comportamentais —, os resultados obtidos reforçam a viabilidade da abordagem proposta. O modelo demonstra que, mesmo utilizando uma técnica relativamente simples, é possível extrair padrões preditivos relevantes a partir dos dados disponíveis.

Entretanto, é importante destacar que algumas relações identificadas diferem da literatura, possivelmente em função da **natureza sintética do dataset** (produzido por Inteligência Artificial) conforme explicitado pelo autor. Dessa forma, os resultados devem ser interpretados com cautela, especialmente no que se refere à generalização para contextos reais.

A análise dos coeficientes do modelo permitiu identificar que variáveis relacionadas ao estilo de vida, como horas de sono e prática de exercícios físicos, apresentam associação com a redução do risco de burnout. Destaca-se também o forte impacto da variável de produtividade, que apresentou coeficiente negativo de alta magnitude. Por outro lado, observou-se que a variável de carga horária apresentou relação inversa ao esperado, indicando que maiores jornadas de trabalho estariam associadas à redução do risco de burnout. Esse resultado contraria evidências da literatura e sugere uma limitação decorrente da **natureza sintética do dataset**, devendo ser interpretado com cautela.

## Decisão do Modelo

Com base nos resultados obtidos, optou-se por utilizar a configuração com `class_weight='balanced'` e penalização L2, considerando seu melhor equilíbrio entre capacidade preditiva e generalização. Essa configuração apresentou desempenho consistente nas métricas avaliadas, com destaque para o recall, métrica prioritária neste estudo.

A escolha está alinhada ao objetivo central do projeto, que consiste na identificação de indivíduos em risco de burnout, minimizando a ocorrência de falsos negativos. Dessa forma, privilegia-se um modelo mais sensível à classe positiva, ainda que isso implique um aumento controlado de falsos positivos.

# Pipeline de pesquisa e análise de dados

O pipeline de pesquisa e análise de dados adotado neste projeto foi estruturado de forma a garantir coerência metodológica, reprodutibilidade e alinhamento com boas práticas em ciência de dados.

O processo teve início com a definição do problema, caracterizado como uma tarefa de classificação binária voltada à predição do risco de burnout. Em seguida, foi realizada a análise exploratória dos dados, permitindo compreender a distribuição das variáveis e identificar padrões iniciais.

Na etapa de preparação dos dados, foram aplicadas técnicas de limpeza, transformação e padronização, bem como estratégias para lidar com o desbalanceamento das classes. Posteriormente, os dados foram divididos em conjuntos de treinamento e teste, possibilitando a avaliação da capacidade de generalização do modelo.

A fase de modelagem consistiu na implementação da Regressão Logística, seguida pela avaliação do desempenho utilizando múltiplas métricas. Por fim, os resultados foram interpretados à luz do problema proposto, buscando extrair insights relevantes para aplicação no contexto organizacional.

Esse pipeline não apenas organiza o fluxo de trabalho, mas também permite sua replicação em diferentes cenários, contribuindo para a robustez, replicabilidade e potencial generalização da solução desenvolvida.

# Observações importantes

Todas as etapas de modelagem, avaliação e análise foram implementadas utilizando a linguagem Python, com suporte das bibliotecas pandas, scikit-learn e numpy.

Os códigos desenvolvidos encontram-se documentados e devem ser disponibilizados integralmente na pasta "src" do repositório, garantindo transparência, reprodutibilidade e aderência às boas práticas de desenvolvimento em projetos de ciência de dados.
