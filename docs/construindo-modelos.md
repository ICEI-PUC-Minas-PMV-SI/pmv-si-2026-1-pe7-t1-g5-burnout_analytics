# Preparação dos dados

Nesta etapa, foram aplicadas técnicas de pré-processamento para garantir que o conjunto de dados estivesse em formato ideal para o treinamento dos algoritmos de aprendizado de máquina. O uso de boas práticas, como a prevenção de vazamento de dados (data leakage), foi priorizado por meio da construção de Pipelines de transformação.

* **Limpeza de Dados**: Conforme identificado na Análise Exploratória (EDA), o dataset original não possuía valores nulos. No entanto, para garantir a robustez do modelo em um ambiente de produção (onde novos dados podem vir incompletos), foram incluídos *SimpleImputers* no pipeline: preenchimento com a mediana para numéricos e com a moda (most_frequent) para categóricos. Além disso, a coluna *Employee_ID* foi removida, pois atua apenas como identificador e não possui poder preditivo. Os outliers da variável *Experience_Years* foram mantidos intencionalmente por serem valores plausíveis no domínio corporativo e característicos da geração sintética do dataset.

* **Transformação e Codificação de Dados**:

  - **Variáveis Alvo e Ordinais**: A variável alvo  *Burnout_Risk* foi mapeada para binário (Yes = 1, No = 0). A variável *Stress_Level* foi codificada ordinalmente (Low = 1, Medium = 2, High = 3) para preservar a hierarquia de intensidade.

  - **Variáveis Categóricas Nominais**: O algoritmo *OneHotEncoder* foi aplicado (com handle_unknown='ignore') em colunas como *Gender, Country, Job_Role, Company_Size e Work_Environment*, transformando representações textuais em matrizes binárias interpretáveis pelos modelos.

  - **Padronização**: As variáveis numéricas contínuas passaram pelo *StandardScaler*, que transforma os dados para possuírem média zero e desvio padrão igual a um. Isso é crucial para algoritmos sensíveis à escala, como a Regressão Logística.
 
* **Separação dos Dados**: Os dados foram divididos em conjunto de **treino (80%)** e **teste (20%**) utilizando a função *train_test_split*.

* **Tratamento de Dados Desbalanceados**: A variável alvo apresentou um desbalanceamento de aproximadamente 80% (Sem Burnout) contra 20% (Com Burnout). Para mitigar esse efeito, aplicou-se a técnica de estratificação *(stratify=y)* durante a divisão de treino/teste, garantindo que a proporção de 80/20 se mantivesse em ambos os conjuntos. Nos modelos aplicáveis (como a Regressão Logística), utilizou-se o parâmetro *class_weight='balanced'* para penalizar erros na classe minoritária.

* **Sistematização (ColumnTransformer)**: Todas essas transformações foram encapsuladas em um ColumnTransformer dentro de um Pipeline do Scikit-Learn. Isso garante que as mesmas regras matemáticas aplicadas aos dados de treino sejam rigorosamente aplicadas aos dados de teste (e a dados futuros), evitando contaminações.

# Descrição dos modelos

Além da Regressão Logística (que atuou como modelo baseline e cujos detalhes foram discutidos em seções anteriores), foram implementados dois modelos robustos baseados em árvores de decisão. Essa abordagem permite avaliar se a captura de relações não-lineares resulta em ganhos de performance.

* **Random Forest**:

O Random Forest é um algoritmo de aprendizado supervisionado baseado na técnica de Ensemble Learning, especificamente o método de *Bagging (Bootstrap Aggregating)*.

**Princípio de funcionament**o: O algoritmo constrói múltiplas árvores de decisão independentes durante o treinamento. Cada árvore é treinada com uma amostra aleatória dos dados (com reposição) e um subconjunto aleatório de variáveis (features). A predição final é obtida por meio da votação majoritária (no caso de classificação) das árvores individuais.

**Vantagens e Limitações**: Suas principais vantagens incluem a alta resistência ao overfitting (devido à aleatoriedade na construção das árvores), capacidade de modelar interações complexas não-lineares e a não exigência de escalonamento dos dados. A limitação recai sobre sua natureza "caixa-preta" (black-box), sendo mais difícil de interpretar matematicamente do que uma Regressão Logística, além do maior custo computacional e de memória.

**Justificativa e Ajuste de Parâmetros**: Escolhido por sua robustez e por ser o padrão-ouro em dados tabulares. Na implementação (n_estimators=200, max_depth=10, random_state=42), o número de estimadores garantiu a estabilidade estatística do conjunto (200 árvores), enquanto a profundidade máxima foi limitada a 10 níveis para restringir a complexidade e forçar a generalização do modelo.

---

* **XGBoost (Extreme Gradient Boosting)**:

O XGBoost é uma implementação altamente eficiente da técnica de *Gradient Boosting*.

**Princípio de funcionamento**: Ao contrário do Random Forest (onde as árvores são independentes), o XGBoost constrói árvores de decisão de forma sequencial. Cada nova árvore é treinada para corrigir os erros residuais (gradientes) cometidos pelas árvores anteriores. O algoritmo utiliza regularização L1 e L2 internamente, otimizando uma função de perda de forma iterativa.

**Vantagens e Limitações**: É amplamente reconhecido por apresentar o estado-da-arte em desempenho preditivo para dados tabulares estruturados, lidando extremamente bem com padrões complexos. Contudo, é altamente sensível à hiperparametrização; se não for bem ajustado, tende a decorar os dados de treino (overfitting).

**Justificativa e Ajuste de Parâmetros**: Escolhido por seu poder preditivo superior. Os parâmetros definidos foram n_estimators=200, learning_rate=0.05 e max_depth=6. A taxa de aprendizado reduzida (0.05) garante que cada árvore contribua de forma conservadora para o modelo final, enquanto a profundidade rasa (nível 6) atua como um forte regulador contra o sobreajuste. Utilizou-se a métrica interna *eval_metric='logloss'*.

# Avaliação dos modelos criados

## Métricas utilizadas

Nesta seção, as métricas utilizadas para avaliar os modelos desenvolvidos deverão ser apresentadas (p. ex.: acurácia, precisão, recall, F1-Score, MSE etc.). A escolha de cada métrica deverá ser justificada, pois esta escolha é essencial para avaliar de forma mais assertiva a qualidade do modelo construído. 

## Discussão dos resultados obtidos

Nesta seção, discuta os resultados obtidos por cada um dos modelos construídos na Etapa 03 e na Etapa 04, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade de cada um deles. Lembre-se de relacionar os resultados obtidos ao problema identificado, a questão de pesquisa levantada e estabelecer relação com os objetivos previamente propostos. Não deixe de comparar os resultados obtidos por cada modelo com os demais.

# Revisão do pipeline de pesquisa e análise de dados

Nesta etapa, os alunos devem revisar o pipeline de pesquisa e análise de dados proposto na Etapa 03, avaliando criticamente cada uma de suas etapas, fluxos e decisões. O objetivo agora é identificar possíveis ajustes, melhorias ou generalizações que tornem o pipeline mais abrangente e adaptável, de forma que ele seja capaz de representar qualquer processo de construção de sistemas de aprendizado de máquina – independentemente da área de aplicação, tipo de dado ou técnica utilizada.

Lembre-se de que um pipeline bem estruturado deve contemplar, de forma flexível e modular, as principais fases da pesquisa e experimentação em ciência de dados e aprendizado de máquina, incluindo (mas não se limitando a): formulação do problema, coleta e preparação dos dados, análise exploratória, definição de métricas, seleção e validação de modelos, interpretação dos resultados e documentação.

O resultado desta etapa deverá ser um pipeline revisado e justificado, acompanhado de uma breve descrição das alterações realizadas e dos motivos que levaram a cada mudança.

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".
