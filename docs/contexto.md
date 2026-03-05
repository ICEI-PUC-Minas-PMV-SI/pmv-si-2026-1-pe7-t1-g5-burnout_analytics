# Introdução

Eventos hidrometeorológicos extremos têm se intensificado nas últimas décadas, especialmente na região Sudeste do Brasil, provocando impactos sociais, econômicos e ambientais significativos. No estado de Minas Gerais, municípios da Zona da Mata apresentam histórico recorrente de enchentes e alagamentos associados a episódios de precipitação intensa. 
 
Diante desse cenário, este projeto propõe o desenvolvimento de um modelo preditivo baseado em técnicas de Machine Learning para estimar o risco de ocorrência de eventos de inundação em dez municípios da Zona da Mata Mineira com maior incidência histórica de enchentes. 
 
A modelagem será construída a partir da integração de dados meteorológicos do Instituto Nacional de Meteorologia (INMET), dados hidrológicos da Agência Nacional de Águas e Saneamento Básico (ANA) e registros oficiais de desastres do Sistema Integrado de Informações sobre Desastres (S2ID). 
 
A abordagem integrada permite modelar a cadeia causal do fenômeno: 
 
Precipitação intensa → Elevação do nível do rio → Registro oficial de desastre 
 
--- 

## Problema

Municípios da Zona da Mata Mineira enfrentam recorrentes eventos de enchentes, gerando impactos à população, infraestrutura urbana e economia local. Embora existam dados meteorológicos e hidrológicos disponíveis, muitas prefeituras ainda não utilizam modelos preditivos integrados para antecipação do risco. 
 
O problema central consiste em verificar se é possível desenvolver um modelo de aprendizado de máquina capaz de prever a ocorrência de eventos de inundação a partir da integração de múltiplas bases de dados públicas. 
 
--- 

## Questão de pesquisa

É possível desenvolver um modelo de aprendizado de máquina capaz de antecipar, com desempenho estatisticamente significativo, a ocorrência de eventos de inundação em municípios da Zona da Mata Mineira, utilizando dados meteorológicos, hidrológicos e registros históricos de desastres? 

---

## Objetivo Geral 

Desenvolver e avaliar um modelo preditivo para estimativa do risco de enchentes em dez municípios da Zona da Mata Mineira, com horizonte de antecedência de 24h a 72h. 
 
## Objetivos Específicos
- Integrar bases de dados meteorológicas, hidrológicas e registros oficiais de desastres;
- Realizar análise exploratória das séries temporais;
- Tratar valores faltantes e inconsistências;
- Construir variáveis derivadas (ex.: precipitação acumulada em 24h, 48h e 72h);
- Definir a variável alvo com defasagem temporal (desastre em t+24h, t+48h e t+72h);
- Aplicar algoritmos de Machine Learning supervisionado;
- Avaliar desempenho por métricas como Accuracy, Precision, Recall, F1-Score e AUC-ROC;
- Comparar resultados entre municípios e entre horizontes de prediçãoho. 
 
--- 

## Justificativa

O estudo se justifica sob três dimensões principais: 
 
### Científica 
 
Ampliação da aplicação de técnicas de Machine Learning para previsão de eventos hidrológicos em escala municipal mineira, com abordagem comparativa regional. 
 
### Social 
 
Potencial contribuição para sistemas de alerta precoce e apoio à Defesa Civil municipal, auxiliando na redução de impactos sociais e econômicos. 
 
### Metodológica 
 
Integração simultânea de três bases públicas nacionais (INMET, ANA e S2ID), permitindo modelar a cadeia completa do fenômeno hidrológico. 
 
A ampliação para múltiplos municípios reduz o risco de viés associado a estudos isolados e aumenta a capacidade de generalização do modelo. 

---

## Público-Alvo

O projeto destina-se principalmente a: 
 
- Defesa Civil municipal; 
- Secretarias de Meio Ambiente; 
- Gestores públicos responsáveis pelo planejamento urbano; 
- Órgãos estaduais de monitoramento ambiental; 
- Pesquisadores em hidrologia e ciência de dados. 
 
Esses stakeholders podem utilizar os resultados do modelo para subsidiar decisões estratégicas e políticas públicas de mitigação de riscos.

---

## Estado da arte

Nesta seção, descreva abordagens da literatura que tratam problemas semelhantes ao seu. Seu objetivo é documentar métodos, dados, métricas e resultados.

### 1. Interpretable machine learning for flood susceptibility mapping in the metropolitan region of São Paulo, Southeast Brazil. Alcântara et al. (2025) [1] 
 
* **Problema e contexto:** Mapeamento de suscetibilidade a enchentes na Região Metropolitana de São Paulo, com foco no apoio à gestão pública urbana. 
* **Dataset:** Dados ambientais, topográficos e hidrológicos da região metropolitana paulista. 
* **Abordagem:** Aplicação de Random Forest, XGBoost, LightGBM e Redes Neurais Artificiais, combinadas com técnicas de Explainable Artificial Intelligence (SHAP e LIME). 
* **Métricas:** F1-score e métricas clássicas de classificação, além de análise de importância das variáveis. 
* **Resultados:** Modelos apresentaram desempenho elevado, com destaque para ganhos interpretativos ao utilizar técnicas de XAI. 
* **Contribuição:** Evidencia a importância da interpretabilidade em modelos aplicados à gestão pública no contexto urbano brasileiro. 

--- 
 
### 2. ML4FF: A machine-learning framework for flash flood forecasting applied to a Brazilian watershed. Ali et al. (2025) [2] 
 
* **Problema e contexto:** Previsão de enchentes rápidas (flash floods) em uma bacia hidrográfica brasileira. 
* **Dataset:** Dados hidrológicos e meteorológicos históricos da bacia analisada. 
* **Abordagem:** Desenvolvimento do framework ML4FF, comparando 34 algoritmos de Machine Learning com validação cruzada e otimização de hiperparâmetros. 
* **Métricas:** NSE (Nash-Sutcliffe Efficiency), RMSE e métricas comparativas de desempenho. 
* **Resultados:** Modelos híbridos e técnicas de aprendizado profundo apresentaram melhor desempenho preditivo. 
* **Contribuição:** Demonstra robustez metodológica e reforça a importância da seleção automatizada de modelos. 
 
--- 
 
### 3. Review and intercomparison of machine learning applications for short-term flood forecasting. Asif et al. (2025) [3] 
 
* **Problema e contexto:** Revisão sistemática sobre previsão de enchentes de curto prazo (1–48 horas). 
* **Dataset:** Análise comparativa de 94 estudos internacionais. 
* **Abordagem:** Avaliação de diferentes técnicas de Machine Learning, incluindo Random Forest, LSTM, ANN e SVM. 
* **Métricas:** Comparação de desempenho reportado na literatura (RMSE, NSE, Accuracy). 
* **Resultados:** Modelos híbridos e abordagens com adequada seleção de variáveis apresentam desempenho superior. 
* **Contribuição:** Fornece base teórica robusta para escolha metodológica no desenvolvimento de modelos preditivos. 
 
--- 
 
### 4. Artificial neural networks applied for flood forecasting in ungauged basin – the Paranaíba river study case. Brandão et al. (2024) [4] 
 
* **Problema e contexto:** Previsão de vazão em bacia não monitorada do Rio Paranaíba (MG), região com histórico de enchentes. 
* **Dataset:** Dados hidrológicos reais provenientes de estação fluviométrica. 
* **Abordagem:** Aplicação de Redes Neurais Artificiais para modelagem de vazão. 
* **Métricas:** NSE (Nash-Sutcliffe Efficiency). 
* **Resultados:** Desempenho satisfatório na previsão de vazões, demonstrando viabilidade da abordagem em contexto mineiro. 
* **Contribuição:** Aplicação prática em Minas Gerais com dados reais, reforçando a relevância regional da modelagem hidrológica baseada em ML. 
 
--- 
 
### 5. Aplicação de técnicas de machine learning na previsão de eventos de alta pluviosidade e riscos de inundação e alagamentos: um estudo de caso em Teófilo Otoni-MG. Brandão, Loureiro e Luiz (2025) [5] 
 
* **Problema e contexto:** Previsão de precipitações severas associadas a riscos de inundação no município de Teófilo Otoni-MG. 
* **Dataset:** Dados meteorológicos históricos reais do município. 
* **Abordagem:** Modelos supervisionados de Machine Learning aplicados à classificação de eventos extremos. 
* **Métricas:** Métricas de classificação e avaliação de desempenho preditivo. 
* **Resultados:** Capacidade satisfatória de antecipação de eventos severos. 
* **Contribuição:** Evidencia viabilidade prática municipal em Minas Gerais, servindo como referência regional. 
 
--- 
 
### 6. Machine learning-based hydrological models for flash floods: a systematic literature. Santos et al. (2025) [6] 
 
* **Problema e contexto:** Revisão sistemática sobre aplicação de modelos hidrológicos baseados em Machine Learning para previsão de flash floods. 
* **Dataset:** Estudos internacionais focados em eventos hidrológicos extremos. 
* **Abordagem:** Análise comparativa de técnicas como Random Forest, SVM, ANN e modelos híbridos. 
* **Métricas:** Síntese das métricas de desempenho reportadas nos estudos analisados. 
* **Resultados:** Identificação de tendências metodológicas e lacunas científicas. 
* **Contribuição:** Aponta necessidade de integração entre variáveis meteorológicas e hidrológicas, além de maior padronização metodológica. 
 
--- 
 
### 7. Deep learning of flood forecasting by considering interpretability and physical constraints. Zhang et al. (2025) [7] 
 
* **Problema e contexto:** Previsão de enchentes em bacia hidrográfica chinesa com horizonte de 1 a 6 horas. 
* **Dataset:** Dados hidrometeorológicos históricos. 
* **Abordagem:** Modelo LSTM com mecanismo de atenção e incorporação de restrições físicas hidrológicas. 
* **Métricas:** NSE superior a 0.90. 
* **Resultados:** Alta precisão preditiva aliada a maior coerência física do modelo. 
* **Contribuição:** Integração entre Deep Learning e conhecimento físico, ampliando confiabilidade científica. 
 
--- 

## Síntese Crítica 

Os trabalhos analisados demonstram consenso quanto à eficácia de técnicas de Machine Learning na previsão de eventos hidrológicos extremos, especialmente Random Forest, Redes Neurais Artificiais e LSTM.

Observa-se evolução recente na incorporação de técnicas de interpretabilidade (Explainable AI) e na integração de restrições físicas aos modelos, aumentando a confiabilidade e aplicabilidade prática. 

Entretanto, identificam-se lacunas relevantes:

- Pouca aplicação comparativa em múltiplos municípios de uma mesma região; 
- Escassez de integração simultânea entre variáveis meteorológicas, hidrológicas e registros oficiais de desastres; 
- Necessidade de validação regional específica para Minas Gerais, especialmente na Zona da Mata Mineira. 
 
O presente projeto diferencia-se ao propor:

- Integração das bases INMET, ANA e S2ID; 
- Uso de rótulos oficiais de desastre como variável alvo; 
- Aplicação comparativa em dez municípios da Zona da Mata Mineira com maior incidência de enchentes; 
- Avaliação de generalização regional do modelo preditivo. 
 
Dessa forma, o projeto preenche as lacunas identificadas ao propor uma abordagem integrada, multirrisco e comparativa, contribuindo metodologicamente para o avanço da predição de enchentes com uso de Machine Learning no contexto brasileiro e, especialmente, mineiro. 
 
--- 

### Ferramentas inteligentes permitidas
Você pode utilizar: Perplexity, SciSpace, Elicit, Research Rabbit, Litmaps.
Use-as para descoberta, organização e triagem de literatura. 

**Atenção:** 
* Sempre acesse a fonte original (PDF/artigo) antes de citar; verifique números e conclusões.
* Registre DOI/URL oficial e dados bibliográficos completos.
* Evite “alucinações” das ferramentas: desconfie de referências sem DOI ou que você não consiga localizar oficialmente.
* Use as ferramentas inteligentes para mapear redes de citação (Research Rabbit), mapas de tópicos (Litmaps), filtrar por período e gerar resumos iniciais (Perplexity/SciSpace/Elicit).
* Leia os trabalhos mais promissores e descarte estudos fora de escopo.

> **Links Úteis**:
> - [Google Scholar](https://scholar.google.com/)
> - [IEEE Xplore](https://ieeexplore.ieee.org/Xplore/home.jsp)
> - [Science Direct](https://www.sciencedirect.com/)
> - [ACM Digital Library](https://dl.acm.org/)

# Descrição do _dataset_ selecionado

## Municípios Analisados
Os dez municípios selecionados da Zona da Mata Mineira são:    
- Juiz de Fora (MG)  
- Ubá (MG)  
- Muriaé (MG)  
- Cataguases (MG)  
- Leopoldina (MG)  
- Viçosa (MG)  
- Manhuaçu (MG)  
- Ponte Nova (MG)  
- Além Paraíba (MG)  
- Carangola (MG)    

## Bases Utilizadas    

O dataset será construído a partir da integração de três bases oficiais:    

### INMET 
– Variáveis Meteorológicas    
- Precipitação  
- Temperatura  
- Umidade  
- Pressão atmosférica  
- Velocidade do vento    

Representa o fator desencadeador primário das enchentes.    

### ANA 
– Variáveis Hidrológicas    
- Nível de rios  
- Vazão  
- Séries históricas fluviométricas    

Permite avaliar a resposta do sistema hídrico à precipitação.    

### S2ID 
– Registro Oficial de Desastres    
- Registros de eventos reconhecidos oficialmente  
- Datas de ocorrência  
- Tipo de evento (inundação, enxurrada, alagamento)  
- Município afetado    

# Estrutura, Qualidade e Tratamento dos Dados
A integração das três bases resultará em um *dataset* consolidado com a estrutura prevista na Tabela 1, contemplando variáveis preditoras (meteorológicas e hidrológicas) e a variável alvo construída com defasagem temporal para diferentes horizontes de predição.    

## Estrutura do Dataset 

| Variável | Fonte | Descrição | Tipo | Tratamento de Faltantes |
|:---|:---|:---|:---|:---|
| `data` | - | Data da observação (dia/mês/ano) | Data | Remoção de registros sem data válida |
| `municipio` | - | Nome do município | Categórico | - |
| `precipitacao_total_diaria_mm` | INMET | Precipitação acumulada nas últimas 24h | Numérico | Interpolação linear para lacunas ≤ 3 dias; remoção se falha superior | 
| `temp_media_compensada_c` | INMET | Temperatura média diária | Numérico | Preenchimento com média histórica do mês | 
| `umidade_relativa_media_pct` | INMET | Umidade relativa do ar média | Numérico | Interpolação linear |  
| `pressao_atmosferica_hPa` | INMET | Pressão atmosférica média | Numérico | Interpolação linear | 
| `velocidade_vento_m_s` | INMET | Velocidade do vento média | Numérico | Interpolação linear |  
| `nivel_rio_cota_cm` | ANA | Nível do rio (cota) no horário de referência | Numérico | Para falhas, utiliza-se o dado da estação mais próxima ou remoção |  
| `vazao_m3_s` | ANA | Vazão do rio | Numérico | Mesmo critério do nível do rio |  
| `precipitacao_acumulada_48h_mm` | INMET (derivada) | Precipitação acumulada nas últimas 48h | Numérico | Calculada apenas se os dias anteriores estiverem disponíveis | 
| `precipitacao_acumulada_72h_mm` | INMET (derivada) | Precipitação acumulada nas últimas 72h | Numérico | Calculada apenas se os dias anteriores estiverem disponíveis | 
| `desastre_em_24h` | S2ID (derivada) | 1 se houve desastre em t+1; 0 caso contrário | Binário | Construída a partir dos registros oficiais | 
| `desastre_em_48h` | S2ID (derivada) | 1 se houve desastre em t+1 ou t+2; 0 caso contrário | Binário | Construída a partir dos registros oficiais | 
| `desastre_em_72h` | S2ID (derivada) | 1 se houve desastre em t+1, t+2 ou t+3; 0 caso contrário | Binário | Construída a partir dos registros oficiais |      

*Tabela 1 – Estrutura prevista do dataset consolidado para predição de enchentes*   

## Volume Estimado    
Considerando um recorte histórico de 10 anos (2015–2025) para os 10 municípios selecionados, estima-se um total aproximado de **36.500 registros diários** (10 municípios × 365 dias × 10 anos). Este volume é adequado para a aplicação de técnicas de aprendizado de máquina e permite a divisão entre conjuntos de treino, validação e teste, respeitando a ordenação temporal das séries.    

## Construção da Variável Alvo (Target)   

Para que o modelo atenda ao objetivo de **antecipação do risco de enchentes**, a variável alvo foi construída com defasagem temporal, permitindo diferentes horizontes de predição alinhados às necessidades da Defesa Civil municipal:    

| Horizonte | Variável Alvo | Descrição | Aplicação Prática |
|:---|:---|:---|:---|
| 24 horas | `desastre_em_24h` | 1 se houve registro de desastre no dia seguinte (t+1); 0 caso contrário | Alerta com 1 dia de antecedência |  
| 48 horas | `desastre_em_48h` | 1 se houve registro de desastre em t+1 ou t+2; 0 caso contrário | Alerta com até 2 dias de antecedência | 
| 72 horas | `desastre_em_72h` | 1 se houve registro de desastre em t+1, t+2 ou t+3; 0 caso contrário | Alerta com até 3 dias de antecedência |    

**Construção a partir do S2ID:**  
A partir dos registros oficiais do Sistema Integrado de Informações sobre Desastres (S2ID), serão considerados exclusivamente os eventos classificados como: 
- Inundação; 
- Enxurrada;  
- Alagamento.   
Para cada município e data de referência `t`, verifica-se a ocorrência de desastres nos dias subsequentes conforme os horizontes definidos. A ausência de registro é interpretada como "não desastre" para o período analisado.   

**Justificativa metodológica:**  A escolha por testar múltiplos horizontes alinha-se às recomendações da literatura ([3], [7]) e permite identificar o melhor equilíbrio entre:  
- **Antecedência:** quanto maior, mais útil para a Defesa Civil no acionamento de protocolos preventivos; 
- **Acurácia:** quanto maior o horizonte, maior a incerteza e a complexidade da predição.    

Os resultados comparativos entre os diferentes horizontes serão analisados para determinar a janela temporal mais adequada ao contexto dos municípios da Zona da Mata Mineira.    ### Qualidade dos Dados e Tratamento de Valores Faltantes   

As bases do INMET e da ANA são conhecidas por apresentarem falhas de medição e séries com dados ausentes, seja por problemas operacionais nas estações, seja por interrupções no registro histórico. Para assegurar a robustez da modelagem, as seguintes estratégias serão adotadas:    

1. **Remoção criteriosa:** dias em que as principais variáveis preditoras (`precipitacao_total_diaria_mm` e `nivel_rio_cota_cm`) estiverem completamente ausentes serão excluídos da amostra, evitando a inserção de ruído no modelo.    

2. **Interpolação temporal:** para lacunas curtas (até 3 dias consecutivos) em variáveis contínuas, será aplicada interpolação linear, considerando a tendência local da série. Esta abordagem é adequada para variáveis climáticas e hidrológicas com forte correlação temporal.    

3. **Médias históricas:** para falhas pontuais em variáveis secundárias (`temp_media_compensada_c`, `umidade_relativa_media_pct`, `pressao_atmosferica_hPa`, `velocidade_vento_m_s`), poderá ser utilizado o valor médio histórico do mesmo mês, calculado a partir dos anos disponíveis na própria estação.    

4. **Estações de referência:** em casos de falhas prolongadas em estações específicas da ANA, será avaliada a possibilidade de utilizar dados da estação fluviométrica mais próxima, desde que localizada no mesmo curso d'água e com correlação estatística comprovada (coeficiente de correlação ≥ 0,7 no período sobreposto).    

5. **Variáveis derivadas:** as acumulações de precipitação em 48h e 72h serão calculadas apenas quando os dias anteriores estiverem disponíveis. Caso haja falha em um dos dias, a variável correspondente será marcada como ausente para aquele registro, evitando a propagação de erros.    

6. **Tratamento da variável alvo:** as variáveis `desastre_em_24h`, `desastre_em_48h` e `desastre_em_72h` não apresentam valores faltantes por construção, uma vez que a ausência de registro no S2ID é codificada como 0 (não desastre). Para os dias no final da série em que não é possível verificar o horizonte completo (ex.: últimos 3 dias para `desastre_em_72h`), os registros serão excluídos da análise para aquele horizonte específico.    

7. **Rastreabilidade e reprodutibilidade:** todas as decisões de tratamento serão registradas em um *log* de transformações, e versões distintas do *dataset* (bruta, tratada, final) serão mantidas no repositório do projeto, garantindo aderência às boas práticas de ciência de dados e permitindo a replicação dos experimentos.    

### Desafio de Desbalanceamento das Classes    
Espera-se que a variável alvo apresente forte desbalanceamento, uma vez que eventos de desastre são raros em comparação com dias sem ocorrência. Este desafio será endereçado por meio de:    
- **Avaliação por métricas adequadas:** priorização de Precision, Recall, F1-Score e AUC-ROC em detrimento da acurácia;  
- **Técnicas de balanceamento:** avaliação da necessidade de aplicar SMOTE, undersampling ou ajuste de pesos das classes durante o treinamento;  
- **Validação temporal:** garantia de que a divisão treino-teste respeite a ordem cronológica, evitando vazamento de dados futuros.    

### Alinhamento com os Objetivos do Projeto    

A estrutura aqui definida está em consonância com os objetivos específicos do projeto:  
- As variáveis derivadas `precipitacao_acumulada_48h_mm` e `precipitacao_acumulada_72h_mm` atendem ao objetivo de "construir variáveis derivadas";  
- As variáveis alvo com diferentes horizontes (`desastre_em_24h`, `desastre_em_48h`, `desastre_em_72h`) permitem avaliar a capacidade preditiva do modelo em diferentes janelas temporais;  
- O tratamento detalhado de valores faltantes garante a qualidade dos dados para as etapas subsequentes de modelagem. 
 
--- 
 
# Aspectos Éticos e Conformidade com a LGPD 

O presente projeto utiliza exclusivamente bases de dados públicas e agregadas em nível municipal (INMET, ANA e S2ID), não havendo coleta, armazenamento ou tratamento de dados pessoais, estando em conformidade com a Lei Geral de Proteção de Dados (LGPD – Lei nº 13.709/2018). Todos os dados são provenientes de fontes oficiais brasileiras e destinam-se exclusivamente a fins de pesquisa e planejamento público. 

Do ponto de vista ético, duas dimensões adicionais foram consideradas: 

i) Viés e justiça algorítmica: a escolha dos dez municípios com maior incidência histórica de enchentes, embora metodologicamente adequada, pode introduzir viés de representação, uma vez que o modelo será otimizado para contextos de alta recorrência de desastres. Caso o modelo venha a ser generalizado para municípios vizinhos com menor incidência, seu desempenho poderá ser distinto. Essa limitação será explicitada na análise dos resultados. 

ii) Impacto social e sensibilidade dos erros de predição: Em cenários de alerta precoce, o custo de um falso negativo (deixar de prever um desastre que ocorre) é significativamente superior ao de um falso positivo (alertar para um evento não concretizado). Por essa razão, métricas como Recall e F1-Score serão priorizadas na avaliação do modelo, em detrimento da acurácia bruta, alinhando a modelagem à aplicação social pretendida. 
Tais preocupações estão alinhadas às boas práticas de pesquisa responsável em Inteligência Artificial e serão mantidas ao longo de todo o ciclo de desenvolvimento do projeto. 
 
--- 

# Canvas analítico

Nesta seção, você deverá estruturar e preencher o seu Canvas Analítico, que tem como objetivo registrar a organização das ideias e apresentar o modelo de negócio do projeto.

O Canvas deve ser preenchido integralmente, mesmo que algumas informações ainda não estejam totalmente definidas. Nessa etapa inicial, é aceitável trabalhar com hipóteses ou estimativas, desde que sejam coerentes com o problema e o contexto definidos.

**Dica:** O Canvas Analítico serve como guia visual para alinhar expectativas e direcionar o desenvolvimento. Ele poderá (e deverá) ser revisitado e atualizado ao longo do projeto.

> **Links Úteis**:
> - [Modelo do Canvas Analítico](https://github.com/ICEI-PUC-Minas-PMV-SI/PesquisaExperimentacao-Template/blob/main/help/Software-Analtics-Canvas-v1.0.pdf)


# Canvas Analítico 

**Problema:**  
Previsão supervisionada de eventos de inundação. 
 
**Stakeholders:**  
Defesa Civil, Prefeituras, órgãos ambientais. 
 
**Dados:**  
Séries meteorológicas + hidrológicas + registros de desastre. 
 
**Técnicas:**  
Random Forest, XGBoost, Regressão Logística e LSTM, aplicadas para predição com horizontes de 24h, 48h e 72h. 
 
**Métricas:**  
Accuracy, F1-Score, Recall e AUC-ROC. 
 
**Impacto Esperado:**  
Apoio à tomada de decisão e mitigação de riscos socioambientais. 
 
---

# Considerações Finais 
 
O projeto propõe uma abordagem integrada para previsão de enchentes na Zona da Mata Mineira, utilizando dados oficiais e técnicas modernas de Machine Learning. 
 
Ao ampliar o escopo para dez municípios, busca-se aumentar a robustez estatística, a capacidade de generalização e o impacto social da pesquisa. 
 
A proposta apresenta viabilidade técnica, relevância científica e aplicabilidade prática, atendendo aos critérios estabelecidos para a Etapa 1.    ---  

---

# Vídeo de apresentação da Etapa 01

Nesta etapa, o grupo deverá produzir um vídeo de 5 a 8 minutos apresentando o trabalho realizado, no qual cada integrante deve dizer seu nome e apresentar uma parte do conteúdo desenvolvido, garantindo que todos participem ativamente da gravação. A ausência de participação de qualquer membro resultará em penalização na nota final desta etapa. Recomenda-se que o grupo elabore previamente um roteiro para organizar a ordem das falas, distribuir o tempo de forma equilibrada e assegurar que todos os tópicos relevantes sejam apresentados de maneira clara e objetiva.

# Referências
1. ALCÂNTARA, Enner; GUIMARÃES, Yasmim C.; BAIÃO, Cheila F. P. *et al.* Interpretable machine learning for flood susceptibility mapping in the metropolitan region of São Paulo, Southeast Brazil. *Discover Geoscience*, v. 3, art. 243, 2025. DOI: https://doi.org/10.1007/s44288-025-00362-9. 

2. ALI, J.; AHMED, A. *et al.* ML4FF: A machine-learning framework for flash flood forecasting applied to a Brazilian watershed. *Journal of Hydrology*, v. 652, 132674, 2025. DOI: https://doi.org/10.1016/j.jhydrol.2025.132674.
  
3. ASIF, Muhammad; KUGLITSCH, Monique M.; PELIVAN, Ivanka. *et al.* Review and intercomparison of machine learning applications for short-term flood forecasting. *Water Resources Management*, v. 39, p. 1971–1991, 2025. DOI: https://doi.org/10.1007/s11269-025-04093-x. 
 
4. BRANDÃO, A. R. A.; MENEZES FILHO, F. C. M.; OLIVEIRA, P. T. S.; FAVA, M. C. Artificial neural networks applied for flood forecasting in ungauged basin – the Paranaíba river study case. *Proceedings of the International Association of Hydrological Sciences*, v. 386, p. 81–86, 2024. DOI: https://doi.org/10.5194/piahs-386-81-2024. 
 
5. BRANDÃO, P. I.; LOUREIRO, G. F.; LUIZ, T. A. Aplicação de técnicas de machine learning na previsão de eventos de alta pluviosidade e riscos de inundação e alagamentos: um estudo de caso em Teófilo Otoni-MG. *Caderno Pedagógico*, v. 22, n. 13, e21735, 2025. DOI: https://doi.org/10.54033/cadpedv22n13-207. 
 
6. SANTOS, Leonardo B. L. *et al.* Machine learning-based hydrological models for flash floods: a systematic literature. *Hydrological Sciences Journal*, 2025. DOI: https://doi.org/10.1007/s44268-025-00071-9. 
 
7. ZHANG, Ting; ZHANG, Ran; LI, Jianzhu; FENG, Ping. Deep learning of flood forecasting by considering interpretability and physical constraints. *Hydrology and Earth System Sciences*, v. 29, p. 5955–5974, 2025. DOI: https://doi.org/10.5194/hess-29-5955-2025.    

8. INSTITUTO NACIONAL DE METEOROLOGIA (INMET). Banco de Dados Meteorológicos para Ensino e Pesquisa – BDMEP. Brasília, 2024. Disponível em: https://bdmep.inmet.gov.br/. Acesso em: 03 mar. 2026.    

9. AGÊNCIA NACIONAL DE ÁGUAS E SANEAMENTO BÁSICO (ANA). Sistema Hidroweb: séries históricas hidrológicas. Brasília, 2024. Disponível em: https://www.snirh.gov.br/hidroweb/. Acesso em: 03 mar. 2026.    

10. BRASIL. Ministério da Integração e do Desenvolvimento Regional. Sistema Integrado de Informações sobre Desastres (S2ID). Brasília, 2024. Disponível em: https://s2id.mi.gov.br/. Acesso em: 03 mar. 2026. 
 
