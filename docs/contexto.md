# Introdução

Nas últimas décadas, transformações no ambiente de trabalho — intensificadas pela digitalização, pela hiperconectividade e pela expansão do trabalho remoto — têm impactado significativamente a saúde mental dos profissionais. O fenômeno do burnout foi reconhecido como fenômeno ocupacional pela World Health Organization, sendo caracterizado por exaustão emocional, redução da eficácia profissional e distanciamento mental do trabalho. 

A crescente sobrecarga digital, o aumento das jornadas, o excesso de reuniões e a pressão por produtividade tornam o ambiente corporativo um espaço propício ao desenvolvimento de estresse crônico. Nesse contexto, técnicas de Machine Learning podem ser aplicadas para identificar padrões comportamentais e ocupacionais associados ao risco de burnout. 

Diante desse cenário, o presente projeto propõe o desenvolvimento de um modelo preditivo supervisionado capaz de estimar o risco de burnout em profissionais, utilizando um dataset estruturado contendo variáveis demográficas, hábitos de trabalho e indicadores de saúde e estilo de vida. 
 
## Problema

Empresas e organizações frequentemente identificam o burnout apenas quando seus efeitos já estão consolidados, como queda abrupta de produtividade, afastamentos médicos ou desligamentos. 

Embora existam indicadores relacionados ao estresse e desempenho, muitas organizações não utilizam modelos preditivos baseados em dados para antecipar o risco. 

O problema central consiste em verificar: 

É possível desenvolver um modelo de aprendizado de máquina capaz de prever, com desempenho estatisticamente significativo, o risco de burnout ocupacional com base em características demográficas, hábitos de trabalho e indicadores de saúde?

## Questão de pesquisa

É possível prever o risco de burnout ocupacional utilizando variáveis relacionadas à jornada de trabalho, ambiente profissional, saúde e estilo de vida, por meio de modelos supervisionados de Machine Learning? 
 
## Objetivo Geral 

Desenvolver e avaliar um modelo preditivo de classificação binária para estimar o risco de burnout em profissionais, utilizando técnicas de aprendizado de máquina supervisionado. 
 
## Objetivos Específicos
- Realizar análise exploratória do dataset; 
- Avaliar qualidade e consistência dos dados; 
- Codificar variáveis categóricas; 
- Verificar correlação entre estresse, produtividade e burnout; 
- Avaliar balanceamento da variável alvo; 
- Aplicar algoritmos supervisionados (Regressão Logística, Random Forest, XGBoost); 
- Avaliar desempenho por métricas como Accuracy, Precision, Recall, F1-Score e AUC-ROC; 
- Analisar importância das variáveis (feature importance); 
- Interpretar resultados sob perspectiva organizacional e social. 

## Justificativa

O estudo se justifica sob três dimensões principais: 
 
### Científica 
Ampliação da aplicação de técnicas de Machine Learning no contexto de saúde ocupacional e predição de risco psicossocial. 

### Social 
Burnout impacta diretamente:
- Saúde mental dos trabalhadores 
- Qualidade de vida 
- Produtividade organizacional 
- Custos com afastamentos 

Modelos preditivos podem apoiar políticas preventivas e estratégias de bem-estar corporativo. 
 
### Organizacional
Empresas podem utilizar modelos semelhantes para: 
- Monitoramento preventivo 
- Programas de qualidade de vida 
- Gestão estratégica de equipes 

## Público-Alvo

O público-alvo deste projeto compreende organizações, gestores de recursos humanos e equipes de saúde ocupacional interessados em identificar e prevenir o risco de burnout entre profissionais em ambiente de trabalho.

Os dados utilizados representam trabalhadores adultos (18+ anos) de diferentes países, setores profissionais e portes de empresas, incluindo variáveis relacionadas a:
- características demográficas (idade, gênero, país);
- perfil profissional (cargo, experiência e porte da empresa);
- hábitos de trabalho (carga horária, reuniões e ambiente de trabalho);
- estilo de vida e saúde (sono, exercício físico, tempo de tela e nível de estresse).

Dessa forma, o modelo desenvolvido poderá apoiar gestores organizacionais e profissionais de saúde ocupacional na identificação precoce de padrões associados ao risco de burnout, possibilitando a implementação de estratégias preventivas voltadas ao bem-estar e à produtividade no ambiente corporativo. 

## Estado da arte

Nesta seção, descreva abordagens da literatura que tratam problemas semelhantes ao seu. Seu objetivo é documentar métodos, dados, métricas e resultados.

### 1. Predicting Workplace Hazard, Stress and Burnout Among Public Health Inspectors: An AI-Driven Analysis in the Context of Climate Change. Adamopoulos et al. (2025) [1] 

- **Problema e contexto:** Predição de risco ocupacional, estresse e burnout em inspetores de saúde pública, considerando a influência de fatores ambientais e mudanças climáticas. 

- **Dataset:** Dados ocupacionais e psicossociais de profissionais do setor público, incluindo variáveis de exposição a riscos e carga de trabalho. 

- **Abordagem:** Aplicação de algoritmos supervisionados de Inteligência Artificial para classificação de risco.

- **Métricas:** Acurácia, precisão, recall e validação cruzada. 

- **Resultados:** Identificação de fatores ambientais e ocupacionais como variáveis preditoras relevantes. 

- **Contribuição:** Demonstra a aplicabilidade de ML em contextos ocupacionais expostos a riscos externos, ampliando o escopo da predição de burnout. 
 

### 2. Machine-Learning-Based Model for Analysing and Accurately Predicting Factors Related to Burnout in Healthcare Workers. Liu et al. (2025) [2] 

- **Problema e contexto:** Identificação e predição de fatores associados ao burnout em profissionais da saúde. 

- **Dataset:** Dados hospitalares com variáveis demográficas, carga horária, experiência profissional e indicadores psicossociais. 

- **Abordagem:** Comparação entre algoritmos supervisionados, com destaque para Random Forest. 

- **Métricas:** Acurácia (~80%), AUC-ROC, precisão e recall. 

- **Resultados:** Experiência profissional e carga de trabalho apresentaram alta relevância preditiva. 

- **Contribuição:** Reforça a eficácia de modelos supervisionados na identificação de fatores críticos de burnout. 


### 3. Machine Learning for Predicting Burnout Among Healthcare Workers: A Systematic Review and Meta-Analysis. Shi et al. (2025) [3] 

- **Problema e contexto:** Avaliação sistemática do uso de Machine Learning na predição de burnout em profissionais da saúde. 

- **Dataset:** Revisão e meta-análise de múltiplos estudos empíricos internacionais. 

- **Abordagem:** Comparação metodológica entre Random Forest, SVM, Gradient Boosting e Redes Neurais. 

- **Métricas:** AUC-ROC, F1-score e robustez metodológica. 

- **Resultados:** Modelos ensemble e Random Forest apresentaram desempenho consistentemente superior. 

- **Contribuição:** Consolida evidências científicas sobre a eficácia de ML para predição de burnout e aponta lacunas quanto à generalização e interpretabilidade. 

### 4. Burnout Protective Patterns Among Oncology Nurses: A Cross-Sectional Study Using Machine Learning Analysis. Rocha et al. (2025) [4] 

- **Problema e contexto:** Identificação de padrões protetivos contra burnout em enfermeiros oncológicos. 

- **Dataset:** Dados psicossociais e organizacionais coletados em estudo transversal. 

- **Abordagem:** Técnicas de Machine Learning para identificação de padrões e fatores protetivos. 

- **Métricas:** Métricas de classificação e análise de importância das variáveis. 

- **Resultados:** Suporte organizacional e fatores psicossociais mostraram-se determinantes na redução do risco. 

- **Contribuição:** Amplia a aplicação de ML ao identificar não apenas fatores de risco, mas também fatores de proteção. 

### 5. A Machine Learning Model to Predict the Risk Factors Causing Feelings of Burnout and Emotional Exhaustion Amongst Nursing Staff in South Africa. Van Zyl-Cillié et al. (2024) [5] 

- **Problema e contexto:** Predição de fatores associados à exaustão emocional em profissionais de enfermagem. 

- **Dataset:** Dados hospitalares com variáveis de carga horária, ambiente de trabalho e suporte institucional. 

- **Abordagem:** Modelo supervisionado para classificação de risco. 

- **Métricas:** Acurácia, sensibilidade e especificidade. 

- **Resultados:** Sobrecarga de trabalho e suporte organizacional foram variáveis críticas. 

- **Contribuição:** Evidencia a importância do contexto organizacional na modelagem preditiva de burnout. 

### 6. Prediction of Job Burnout in Nurses Based on the Job Demands-Resources Model: An Explainable Machine Learning Approach. Zeng et al. (2025) [6] 

- **Problema e contexto:** Predição de burnout em enfermeiros com base no modelo teórico Job Demands–Resources (JD-R). 

- **Dataset:** 3.449 profissionais de enfermagem com dados ocupacionais e psicossociais. 

- **Abordagem:** Modelos supervisionados combinados com técnicas de Explainable Artificial Intelligence (SHAP). 

- **Métricas:** AUC-ROC, F1-score e análise interpretativa das variáveis. 

- **Resultados:** Altas demandas e baixos recursos organizacionais foram determinantes para o risco. 

- **Contribuição:** Integra teoria organizacional e interpretabilidade em ML, reforçando transparência e confiabilidade dos modelos. 

## Síntese Crítica 
A literatura recente (2024–2025) evidencia consolidação do uso de **Machine Learning para predição de burnout**, com predominância de: 

- Modelos supervisionados (Random Forest, SVM, Gradient Boosting); 
- Métodos ensemble com desempenho superior; 
- Ênfase crescente em interpretabilidade (SHAP e XAI); 
- Relevância recorrente de variáveis como carga horária, estresse e suporte organizacional.

Entretanto, observa-se forte concentração dos estudos no setor da saúde, especialmente na enfermagem, o que limita a generalização para outros contextos organizacionais. 

O projeto proposto com o dataset **Work Productivity & Burnout Risk Dataset** diferencia-se por:
- Abranger múltiplos setores profissionais; 
- Incluir variáveis comportamentais e de estilo de vida (sono, exercício, tempo de tela); 
- Permitir comparação entre modelos clássicos e técnicas de interpretabilidade; 
- Possuir base estruturada com 30.000 registros, ampliando robustez estatística. 

Dessa forma, o trabalho alinha-se às tendências metodológicas recentes, ao mesmo tempo em que amplia o escopo empírico da literatura, contribuindo para a predição de burnout em contextos organizacionais mais amplos. 

# Descrição do _dataset_ selecionado

Dataset: **Work Productivity & Burnout Risk Dataset (Kaggle)**
- 30.000 registros
- 17 colunas
- Sem valores ausentes 

# Estrutura, Qualidade e Tratamento dos Dados

## Estrutura do Dataset

| Variável | Tipo | Descrição |
|----------|------|-----------|
| ``Employee_ID`` | Identificador | Código único do funcionário |
| ``Age`` | Numérico | Idade |
| ``Gender`` | Categórico | Gênero |
| ``Country`` | Categórico | País |
| ``Job_Role`` | Categórico | Cargo |
| ``Experience_Years`` | Numérico | Anos de experiência |
| ``Company_Size`` | Categórico | Porte da empresa |
| ``Work_Hours_Per_Day`` | Numérico | Horas trabalhadas por dia |
| ``Meetings_Per_Day`` | Numérico | Reuniões diárias |
| ``Internet_Speed_Mbps`` | Numérico | Velocidade da internet |
| ``Work_Environment`` | Categórico | Remoto, Híbrido ou Presencial |
| ``Sleep_Hours`` | Numérico | Horas de sono |
| ``Exercise_Hours_Per_Week`` | Numérico | Exercício semanal |
| ``Screen_Time_Hours`` | Numérico | Tempo de tela |
| ``Stress_Level`` | Numérico | Escala de estresse |
| ``Productivity_Score`` | Numérico | Escala de 1–100 |
| ``Burnout_Risk`` | Binário | Yes/No |

## Variável Alvo

### Burnout_Risk
Problema de classificação binária:
Yes → 1
No → 0

## Volume Estimado
O dataset contém 30.000 registros, que são suficientes para: 
- Divisão treino/validação/teste
- Aplicação de modelos robustos
- Análise de generalização 

## ESTRUTURA E TRATAMENTO DOS DADOS

- Dataset não apresenta valores faltantes.
- Variáveis categóricas serão codificadas via One-Hot Encoding.
- Verificação de outliers em variáveis contínuas.
- Avaliação de balanceamento da classe alvo.
- Análise de correlação para evitar multicolinearidade excessiva. 

## DESBALANCEAMENTO 

Caso a classe "Yes" seja minoritária:
- Ajuste de pesos das classes
- SMOTE
- Avaliação por Recall e F1 
 
# Aspectos Éticos e Conformidade com a LGPD 

O dataset não contém informações pessoais identificáveis, estando em conformidade com a Lei Geral de Proteção de Dados (LGPD – Lei nº 13.709/2018). 

Dimensões éticas consideradas:
1. Viés algorítmico: Não há discriminação por gênero, país ou porte de empresa.
2. Custo de erro: Falso negativo (não prever burnout) pode gerar impacto maior que falso positivo.
3. Uso responsável: O modelo deve ser ferramenta de apoio e não instrumento punitivo organizacional.

# Canvas Analítico

- **Problema:** Predição supervisionada de risco de burnout.
- **Stakeholders:** RH, gestores organizacionais, equipes de saúde corporativa.
- **Dados:** Demográficos + hábitos de trabalho + saúde + produtividade.
- **Técnicas:** Regressão Logística, Random Forest, XGBoost, SVM.
- **Métricas:** Accuracy, Precision, Recall, F1-Score, AUC-ROC.
- **Impacto Esperado:** Apoio à prevenção de esgotamento profissional e promoção de bem-estar.

# Considerações Finais
O projeto propõe uma abordagem estruturada e metodologicamente sólida para predição de risco de burnout utilizando técnicas modernas de Machine Learning. 

A base de 30.000 registros, com variáveis demográficas, ocupacionais e de saúde, permite análise robusta e aplicação de modelos supervisionados com potencial de generalização.

# Referências
1. ADAMOPOULOS, I.; VALAMONTES, A.; TSIRKAS, P.; DOUNIAS, G. Predicting workplace hazard, stress and burnout among public health inspectors: An AI-driven analysis in the context of climate change. European Journal of Investigation in Health, Psychology and Education, v. 15, n. 5, p. 65, 2025. DOI: 10.3390/ejihpe15050065.

2. LIU, C.; CHUANG, Y.-C.; QIN, L.; REN, L.; CHIEN, C.-W.; TUNG, T.-H. Machine-learning-based model for analysing and accurately predicting factors related to burnout in healthcare workers. BMJ Public Health, 2025. DOI: 10.1136/bmjph-2023-000777.

3. SHI, H.; LIU, J.; YANG, C.; SHANG, J.; ZENG, Y. Machine learning for predicting burnout among healthcare workers: a systematic review and meta-analysis. Contemporary Nurse, 2025. DOI: 10.1080/10376178.2025.2593294.

4. ROCHA, A.; COSTEIRA, C.; BARBOSA, R.; et al. Burnout protective patterns among oncology nurses: a cross-sectional study using machine learning analysis. BMC Nursing, v. 24, p. 805, 2025. DOI: 10.1186/s12912-025-03277-5.

5. VAN ZYL-CILLIÉ, M. M.; BÜHRMANN, J. H.; BLIGNAUT, A. J.; DEMIRTAS, D.; COETZEE, S. K. A machine learning model to predict the risk factors causing feelings of burnout and emotional exhaustion amongst nursing staff in South Africa. BMC Health Services Research, v. 24, p. 1665, 2024. DOI: 10.1186/s12913-024-12184-5.

6. ZENG, Y.; ZHAO, X.; XIE, Z.; LIN, X.; QI, M.; LI, P. Prediction of job burnout in nurses based on the job demands-resources model: an explainable machine learning approach. Journal of Advanced Nursing, 2025. DOI: 10.1111/jan.17071. 

 
