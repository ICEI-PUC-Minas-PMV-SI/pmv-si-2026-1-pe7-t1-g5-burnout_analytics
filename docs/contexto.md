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
O projeto tem potencial de impacto em diferentes perfis estratégicos dentro do ambiente corporativo:

### 1. Gestores de Recursos Humanos (RH) e Lideranças:

- **Contexto profissional**: Responsáveis por gerir equipes e garantir níveis adequados de produtividade e retenção de talentos.

- **Necessidades e expectativas**: Precisam de ferramentas de apoio à decisão para identificar equipes ou perfis em risco de exaustão antes que ocorram afastamentos ou quedas de rendimento.

- **Possíveis barreiras**: Falta de conhecimento técnico em IA, exigindo que o modelo forneça resultados interpretáveis e acionáveis.

### 2. Profissionais e Equipes de Saúde Ocupacional:

- **Contexto profissional**: Médicos do trabalho e psicólogos corporativos focados no bem-estar físico e mental.

- **Necessidades e expectativas**: Utilizar os insights gerados pelas variáveis de saúde (sono, exercício, tempo de tela) para embasar campanhas preventivas direcionadas.

### 3. Alta Gestão e Diretores:

- **Contexto profissional**: Executivos responsáveis pela estratégia geral da empresa, controle de custos operacionais e sustentabilidade do negócio.

- **Necessidades e expectativas**: Necessitam de métricas que demonstrem o risco sistêmico do esgotamento na produtividade global da organização. Buscam utilizar previsões baseadas em dados para justificar o ROI (Retorno sobre Investimento) em políticas de bem-estar, visando a redução de custos com processos trabalhistas e alta rotatividade (turnover).

### 4. Os Próprios Colaboradores (Trabalhadores):

- **Contexto profissional**: Profissionais de diversos setores que vivenciam a rotina corporativa na prática, lidando diretamente com prazos, reuniões e hiperconectividade.

- **Necessidades e expectativas**: Podem se beneficiar do modelo (através de autoavaliações ou ferramentas internas) para ganhar consciência sobre seus próprios limites. Ao visualizarem como seus hábitos afetam o risco predito de burnout, ganham autonomia para buscar equilíbrio na rotina antes de atingirem a exaustão clínica.

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


### 3. Machine Learning for Predicting Burnout Among Healthcare Workers: A Systematic Review and Meta-Analysis. Shi et al. (2025) [4] 

- **Problema e contexto:** Avaliação sistemática do uso de Machine Learning na predição de burnout em profissionais da saúde. 

- **Dataset:** Revisão e meta-análise de múltiplos estudos empíricos internacionais. 

- **Abordagem:** Comparação metodológica entre Random Forest, SVM, Gradient Boosting e Redes Neurais. 

- **Métricas:** AUC-ROC, F1-score e robustez metodológica. 

- **Resultados:** Modelos ensemble e Random Forest apresentaram desempenho consistentemente superior. 

- **Contribuição:** Consolida evidências científicas sobre a eficácia de ML para predição de burnout e aponta lacunas quanto à generalização e interpretabilidade. 

### 4. Burnout Protective Patterns Among Oncology Nurses: A Cross-Sectional Study Using Machine Learning Analysis. Rocha et al. (2025) [3] 

- **Problema e contexto:** Identificação de padrões protetivos contra burnout em enfermeiros oncológicos. 

- **Dataset:** Dados psicossociais e organizacionais coletados em estudo transversal. 

- **Abordagem:** Técnicas de Machine Learning para identificação de padrões e fatores protetivos. 

- **Métricas:** Métricas de classificação e análise de importância das variáveis. 

- **Resultados:** Suporte organizacional e fatores psicossociais mostraram-se determinantes na redução do risco. 

- **Contribuição:** Amplia a aplicação de ML ao identificar não apenas fatores de risco, mas também fatores de proteção. 

### 5. A Machine Learning Model to Predict the Risk Factors Causing Feelings of Burnout and Emotional Exhaustion Amongst Nursing Staff in South Africa. Van Zyl-Cillié et al. (2024) [6] 

- **Problema e contexto:** Predição de fatores associados à exaustão emocional em profissionais de enfermagem. 

- **Dataset:** Dados hospitalares com variáveis de carga horária, ambiente de trabalho e suporte institucional. 

- **Abordagem:** Modelo supervisionado para classificação de risco. 

- **Métricas:** Acurácia, sensibilidade e especificidade. 

- **Resultados:** Sobrecarga de trabalho e suporte organizacional foram variáveis críticas. 

- **Contribuição:** Evidencia a importância do contexto organizacional na modelagem preditiva de burnout. 

### 6. Prediction of Job Burnout in Nurses Based on the Job Demands-Resources Model: An Explainable Machine Learning Approach. Zeng et al. (2025) [7] 

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

Dataset: **[Work Productivity & Burnout Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/shree0910/work-productivity-and-burnout-risk-dataset)**
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

# Variável Alvo

### Burnout_Risk
Problema de classificação binária:
Yes → 1
No → 0

### Volume Estimado
O dataset contém 30.000 registros, que são suficientes para: 
- Divisão treino/validação/teste
- Aplicação de modelos robustos
- Análise de generalização 

### Estrutura e Tratamento dos Dados

- Dataset não apresenta valores faltantes.
- Variáveis categóricas serão codificadas via One-Hot Encoding.
- Verificação de outliers em variáveis contínuas.
- Avaliação de balanceamento da classe alvo.
- Análise de correlação para evitar multicolinearidade excessiva. 

### Desbalanceamento 

Caso a classe "Yes" seja minoritária:
- Ajuste de pesos das classes
- SMOTE
- Avaliação por Recall e F1 
 
### Aspectos Éticos e Conformidade com a LGPD 

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

<img width="1920" height="1080" alt="CanvasAnalitico" src="https://github.com/user-attachments/assets/c984ea6f-9b14-45f8-8051-d128fc2db7ba" />


# Considerações Finais
O projeto propõe uma abordagem estruturada e metodologicamente sólida para predição de risco de burnout utilizando técnicas modernas de Machine Learning. 

A base de 30.000 registros, com variáveis demográficas, ocupacionais e de saúde, permite análise robusta e aplicação de modelos supervisionados com potencial de generalização.

# Vídeo de apresentação da Etapa 01
...

# Referências
1. ADAMOPOULOS, I.; VALAMONTES, A.; TSIRKAS, P.; DOUNIAS, G. Predicting workplace hazard, stress and burnout among public health inspectors: An AI-driven analysis in the context of climate change. *European Journal of Investigation in Health, Psychology and Education*, v. 15, n. 5, p. 65, 2025. DOI: 10.3390/ejihpe15050065. Disponível em: https://doi.org/10.3390/ejihpe15050065. Acesso em: 7 mar. 2026.

2. LIU, C.; CHUANG, Y.-C.; QIN, L.; REN, L.; CHIEN, C.-W.; TUNG, T.-H. Machine-learning-based model for analysing and accurately predicting factors related to burnout in healthcare workers. *BMJ Public Health*, 2025. DOI: 10.1136/bmjph-2023-000777. Disponível em: https://doi.org/10.1136/bmjph-2023-000777. Acesso em: 7 mar. 2026.

3. ROCHA, A.; COSTEIRA, C.; BARBOSA, R.; et al. Burnout protective patterns among oncology nurses: a cross-sectional study using machine learning analysis. *BMC Nursing*, v. 24, p. 805, 2025. DOI: 10.1186/s12912-025-03277-5. Disponível em: https://doi.org/10.1186/s12912-025-03277-5. Acesso em: 7 mar. 2026.

4. SHI, H.; LIU, J.; YANG, C.; SHANG, J.; ZENG, Y. Machine learning for predicting burnout among healthcare workers: a systematic review and meta-analysis. *Contemporary Nurse*, 2025. DOI: 10.1080/10376178.2025.2593294. Disponível em: https://doi.org/10.1080/10376178.2025.2593294. Acesso em: 7 mar. 2026.

5. SHREE0910. Work Productivity and Burnout Risk Dataset. *Kaggle*, [s.d.]. Disponível em: https://www.kaggle.com/datasets/shree0910/work-productivity-and-burnout-risk-dataset. Acesso em: 8 mar. 2026.

6. VAN ZYL-CILLIÉ, M. M.; BÜHRMANN, J. H.; BLIGNAUT, A. J.; DEMIRTAS, D.; COETZEE, S. K. A machine learning model to predict the risk factors causing feelings of burnout and emotional exhaustion amongst nursing staff in South Africa. *BMC Health Services Research*, v. 24, p. 1665, 2024. DOI: 10.1186/s12913-024-12184-5. Disponível em: https://doi.org/10.1186/s12913-024-12184-5. Acesso em: 7 mar. 2026.

7. ZENG, Y.; ZHAO, X.; XIE, Z.; LIN, X.; QI, M.; LI, P. Prediction of job burnout in nurses based on the job demands-resources model: an explainable machine learning approach. *Journal of Advanced Nursing*, 2025. DOI: 10.1111/jan.17071. Disponível em: https://doi.org/10.1111/jan.17071. Acesso em: 7 mar. 2026.
