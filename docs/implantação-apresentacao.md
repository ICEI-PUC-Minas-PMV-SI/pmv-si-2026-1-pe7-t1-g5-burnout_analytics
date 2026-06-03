# Implantação da solução

Nesta etapa, deverá ser realizada e descrita em detalhes a implantação da solução em ambiente de computação em nuvem, incluindo o planejamento da capacidade operacional com base em modelagem matemática e simulação do sistema, a escolha e configuração do provedor de nuvem, o empacotamento e publicação da aplicação, bem como a execução de testes que comprovem seu correto funcionamento.

Importante!! É fundamental que a aplicação em produção (_deploy_ em nuvem) esteja preparada preparada para realizar a inferência dinâmica, utilizando os modelos previamente treinados, com interface para entrada de novos dados fornecidos pelo usuário, de modo que as **predições sejam realizadas em tempo de execução**, sem reprocessamento ou atualização do treinamento.

# Apresentação da solução

Nesta seção, deve ser produzido um vídeo de até 15 minutos apresentando o escopo geral do projeto, um resumo das etapas desenvolvidas, a demonstração da solução publicada e as conclusões finais, destacando aprendizados, impacto e possibilidades de melhorias.

# É IMPRESCINDÍVEL: 
* Atualizar o arquivo **CITATION.cff** disponível no diretório raiz do repositório
* Atualizar as **Instruções de utilização** no arquivo read.me

---

# Implantação da solução

## Planejamento da Capacidade Operacional

A implantação da solução foi planejada considerando o caráter acadêmico do projeto e a necessidade de disponibilizar a aplicação para acesso remoto e demonstração das funcionalidades desenvolvidas.

Como o modelo de Machine Learning já foi previamente treinado durante a fase de desenvolvimento, o ambiente de produção será utilizado apenas para execução das inferências. Dessa forma, não há necessidade de recursos computacionais elevados, uma vez que o treinamento do modelo não ocorrerá em produção.

A infraestrutura foi dimensionada para atender aos seguintes requisitos:

 * Hospedagem da aplicação web;
 * Disponibilização da API responsável pelas predições;
 * Execução das inferências em tempo real;
 * Baixo consumo de memória durante a inferência;
 * Disponibilidade para acesso remoto durante apresentações e avaliações do projeto.

Dessa forma, uma configuração básica de hospedagem em nuvem é suficiente para atender às necessidades da aplicação.

## Escolha do Provedor de Nuvem

Para o deploy da aplicação foi escolhido o serviço de computação em nuvem da Microsoft Azure.

A escolha do Azure foi motivada pelos seguintes fatores:

  * Disponibilidade de hospedagem para aplicações web;
  * Suporte nativo a aplicações desenvolvidas em Python;
  * Facilidade de integração com repositórios Git;
  * Ambiente amplamente utilizado no mercado;
  * Disponibilidade de recursos suficientes para projetos acadêmicos.

Como o objetivo do projeto é apenas disponibilizar a aplicação para acesso via navegador, não foi necessária a utilização de serviços avançados de escalabilidade ou infraestrutura distribuída.
