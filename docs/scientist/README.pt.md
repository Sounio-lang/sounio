<!-- docs:meta
topic_id: repo.docs.scientist.readme.pt
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.scientist.readme.pt
-->

# Pesquisa em linguagem natural

[English](README.md) · [Ελληνικά](README.el.md) · [Espaço do cientista](https://www.souniolang.org/pt/scientist/)

Comece pela pergunta científica. Descreva em linguagem comum seus dados e sua origem, unidades de medição, hipóteses do modelo e artefatos desejados. Identifique observações ausentes explicitamente e descreva separadamente incerteza de medição e covariância; valor ausente não é zero e covariância ausente não comprova independência. Você não precisa fornecer código ou comandos de terminal.

Escolha uma receita no [Espaço do cientista](https://www.souniolang.org/pt/scientist/), responda às perguntas científicas e leve a solicitação preparada ao LLM escolhido. Peça tabelas, gráficos, relatório de comparação e registro de evidências. O LLM conectado inspeciona o código existente completo, prepara o código e cuida dos comandos reais no ambiente e dentro da autorização fornecidos. Deve relatar incógnitas e recusas do compilador em vez de alterar silenciosamente modelo, unidades, representação da incerteza ou motor.

Um agente local conectado é opcional e pode ler o checkout e executar checagens autorizadas onde houver compilador e dependências compatíveis. Um LLM apenas em chat não consegue executar o fluxo: código, comandos sugeridos e saídas previstas permanecem não executados até que um ambiente conectado devolva evidências reais. Esta interface fornece prompts; não oferece runtime hospedado atual nem executa pesquisa no site.

## Duas etapas, com confirmação do cientista

As ações iniciais de copiar, baixar e visualizar preparam somente seu resumo científico e o protocolo fechado `SOUNIO_INTAKE_V1`, sem código, links de fontes de domínio ou exemplos da linguagem. Cole o pedido na IA escolhida e cole a resposta completa, sem alterações, no campo de esclarecimento. Você não precisa escrever JSON ou código. Use as dicas dos campos e a barra lateral científica para resolver informações ausentes no resumo.

Selecione **Validar esclarecimento**. O validador local verifica sintaxe do recibo e correspondência com os seis campos científicos atuais, tarefa e idioma. Revise o resumo e a resposta correspondente e marque explicitamente a confirmação do cientista. Só então é possível copiar ou baixar instruções da segunda etapa, com o pacote canônico completo de contexto vinculado às fontes. Cada ação de implementação valida novamente a resposta atual. Alterar campo científico, tarefa ou resposta, ou desmarcar a confirmação, revoga a aceitação; valide e confirme novamente. Respostas acima de 60.000 caracteres são rejeitadas.

Este gate não estabelece verdade científica, autoria por LLM, atestação de fontes ou uso real de ferramentas. `source_access: not_verified` e `execution: not_executed` descrevem evidências deste pedido, não tudo que um modelo possa ter lido antes. A confirmação prepara instruções; não executa o estudo nem aprova automaticamente um modelo científico. Escolhas científicas relevantes continuam exigindo revisão do cientista.

## Receitas de pesquisa disponíveis

O [catálogo](../../website/src/data/scientistWorkflows.json) vincula as receitas a códigos existentes em `106f7a988d5b007472d0eead724f58ac50f92200`. Todas têm estado `prompt-template-not-executed`. O campo `starter_task` aponta para uma tarefa existente de aprendizado da linguagem, não para um ID de validação de domínio. Nenhum desses programas de domínio é um ID de exemplo no [manifesto de execução LLM](../llm-guide/example-manifest.json).

| Receita | Entradas científicas → artefatos solicitados | Código e limite da evidência histórica |
|---|---|---|
| Ignição de hidrogênio e incerteza | Mistura, condições iniciais, unidades e incertezas declaradas → relatórios de espécies, atraso de ignição e atribuição de incerteza. | [Demonstração](../../examples/chemistry/h2_ignition_uq_demo.sio); [saída histórica](../../benchmarks/chemistry/golden/h2_ignition_uq_demo.lean_single.txt), selecionada pelo [gate lean_single](../../scripts/ci/chemistry_probe_golden_gate.sh). Saída preservada não é resultado novo do gate nem evidência do Madaros atual. |
| Mecanismo superficial e resposta à pressão | Hipótese de adsorção, pressões, unidades e proveniência dos parâmetros → coberturas, resíduos de convergência e ordens aparentes. | [Demonstração](../../examples/chemistry/h2_surface_reaction_order.sio); [medição histórica](../chemistry/SOUNIO_FOR_SURFACE_MICROKINETICS.md). O código identifica execução gen3 e falha de lowering no Madaros; preservar perfis distintos. Taxas absolutas de perda ausentes não podem ser inventadas. |
| Medições ácido-base e incerteza | Concentrações ou entradas de tampão/titulação, unidades e incerteza de medição → tabela delimitada e limites da implementação. | [Funções](../../stdlib/chemistry/acids.sio), [teste didático](../../tests/stdlib/chemistry/test_acids_stdlib.sio); [auditoria de identidade de funções privadas](../audit/MADAROS_PRIVATE_FN_IDENTITY_2026-09-21.md) documenta correção histórica de importação do pH, não validação metrológica completa. Há expressões de incerteza explicitamente aproximadas; o teste combinado é check-only. |
| Checagem de modelo farmacocinético oral | Hipóteses sintéticas de absorção, eliminação, volume e tempo → trajetória de concentração, resumo de pico e comparação analítica. | [Programa](../../examples/science/pkpd_simulation.sio); [registro histórico de checagem](../audit/USER_FRONTIER_EXAMPLES_CHECK_2026-08-19.tsv) registra apenas sucesso de checagem. O código não tem modelo de resposta farmacodinâmica nem propagação de incerteza. |
| Atualizações sequenciais de ensaio sintético | Desfechos binários sintéticos, prior e hipóteses dos lotes → histórico de atualizações do posterior e explicação dos intervalos. | [Programa](../../examples/science/bayesian_clinical_trial.sio); o mesmo registro histórico registra apenas sucesso de checagem. Os intervalos usam aproximação normal limitada ao domínio; não são quantis exatos do posterior nem desenho de ensaio validado. |
| Distribuição PBPK e auditoria de massa | Perfil sintético, unidades dos compartimentos e limites declarados do modelo → trajetórias teciduais e checagens delimitadas de massa. | [Teste smoke](../../tests/run-pass/darwin_pbpk28_smoke.sio); [censo histórico entre motores](../audit/CROSS_ENGINE_RUNPASS_CENSUS_2026-08-17.md) registra marcadores coincidentes de massa e passos. A [auditoria posterior de massa](../audit/PBPK28_CN_RANNACHER_MASS_BALANCE_2026-09-26.md) explica por que esse teste não estabelece conservação geral. Ler configurações executáveis, não seu cabeçalho desatualizado. |

## O que esperar do assistente

Antes de executar, o agente deve identificar revisão do código, motor e artefato do compilador, alvo e dependências da biblioteca padrão. Exemplos com importações precisam dos provedores reais. Observações históricas de lean_single, gen3 e Madaros têm pré-requisitos distintos e não podem ser substituídas silenciosamente. Entradas não suportadas, taxas ausentes ou covariância indisponível exigem decisão explícita antes da adaptação.

Peça registro das entradas, hipóteses separadas, artefatos de saída nomeados e comandos reais, identidade do código, status de saída e saídas brutas de cada execução autorizada. Separe checagem, compilação, execução, testes, comparação científica e validação clínica. Código copiado, asserções aprovadas e recibos históricos não validam uma adaptação. Nenhum fluxo científico foi executado na preparação deste catálogo.

Use demonstrações sintéticas. Não envie prontuários ou registros de pacientes a serviços públicos ou não aprovados, inclusive pela conexão de modelo de um agente local. A autoridade clínica e de pesquisa permanece com os profissionais responsáveis: estas receitas não fornecem diagnóstico automático, doseamento, prescrição individual ou alegação de prontidão clínica.

Para o agente conectado, use o [guia existente de fluxos LLM](../llm-guide/README.md), o [guia da linguagem em português](../guide/LLM_PROGRAMMING_GUIDE_PT.md) e a [ferramenta de contexto de tarefas](../../scripts/docs/sounio_task_context.mjs). Eles fornecem contexto da linguagem e contrato de evidência; o cientista fornece a pergunta de pesquisa. A geração de prompts pertence ao Espaço do cientista; estas páginas não duplicam um prompt extenso.
