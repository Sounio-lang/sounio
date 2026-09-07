<!-- docs:meta
topic_id: repo.docs.roadmap.pireus-continuity-plan
authority: repo_only
audience: users
last_validated: 2026-09-06
validated_by: Codex
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.roadmap.pireus-continuity-plan
-->

# Plano canônico de continuidade do PIREUS

Data: 2026-09-06. Owner: codex-pireus / continuity-20260906.
Status: execução autorizada. Resultados ficam em tools/pireus/continuity/status.json.
Destino: origin/main por commits, push e PRs encadeados; sem merge automático.

## Objetivo e decisões do fundador
Inkling consulta contexto ontológico congelado e propõe candidatos; Sounio
admite e reconstrói sua semântica; o compilador materializa os aceitos; hardware
mede; contraexemplos alimentam outra rodada.
- Modelo: thinkingmachines/Inkling-Small-NVFP4 nos dois Sparks, TP=2.
- LLM propõe/submete; autoridade de semântica, equivalência e admissão é Sounio.
- Primeiro lowerings Sedenion/XOR; depois operadores novos.
- Geração e benchmark alternam em lotes exclusivos pelo Slurm.
- GRPO é posterior ao ciclo validado; não substitui o Inkling.

## Base e preservação
Main inicial: 6b2b7ff9b0548e5745045388234466fd955757ed.
Linha Spark: 6e2f04100017dce34d3998c41040ccaefe9b8033.
Linha ontologias/V0–V14: efc2ed41c2f7c6e8ef1e3940827aac742d23e2a0.
Branches de preservação: archive/pireus-spark-20260906 e
archive/pireus-ontology-20260906. Inventário em source_inventory.json neste
diretório de ferramentas. Preservar índice e arquivos não rastreados de origem.
Binários grandes permanecem fora de Git, acompanhados de hashes/manifestos.

Em 2026-09-06: Sparks Ready, Slurm idle, Lease slurm-owned, 1 GB10 por nó.
Pacote Walsh: seis arquivos staged; gate reproduzido com rc=0 sob build local
Madaros de 2026-09-04. Isso não é validação de compilador reconstruído de main.
Contratos V13/V14 contêm obrigações abertas; claim_ready não é promovido.

Sequência de integração:
1. Publicar plano/inventário e preservar referências.
2. Revisar matematicamente Walsh, repetir gate e fazer commit próprio.
3. Integrar em checkout isolado de main, transportando dependências em ordem,
   com vínculo aos commits originais; preservar alterações alheias.
4. Conectar stdlib/hardware/pireus à síntese e materialização recentes.
   Catálogo fixo existente permanece referência de regressão.
5. Reconstruir compilador via Foundry/Slurm e resolver pela interface canônica;
   registrar fonte, executável, modo e hashes e repetir gates pertinentes.

## Novo contrato de propostas
V10 histórico permanece válido para seu experimento determinístico; nova
fronteira admite sugestões externas antes do congelamento, sem autoridade LLM.

ResearchContext: operador/atlas, alvo, fatos ontológicos/proveniência, precisão,
leis, gramática, orçamento e hashes.
UntrustedProposal: construção tipada, requisitos, justificativa, contexto/lote,
modelo/revisão; nunca resultado esperado ou autopromoção com autoridade.
AdmissionReceipt: produzido por Sounio, com identidade reconstruída, decisão,
obrigações, contraexemplo/motivo e vínculo à proposta.

Transporte JSONL; preservar requisição/resposta originais. Adaptadores só
transportam/observam. Consultas determinísticas a snapshots geram o contexto.
Fato desconhecido/contraditório recusa a materialização dependente; expansão
da ontologia é hipótese separada.

Primeiro fragmento: Sedenion, dimensão 16, CayleyDicksonSign. Variar seleção de
lanes, movimento de dados, layout e escalonamento na gramática permitida.
Manter ordem de avaliação numérica; precisão, reassociação e FMA modificados
exigem identidade/experimento próprios. Walsh denso em dimensão 16 mantém
recusa por esparsidade; não implica limite assintótico.

Operadores novos: fragmento bilinear inteiro de dimensão 16; Sounio reconstrói
tensor completo e classifica sob equivalência declarada. Novidade relativa ao
atlas, ganho material e novidade científica são resultados separados.
Paridade V13/V14 continua aberta até seus gates específicos fecharem.

## Ciclo e retomada
CLI: prepare, generate, validate, benchmark, report, resume.
Contexto -> geração -> persistência/hash do lote -> admissão/reconstrução
Sounio -> compilação/materialização -> medição -> relatório -> outra rodada.
Diário persistente por contexto/lote/proposta/etapa. Reutilizar somente etapas
com dependências idênticas. Não alterar atlas, gramática, orçamento ou critérios
durante um lote. Feedback só altera o contexto da rodada seguinte.

## Implantação Inkling
Checkpoint fixado por revisão e arquivos; imagem ARM64
lmsysorg/sglang:dev-inkling-small-dgx-spark fixada por digest antes do download.
Base oficial SM121: TP2, atenção Triton, FP4/MoE Marlin,
disable-prefill-cuda-graph. Sem decodificação especulativa na qualificação.
Slurm/srun + Apptainer ARM64 com imagem derivada do OCI. Preparar runtime,
ausente nos workers inspecionados. Provar GPU, RDMA, bibliotecas, memória,
cgroups e encerramento no job antes de carregar o modelo. TP2 sobre ConnectX-7,
não NVLink. Receita do fornecedor não é execução comprovada no Darwin.

Defaults experimentais: texto; concorrência 1; contexto 16384; saída 4096
tokens. Endpoint interno /v1/chat/completions por lote; não depende de residência
no roteador geral. Falha TP2 permanece falha TP2, sem fallback silencioso.
Gerar -> persistir -> encerrar -> liberar -> validar CPU/Xeon -> reservar
benchmark -> medir -> liberar. Não executar GPU por daemon fora da alocação.

## Marcos
| ID | Entrega | Aceitação |
| --- | --- | --- |
| M0 | plano, preservação, Walsh revisado, PR | recuperação pelo origin, pendências explícitas |
| M1 | ontologias, contratos, síntese, compiler path | gates com compilador reconstruído |
| M2 | Inkling TP2 / launcher | 2 ranks, comunicação, respostas, teardown |
| M3 | propostas/ontologia/admissão | proposta real aceita e recusas pela causa correta |
| M4 | ciclo lowerings | ciclo reproduzível, controles e negativos |
| M5 | novos operadores | tensor/classificação reproduzíveis, obrigações explícitas |
| M6 | corpus para avaliar GRPO | holdout separado, recompensa Sounio |

## Testes e critérios
Autoridade: formato inválido, capability inexistente, snapshot errado, precisão
incompatível, resultado esperado injetado e autopromoção devem ser recusados.
Semântica: 256 pares de base/4096 componentes; controles de convenção, ordem,
parentização; casos densos e bordas numéricas. Paridade tensorial não prova
equivalência geral de ponto flutuante.
Infra: perda de rank, timeout, interrupção, comunicação e teardown.
Retomada: interromper entre etapas sem perda/duplicação; mudanças de dependência
invalidam reutilização.

Smoke 8 propostas; piloto 32 por condição em 3 rodadas predefinidas.
Condições: busca determinística, Inkling sem fatos ontológicos de hardware,
Inkling com os fatos; mesmos contratos/problemas/orçamento. Controle fixo:
melhor lowering existente. Medir admissão, diversidade, custo de geração e
validação, desempenho. Lowerings deduplicam por plano normalizado incluindo
layout/escalonamento; operadores por identidade sob equivalência declarada.

Benchmark intercalado candidato/controle, aquecimento, 30 blocos por nó.
Promoção: paridade, ganho mediano >=5%, intervalo de confiança 95% favorável
em ambos Sparks. Congelar método/dados antes da decisão. Nenhum cálculo
externo define um claim Sounio.
Um ciclo correto pode concluir sem ganho/novidade. Não ajustar critérios para
fabricar sucesso. CI, PR, revisão, hardware e prova formal têm estados próprios.
Revisões matemáticas seguem .claude/AGENT_OFFLOAD_POLICY.md.

## Fontes primárias
- https://huggingface.co/thinkingmachines/Inkling-Small
- https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4
- https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-Small
- https://apptainer.org/docs/user/main/gpu.html

## Retomada operacional
Consultar tools/pireus/continuity/status.json, a branch e as claims antes de
escrever. Atualizações substantivas incluem comando, resultado, artefato,
blocker com owner/gate quando necessário, e próxima ação. Não marcar marcos
planejados como PASS.
