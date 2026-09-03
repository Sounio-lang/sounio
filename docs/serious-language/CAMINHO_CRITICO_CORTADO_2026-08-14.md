<!-- docs:meta
topic_id: repo.docs.serious-language.caminho-critico-cortado-2026-08-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.serious-language.caminho-critico-cortado-2026-08-14
-->

# Caminho crítico — versão cortada (2026-08-14)

Substitui a versão de 5 tiers. Aquela era um mapa do território; esta é uma rota.

Tier 0 está FEITO e medido (`tests/witness_matrix/`, PR #1737). O medo que a motivava
-- portar 144 commits de uma branch órfã -- evaporou: a `main` já havia fechado 8 dos 10
defeitos por rota própria. Sobraram dois, ambos corrigidos e verificados rodando ELF.

## A jogada que corta mais escopo: assumir `alpha`, não `beta`

`CITATION.cff` declara `1.0.0-beta.6`. Pela tabela do próprio `docs/RELEASE_POLICY.md`:

    alpha  -> souc check passes on canonical fixtures
    beta   -> stdlib reliability gate pass (0 fail)

Com o corpus na casa de ~32% verde, **o rótulo atual já é um overclaim pela política do
repositório**. Descer para `alpha` não é recuo: é fechar uma lacuna existente, e é a regra
do FOUNDER_INTENT (nunca alargar a afirmação além da testemunha) aplicada ao número da
versão. Efeito colateral: remove o corpus inteiro do caminho crítico de lançamento.

## O deliverable único

**Um estranho extrai o tarball e completa o first-user path de 4 comandos, com registro.**

Os três vetores escolhidos (release, paper, site) compartilham uma precondição que nenhum
tem: ninguém além do autor jamais rodou Sounio, e `docs/audit/RELEASE_PACKAGING_E2E_2026-06-07.md`
conclui que o pacote *não funciona como documentado*. Docs sobre um tarball quebrado não são
lançamento; paper sem usuário externo não passa em artifact evaluation.

Conteúdo: portar os 6 bugs de empacotamento já corrigidos na branch órfã (VERSION com
newline no nome do tarball, `release.sh` saindo 1 com tarball bom, `SOUNIO_STDLIB_PATH`
ignorado, smoke check que não compila nada, CLI documentada que binário nenhum implementa),
rebuildar o prebuilt (o enviado é de 25/jul), e assistir uma pessoa de fora rodar sem ajuda.

## Adiado explicitamente, com o porquê

| Item | Por que espera |
|---|---|
| Expandir a conformance spine (16 linhas) | Só importa para o paper, e o paper não vem antes de 22/09 |
| Benchmarks vs Julia/Python | Idem; e o harness aponta para o compilador Rust aposentado |
| Levar o corpus a 100% | Deixa de ser bloqueador no minuto em que se assume `alpha` |
| Investigar o `w5` (arena de IR) | Pré-existente e fail-closed: recusa emitir binário, não corrompe em silêncio |
| Conflito de duas vozes nas docs | Uma tarde de escrita; fazer junto com o release, não antes |
| Ligar o `witness_matrix` no CI | Decisão do mantenedor: muda o que bloqueia merge de todos |

## Sequência

22/09 (submissão da dissertação) é a única data dura do repositório. O lançamento não tem
data -- pela `RELEASE_POLICY.md`, releases são orientados a evento, não a calendário.
Recomendação: o deliverable acima vem DEPOIS da submissão. Ele é pequeno o bastante para
caber antes se houver vontade, mas então é ele sozinho.

---

## CORREÇÃO (2026-08-14, mesmo dia) — o corte acima partia de um número errado

O operador desconfiou do "~32% verde" e estava certo. Medi.

### 1. O corpus, medido hoje: 62%, não 32%

`bash scripts/dev/corpus_census.sh` sobre os 2.538 arquivos versionados:

    stdlib     total=1567   OK=1059   (67%)   REJECT=508   TIMEOUT=0  CRASH=0
    examples   total=971    OK=506    (52%)   REJECT=464   TIMEOUT=0  CRASH=1
    TOTAL      1565 / 2538 aceitos por souc check

O 32% vinha de `docs/EPISTEMIC_RELEASE_STATUS.md` (638/2003) e estava errado de três formas
somadas: é doc da **branch órfã**, medido em **junho**, sobre um corpus **500 arquivos menor**.
Aplicá-lo à main de hoje foi exatamente a falha que esta sessão passou o tempo todo evitando --
e o plano original já dizia que o entregável era ESCREVER o censo, não escolher um dos quatro
números publicados. Escolhi um.

Ressalva do próprio censo: `souc check` não é `compile+run`. Por CLAUDE.md princípio 3, check
mais um caller é o teste de existência para bibliotecas -- mas 62% aceito NÃO significa 62%
executável. É uma pergunta estritamente mais forte e este script não a responde.

### 2. A barra de `beta` não é o corpus -- e o gate nomeado está VERMELHO hoje

A política aponta para um artefato específico. O commitado
(`artifacts/stdlib/stdlib_reliability_status.v1.json`, gerado 2026-05-12) diz:

    totals: pass=251 fail=0 skip=0    status_summary: pass

Ou seja, no papel a barra de beta está cumprida e `1.0.0-beta.6` NÃO é overclaim.
Minha recomendação de descer para alpha estava errada.

Mas re-rodando `bash scripts/stdlib_reliability_gate.sh` hoje: **fail**. O sub-gate de
execução hyper dá `pass=0 fail=7`, e os 7 têm a MESMA causa raiz:

    closure parser incomplete: invalid raw AST node

Atinge nn, onn, qnn, snn, spnn, quantnn e math (hyper). Mais um `golden_mismatch`
(`lane=nn metric=sum_w missing_metric`). É **um defeito, não sete**.

Hipótese não verificada: pode ser da mesma família do `w5` (closure + arena de IR). Não medi.

NÃO commitei o flip pass->fail dos artefatos: a execução foi local, num worktree com o
compilador modificado, e eu não descartei esse confundidor. O número precisa ser reproduzido
em CI limpa antes de virar registro oficial.

### 3. O corte revisado

O corpus SAI do caminho crítico -- não por rebaixar o rótulo, mas porque nunca foi a barra.
A distância para um `beta` defensável é: **uma causa raiz no parser de closures** + uma
métrica golden. Isso é ordens de magnitude menor do que "levar o corpus a 100%".

O deliverable único (um estranho instala e roda) continua valendo, e agora sem a premissa
falsa de que era preciso rebaixar a versão para chegar lá.

---

## ATUALIZAÇÃO (2026-08-14, mesmo dia) — de "um defeito" para 6 de 7 lanes verdes

A seção anterior generalizou cedo demais: dizer "um defeito, não sete" a partir de um
trecho de saída, sem rodar cada lane. Terceira vez no mesmo dia que inferir em vez de
medir me deu um número errado -- registrado aqui porque é padrão, não acaso.

Medido lane por lane, eram **três causas distintas**:

| Causa | Lanes | Tamanho |
|---|---|---|
| Export faltando (E175) + forma de import não suportada | nn, onn, math, spnn, quantnn | 5 lanes |
| Módulo com erros de tipo genuínos (f32/f64, `.sqrt()`, `tanh` inexistente no backend) | qnn, math, snn | sobreposto acima |
| Coerção de literal float→f32 ausente no checker (2 sites: contextual e operando binário) | onn, nn | via o defeito 1 |

3 agentes em paralelo + 1 fix de compilador depois: **6 de 7 lanes compilam, rodam e emitem
o marcador `HYPER_*_OK`**, partindo de 0.

    test_hyper_math_e2e        compile=0 run=0 HYPER_MATH_OK
    test_hyper_qnn_e2e         compile=0 run=0 HYPER_QNN_OK
    test_spiking_e2e           compile=0 run=0 HYPER_SPNN_OK
    test_quantum_e2e           compile=0 run=0 HYPER_QUANTNN_OK
    test_hyper_onn_e2e         compile=0 run=0 HYPER_ONN_OK
    test_hyper_quaternion_e2e  compile=0 run=0 HYPER_NN_OK
    test_snn_e2e               compile=1 NO-ELF

O `golden_mismatch` de `nn sum_w` do relato original também se resolveu sozinho: não era
defeito separado, era o `nn` nunca tendo chegado a rodar. Agora emite `sum_w 2.000000`,
igual ao `expected: 2.0` do gate.

A correção de compilador que destravou 2 das 5 lanes do primeiro grupo (onn, nn) foi a
coerção de literal float→f32 no checker -- que também fecha, do outro lado, o miscompile
do Tier 0 (`as f32` truncava porque não havia caminho de coerção; ver `tests/witness_matrix/`).
Os dois problemas eram o mesmo visto de dois ângulos.

`snn` fica **FAIL_HONEST**, não contornado: falha em codegen nativo (crescer array de
struct via `++`), reduzido a 5 linhas sem imports. Dava para "passar" mudando o tipo
público de `SedenionLayer` para array fixo, mas isso esconderia um defeito de compilador
atrás de uma mudança de API -- não fiz.

### O que isso muda no caminho crítico

A barra de `beta` (`stdlib_reliability_status.v1.json` = pass) está a **uma lane** de
distância, não sete. Mas continua valendo a ressalva anterior: não posso declarar o gate
oficial verde a partir de execução local -- e agora sei por quê de forma mais precisa.
`scripts/stdlib/stdlib_hyper_execution_gate.sh:174` faz
`export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"`, e essa é a
**mesma linha em 50 scripts de CI** -- não bug isolado, padrão deliberado do repositório
que vira armadilha quando `SOUNIO_STDLIB_PATH` está exportado globalmente (como o
próprio `CLAUDE.md` manda fazer fora da raiz). Rodar o gate de um worktree lê a stdlib
do checkout compartilhado. Mudar isso é decisão de arquitetura de CI sobre 50 arquivos,
não conserto de uma linha -- registrado, não executado.

Trabalho consolidado em `worktree-witness-matrix-20260814` (PR #1737), três lanes
paralelas mergeadas sem conflito (conjuntos de arquivos disjuntos) + o fix final.

### ATUALIZAÇÃO 2026-08-15 — SOUNIO_STDLIB_PATH NÃO é bug de CI real

O parágrafo acima ("mudar isso é decisão de arquitetura de CI sobre 50 arquivos")
superestimava o escopo. Medido, não inferido:
`grep -rn SOUNIO_STDLIB_PATH .github/workflows/*.yml` retorna só duas linhas, as
duas em `madaros-prebuilt-refresh.yml`, as duas locais a um único step
(`export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"`) -- não há export global em nenhum
workflow. E cada runner do GitHub Actions é um checkout efêmero e isolado: não
existe "árvore compartilhada" para um export vazar. O padrão
`export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"` nos 50
scripts é seguro em CI -- a variável nunca chega pré-setada de fora.

A armadilha é real, mas só existe no pod interativo multi-agente, onde
`CLAUDE.md` manda exportar `SOUNIO_STDLIB_PATH` fora da raiz do checkout
principal: qualquer worktree que rode um desses 50 scripts sem sobrescrever a
variável lê a stdlib do checkout compartilhado, não a própria. Já mordeu esta
sessão uma vez (a falsa conclusão "pub não é honrado em mod.sio", retraída em
`f0e7869765`) e a lane paralela B independentemente. Escopo do item, portanto:
não é decisão de arquitetura de CI sobre 50 arquivos -- é uma nota de operação
para agentes em worktree, que devem sempre setar `SOUNIO_STDLIB_PATH`
explicitamente para o próprio worktree (ou rodar `unset SOUNIO_STDLIB_PATH`
antes de qualquer `souc`/gate) em vez de confiar no fallback do script.

### ATUALIZAÇÃO 2026-08-15 — a MESMA armadilha existe para `SOUC_BIN`, e ela
### invalidou uma alegação já publicada num commit

O commit `751e495b2c` (fix do arena de IR, mergeado via PR #1741) alegou:
*"Regression-checked: `scripts/run_sio_test_suite.sh closure` gives
byte-identical pass/fail/skip counts against both the pre-fix and post-fix
builds."* Essa alegação está **sem sustentação pelo método descrito** --
medido depois, não antes de publicar, o que é exatamente o tipo de erro que
uma auditoria adversarial pega.

`SOUC_BIN` vem exportado globalmente no pod interativo (`SOUC_BIN=/workspace/sounio/bin/souc`),
e `scripts/lib/resolve_souc.sh`'s `_sounio_resolve_bin()` respeita uma
`SOUC_BIN` pré-setada ANTES de tentar `$ROOT_DIR/bin/souc` do próprio
worktree. Resultado: a suíte `closure` nunca rodou contra o binário
corrigido desta sessão -- rodou contra o `bin/souc` do checkout
compartilhado o tempo todo, independente de `MADAROS_RAW_BIN`. Os números
"idênticos" antes/depois não provavam ausência de regressão; provavam que a
suíte não tinha testado a mudança.

Refeito com `unset SOUC_BIN` (e `MADAROS_RAW_BIN` apontando para o build do
worktree): baseline pré-fix tinha 16 falhas na suíte `closure`; pós-fix tem
15 -- **duas** falhas resolvidas de verdade (`closure_basic.sio`,
`closure_effect_transparent_hof.sio`, ambas o caso simples que w5 testemunha)
e **um** achado novo, não uma regressão: `closure_effect_escape.sio` estava
"passando" (rejeitado corretamente) pelo motivo ERRADO -- o lowering
crashava no mesmo bug de arena antes de alcançar o vazamento de efeito real
que o teste pretende verificar. `souc check` nesse arquivo hoje retorna
`check: OK`; o verificador de efeitos não pega o vazamento de `IO` através
de uma HOF sem anotação. Gap pré-existente em `self-hosted/check/`, não
relacionado ao fix de lowering, documentado como `w17` (não corrigido).

Lição: `SOUNIO_STDLIB_PATH` não é o único var de ambiente do pod que
sombreia um worktree -- `SOUC_BIN` faz o mesmo, e é mais perigoso porque
seu efeito é silencioso (a suíte roda, dá números, só que sobre o binário
errado). Antes de qualquer checagem de regressão neste pod: `unset SOUC_BIN
SOUNIO_STDLIB_PATH` primeiro.
