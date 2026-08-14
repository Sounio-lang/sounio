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
