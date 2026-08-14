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
