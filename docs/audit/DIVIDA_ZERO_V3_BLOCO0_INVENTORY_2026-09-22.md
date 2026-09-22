<!-- docs:meta
topic_id: repo.docs.audit.divida-zero-v3-bloco0-inventory-2026-09-22
authority: repo_only
audience: users
last_validated: 2026-09-22
validated_by: claude-divida-zero-v3-bloco0
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.divida-zero-v3-bloco0-inventory-2026-09-22
-->

# Dívida Zero v3 — Bloco 0: instrumentos e inventário (2026-09-22)

Natureza: primeira entrega do plano `SOUNIO_DIVIDA_ZERO_PLANO_v3_2026-09-22` (§5, Bloco 0 /
DZ-00, DZ-01). Este documento é instrumento e inventário — **não** é uma correção de
defeito, não fecha M1/M2/M3, e não afirma que a suíte completa foi executada nesta
sessão. Base: commit `69b7fe7546e8` na branch
`claude/sounio-divida-zero-v3-g7svhy`.

## 1. Recalibração de escala (§1 do CLAUDE.md)

`bash scripts/dev/measure_repo_scale.sh` em `69b7fe7546e8`:

| Métrica | Valor medido nesta sessão | Valor no CLAUDE.md (2026-07-11) | Delta |
|---|---:|---:|---:|
| `.sio` rastreados (arquivos) | 8,584 | 6,130 | +2,454 |
| `.sio` rastreados (linhas, soma `git ls-files`) | 3,345,790 | 2,208,306 | +1,137,484 |
| `self-hosted/` (linhas) | 631,508 | 554,892 | +76,616 |
| `stdlib/` (linhas) | 640,015 | 478,355 | +161,660 |
| `tests/` (linhas) | 305,415 | 236,693 | +68,722 |
| `examples/` (linhas) | 158,544 | 130,370 | +28,174 |
| CI gate scripts (`scripts/ci/`) | 596 | — | — |

**Achado:** a tabela de calibração do `CLAUDE.md` §1 está desatualizada em ~40% (contagem
de linhas) desde 2026-07-11. Isso não bloqueia este Bloco 0, mas é uma obrigação
separada e barata (atualizar §1 do `CLAUDE.md` com a saída deste comando) — registrada
aqui, não corrigida nesta entrega, para não misturar escopo de documentação com escopo
de inventário.

## 2. Hazards de instrumento (§5.4 — controle antes de confiar no instrumento)

Achados sobre as próprias ferramentas de medição, antes de usá-las para julgar código:

### 2.1 `scripts/dev/doctor_workspace.sh` assume `/workspace`

O script abre com `cd /workspace` incondicional (linha 4). Neste checkout
(`/home/user/sounio`, sem `/workspace/sounio`) ele falha imediatamente com
`No such file or directory` antes de emitir qualquer diagnóstico de saúde do
workspace. **Não foi corrigido nesta entrega** — é preservação de comportamento
observado, não um bug de compilador; corrigi-lo é uma mudança de uma linha fora do
escopo declarado deste Bloco (instrumentos de medição de corpus/inventário), e alterar
scripts de outra lane sem coordenação prévia viola §11 do plano v3. Registrado como
achado; publicado no bus de coordenação (`bin/sounio-coord send`, `msg-1790076290-751-10405`).

### 2.2 Seletor "slow" não tem marcador localizável

`grep -n "slow" scripts/run_sio_test_suite.sh` não retornou nenhuma ocorrência.
`grep -rl "//@ slow\|@slow\|requires: slow" tests --include="*.sio"` também não
retornou nenhum arquivo. Isso é consistente com a revisão de #2622 citada no plano v3
§3 e §6.2 ("revalidar o filtro slow" e "corrigir o seletor slow e provar a cobertura de
N e S") — o mecanismo de seleção lenta não está onde um grep ingênuo o esperaria, e
esta sessão não teve tempo de localizar o mecanismo real (pode estar embutido em
harness Sounio, não em shell). **Não resolvido aqui** — fica como pré-requisito
explícito para §5.2/§6.2 antes de qualquer partição N/S ser declarada confiável.

## 3. Amostra do universo do corpus (§5.2) — contagens, não partição qualificada

Contagens brutas em `69b7fe7546e8`, sem execução da suíte (nenhum teste foi rodado
nesta sessão):

| Conjunto amostrado | Contagem | Método |
|---|---:|---|
| `tests/known_failures/*.sio` | 31 | `ls tests/known_failures \| wc -l` |
| Arquivos com `known_failure` no nome fora desse diretório | 7 | `find tests -iname '*known_failure*'` |
| Arquivos citando `known_failure`/`expected_failure`/`XPASS`/`xfail` no corpo | 18 | `grep -rl` sobre `tests/**/*.sio` |
| Arquivos anotados `requires: madaros` | 1,165 | `grep -rl "requires: madaros" tests/**/*.sio` |
| Arquivos `.sio` totais em `tests/` (informativo, do measure_repo_scale) | 4,530 | `scripts/dev/measure_repo_scale.sh` |

Isto **não é** a partição `U = N ⊎ S ⊎ X` exigida por §5.2 — é uma amostra de
instrumentação para dimensionar o problema antes de construir a partição real. A
partição qualificada requer: (a) resolver o achado §2.2 acima, (b) rodar o gerador de
census/datasets mencionado em §6.2, (c) declarar `X` (exclusões contratuais) por
engine/target explicitamente. Nenhuma dessas três etapas foi executada nesta sessão.

## 4. Esqueleto do inventário por identidade (§5.1)

Campos do schema §5.1 aplicados às referências já citadas no plano v3 §3/§14. Os campos
de evidência de execução (`compiler_digest`, `witness_id`, `evidence_receipt`) ficam
`PENDING` — não foram gerados nesta sessão, que não compilou nem executou testes.

| contract_id (local) | capability | referência | base_commit | state | evidence_receipt |
|---|---|---|---|---|---|
| DZ-BLOCO1-2622 | BSS sizing, primal preservation, tuple-borrow local | PR #2622 `35513cf80f22…` | 69b7fe75 | PENDING (não requalificada nesta sessão) | — |
| DZ-BLOCO1-2557 | CI em camadas / artifact compartilhado | PR #2557 `c5248c46436c…` | 69b7fe75 | PENDING | — |
| DZ-BLOCO2-2598 | identidade de função homônima cross-module | PR #2598 `f23be0816d17…` | 69b7fe75 | PENDING | — |
| DZ-BLOCO2/5-2501 | inventário de deltas estabilizados (não integrar monolítico) | PR #2501 `8bfab7e6939a…` | 69b7fe75 | PENDING (revisão em blocos, §7.2 do plano) | — |
| DZ-BLOCO3-2511 | ODE/covariância/modelo | PR #2511 `0d389e07dd98…` | 69b7fe75 | PENDING | — |
| DZ-BLOCO3-2612 | EL+/SNOMED fail-closed | PR #2612 `bf7172c98b66…` | 69b7fe75 | PENDING | — |
| DZ-BLOCO2/3-2615 | array-reference lowering + wide-float coverage | PR #2615 `a80ca04ac4ec…` (ver polaridade anti-f64 apontada em revisão) | 69b7fe75 | PENDING | — |

`owner`/`write_set`/`dependencies` não foram atribuídos — §11 do plano v3 é explícito
que "responsáveis/lane assignments são propostas, não leases efetivamente adquiridos";
este documento não cria leases além do já registrado em `bin/sounio-coord scope` para
esta sessão (`claude`, lane `session-fc7d9b6f-e027-5e0f-89b0--66f0de5791`, intenção
"Bloco 0: inventário e instrumentos, sem build pesado").

## 5. O que este documento não faz

- Não executa `full-test-suite`, `make build`, `make build-madaros`, Contracts, nem
  qualquer build pesado (respeitando a disciplina de concorrência do `CLAUDE.md`,
  seção "Concurrency discipline").
- Não requalifica nenhuma PR citada; não muda estado de merge de nenhuma delas.
- Não corrige `doctor_workspace.sh` nem o seletor slow — apenas os cataloga.
- Não declara M1, M2 ou M3. Não fecha DZ-00 nem DZ-01; abre o instrumento para que
  Bloco 1 (§6 do plano v3) possa correr sobre uma base medida, não presumida.

## 6. Próximo passo declarado

Bloco 1 (§6.1–§6.3 do plano v3): localizar o mecanismo real de seleção `slow`
(resolvendo o achado §2.2 deste documento) antes de qualquer tentativa de declarar a
partição N/S para a candidata baseada na #2622. Esse é um trabalho de leitura/repro
mínima (§6.1 do `CLAUDE.md`, "um blocker sem reprodução mínima não está diagnosticado"),
não um build completo, e deve ser o próximo item desta lane.
