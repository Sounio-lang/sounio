<!-- docs:meta
topic_id: repo.docs.audit.gum-uncertainty-tail-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: A2 (lane/minimax-cli3/gum-uncertainty-tail-20260819-v2)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.gum-uncertainty-tail-2026-08-19
-->

# GUM/Uncertainty e a cauda do enumerado — tres denominadores separados, sem denominador verdadeiro

Lane: `lane/minimax-cli3/gum-uncertainty-tail-20260819-v2` (census-only; sem alteracoes ao compilador). Data: 2026-08-19. Validacao contra `origin/main` = `f9b3147364`.

## Declaracao semantica (leitura primeiro)

Este documento NAO enumera "os efeitos que o founder desenhou". **Essa lista nao existe em lado nenhum verificavel** — nenhum commit, manifesto, spec ou doc de design a declara como conjunto fechado. O que este documento faz:

1. Mede TRES denominadores SUBSTITUTOS, cada um com a sua razao de ser, e apresenta os racios para cada um.
2. Apresenta a linhagem (primeira/ultima ocorrencia) de 11 nomes epistemicos.
3. Verifica, via probes de reconhecimento, quais dos 11 nomes estao no enumerado de producao (29 ids pos-#1963) e quais ficam fora.
4. Classifica cada nome num dos tres baldes que o despacho pediu: "tentado e falhou a ultima aresta" / "nunca foi tentado" / "nasceu no desenho e desapareceu do codigo".

A interpretacao do que estes numeros dizem sobre o desenho efectivo do founder fica para outra leitura. As tres contagens sao proxies, nao a verdade.

## Os tres denominadores

| Simbolo | Mede | Instrumento | Universo |
|---|---|---|---|
| **D1** | Intencao expressa em codigo | `find_with_prose.py` — regex `\bNAME\b` segue imediatamente `with ` em qualquer posicao (declaracao ou prosa) | `stdlib/`, `self-hosted/`, `examples/`, `tests/` (7492 ficheiros, 164497 ocurrencias) |
| **D2** | Ambicao do dia um | `git archive b6d03ae18a \| tar -x` + `grep -rh` com `\bNAME\b` | Arvore completa do commit fundador (275 ficheiros; stdlib em `.d`) |
| **D3** | O que ficou por exprimir | `grep -rh` com `\bNAME\b` em `docs/`, `README.md`, `FOUNDER_INTENT.md` | So prosa (1288 ficheiros) |

D1 e um instrumento frouxo: conta a palavra `with` seguida de nome, mesmo em comentarios dentro de declaracoes. D2 e tight mas parcial: o `.d` founding tinha so 275 ficheiros; a prosa fundadora e parte do universo. D3 e so prosa actual; pode sobrestimar a prosa fundadora (que migrou de sitio, foi reformatada).

Nenhum dos tres, sozinho, e o universo verdadeiro. Juntar os tres ainda nao e: ha efeitos que foram pensados em conversa ou em issues que nao tocaram nem o repo nem a prosa rastreavel.

## Os 11 nomes (epistemicos + dois designados pelo despacho)

O despacho `dispatch_gum_uncertainty_claude1.md` chamou a atencao para GUM (91 usos reportados) e Uncertainty (20 usos reportados), ausentes de quatro listas reconciliadas na fase 1 da grok-cli5. A adicao `dispatch_epistemic_lineage_claude1.md` estendeu a linhagem a Observe/Witness/Prob. Os onze nomes:

| # | Nome | D1 (atual, `with X`) | D2 (founding b6d03ae18a) | D3 (prosa atual) |
|---:|---|---:|---:|---:|
| 1 | GUM | **7** (so prosa) | **144** em 33 ficheiros | 3037 em 219 ficheiros |
| 2 | Uncertainty | **14** (so prosa) | **156** em 55 ficheiros | 612 em 88 ficheiros |
| 3 | Epistemic | 426 | 137 em 59 ficheiros | 2442 em 199 ficheiros |
| 4 | Observe | 47 | **0** (ausente no founding) | 100 em 30 ficheiros |
| 5 | Witness | 17 | **0** | 214 em 89 ficheiros |
| 6 | Prob | 58 | 55 em 11 ficheiros | 99 em 21 ficheiros |
| 7 | Learn | 19 | 7 em 4 ficheiros | 37 em 15 ficheiros |
| 8 | Temporal | 11 | 7 em 3 ficheiros | 79 em 26 ficheiros |
| 9 | ZD | 87 | **0** | 1380 em 133 ficheiros |
| 10 | NonAssoc | 83 | **0** | 47 em 11 ficheiros |
| 11 | Audit | 11 | 1 em 1 ficheiro | 159 em 116 ficheiros |

Para comparar: D1 capta 50522 `with Mut`, 47660 `with Panic`, 45998 `with Div` — os verdadeiros blocos estruturais — e 426 `with Epistemic` (vs os 316 reportados pelo despacho; a diferenca esta nos matches em comentarios que o loose-instrument apanha). GUM=7 e Uncertainty=14 sao ordens de grandeza abaixo disso: estao em prosa inline em stdlib, nao em assinaturas de efeito.

## Linhagem (primeira / ultima ocorrencia nos paths do codigo)

`git log main --reverse --max-count=1 -G"\bNAME\b"` para primeira, sem `-G` max-count-1 para ultima. Para a primeira, todos os caminhos `stdlib/ self-hosted/ examples/ tests/ docs/`.

| Nome | Primeira ocorrencia | Ultima ocorrencia |
|---|---|---|
| GUM | b6d03ae18a (founding, 2025-12-25) | db750980b4 docs(audit) — forensic dispatch 2026-08-17 |
| Uncertainty | b6d03ae18a | 8999e0fdff WS-C PR1 ENIR/MIR shadow 2026-08-16 |
| Epistemic | b6d03ae18a | 16c45b866c darwin_pbpk Knightian 2026-08-16 |
| Observe | (pos-founding) | 04cc3ef6fc test(witness) 3-field 2026-08-14 |
| Witness | (pos-founding) | 453b2e6e2f feat(mli) S1 kind model 2026-08-16 |
| Prob | b6d03ae18a | 3cac951fa6 docs(trust) monte_carlo 2026-08-06 |
| Learn | b6d03ae18a | 81100d4607 research Mercyful Learning MIMIC-IV 2026-08-03 |
| Temporal | b6d03ae18a | 750f61da40 FFI system() LEMON G2 2026-08-17 |
| ZD | (pos-founding) | 6f2c4e2461 docs(madaros) wave-1 MIR 2026-08-16 |
| NonAssoc | (pos-founding) | 750f61da40 FFI system() LEMON G2 2026-08-17 |
| Audit | b6d03ae18a (1 soa referencia) | 750f61da40 FFI system() LEMON G2 2026-08-17 |

"Pos-founding" = a primeira ocorrencia vivel esta num commit posterior ao dia 1 do projecto (git log --reverse foi lento em casos pontuais; o que importa e que D2 = 0 para esses nomes — eles nao estao no `.d` fundador).

## Reconhecimento no enumerado de producao (29 ids pos-#1963)

`self-hosted/check/effects.sio` enumera 29 ids de producao: `IO, Mut, Alloc, Panic, Div, GPU, Async, Prob, Epistemic (+alias Confidence), Causal, Network, Sensor, Render, Observe, NonAssoc, Audit, Hypothesis, MultiTest, ZD, Witness, Temporal, Learn, Chaotic, Approx, NaturalityG2, Deterministic, Perturbative, NarrowWidthApproximation, NonUnitary`. O `Mod` existe mas e "held (phase 2b)" — nao conta como producao.

| Nome | ID de producao? |
|---|---|
| GUM | **NAO** (sem id; `effect_name_to_id("GUM",3)` retorna -1) |
| Uncertainty | **NAO** |
| Epistemic | id 8 (alias de Confidence tambem) |
| Observe | id 13 |
| Witness | id 19 |
| Prob | id 7 |
| Learn | id 21 |
| Temporal | id 20 |
| ZD | id 18 |
| NonAssoc | id 14 |
| Audit | id 15 |

**Probe de reconhecimento** (`bin/souc run` em ficheiros de 4 linhas):

```sio
fn f() with Epistemic, Mut { }              // epi_run.sio   → PASS
fn main() with Epistemic, Mut { f(); print("...") }

fn f() with GUM, Mut { }                    // gum_run2.sio → PASS
fn main() with GUM, Mut { f(); print("...") }

fn f() with Uncertainty, Mut { }            // unc_run2.sio → PASS
fn main() with Uncertainty, Mut { f(); print("...") }

fn f() with NaoExisteIsto, Mut { }          // nao_run2.sio → PASS (controlo negativo)
fn main() with NaoExisteIsto, Mut { f(); print("...") }
```

Os quatro compilaram, correram e imprimiram o `PASS`. **O parser nao distingue nenhum dos quatro.** A clausula `with X` aceita qualquer identificador sem diagnostico, e o codigo que sai tem o mesmo efeito (nenhum, alem dos ids reais que tiverem sido misturados na mesma clausula).

**Probe de discriminacao** — `f()` requer `Epistemic`; `main()` declara `X, Mut`:

```
main com GUM            → E035 missing Epistemic
main com NaoExisteIsto  → E035 missing Epistemic  (mesmo erro)
main com IO             → E035 missing Epistemic  (controlo positivo)
main com Epistemic      → OK
```

Isto confirma: `with GUM` e `with NaoExisteIsto` contribuem ZERO para a mascara de efeitos do type checker. Identicos entre si. A diferenca so e visivel no `effect_name_to_id` da tabela de bytes (que retorna 0..28 ou -1), e esse id nao parece estar ligado a nada que o utilizador consiga observar em tempo de compilacao.

## Classificacao (os tres baldes do despacho)

| Nome | D1 | D2 | D3 | 29 ids? | Classe |
|---|---|---|---|---|---|
| GUM | so prosa | 144 | 3037 | NAO | **D2-not-D1**: nasceu no desenho (144 occorencias no `.d` fundador) e desapareceu das clausulas `with`. E o caso classico de "tentado e falhou a ultima aresta" — D2 confirma a tentativa, D1 actual mostra a queda. |
| Uncertainty | so prosa | 156 | 612 | NAO | **D2-not-D1**: mesma leitura. 156 occorencias no fundador (mais que o proprio Epistemic!), todas em prosa. Foi pensado; nunca chegou ao compilador. |
| Epistemic | 426 | 137 | 2442 | SIM (id 8) | Vive. Fundador + presente em 426 clausulas `with` hoje. |
| Observe | 47 | 0 | 100 | SIM (id 13) | Adicionado depois do dia 1; presente em 47 `with`. |
| Witness | 17 | 0 | 214 | SIM (id 19) | Adicionado depois; presente em 17 `with`. |
| Prob | 58 | 55 | 99 | SIM (id 7) | Vive desde o dia 1. |
| Learn | 19 | 7 | 37 | SIM (id 21) | Vive desde o dia 1 (embora tenuamente — so 7 occorencias no fundador). |
| Temporal | 11 | 7 | 79 | SIM (id 20) | Vive desde o dia 1. |
| ZD | 87 | 0 | 1380 | SIM (id 18) | Adicionado depois; prosa muito mais densa (1380) do que `with` (87) — o nome vive mais em discussao do que em codigo. |
| NonAssoc | 83 | 0 | 47 | SIM (id 14) | Adicionado depois; mais denso em `with` (83) do que em prosa (47) — o oposto do ZD. |
| Audit | 11 | 1 | 159 | SIM (id 15) | Quase nao estava no fundador (1 occorencia soa); hoje aparece em prosa (159) e em 11 `with`. Adicionado cedo, mas a prosa rebentou depois. |

### O caso limite `GetTid`

O loose-instrument captou `GetTid` 13 vezes em comentarios `// emit: get_tid = ...` em codigo GPU. NAO aparece em qualquer `with` real (so em comentarios de emit). D1 = 13 (prosa). D2 = 0 (founding nao tem GPU). D3 = 1 (so `docs/`). **D3-only**: nunca foi tentado como efeito, so mencionado uma vez em prosa e varias em comentarios de codigo.

### Os tres baldes, formalizados

1. **D1-only (tentado, falhou a ultima aresta)** — nenhum dos 11 nomes cai aqui. Toda a tentativa de efeito que sobreviveu em prosa fundadora migrou ou para a producao ou para o esquecimento total; nenhum ficou num estado intermedirio de "as pessoas ainda usam mas o compilador nao sabe".
2. **D2-not-D1 (nasceu no desenho, desapareceu)** — GUM, Uncertainty. E o unico balde com membros entre os 11; ambos tem D2 massiva (>140 occorencias) e D1 actual zero em clausulas reais.
3. **D3-only (nunca tentado, so escrito)** — `GetTid`. E um membro orfao entre os efeitos; um dia pode aparecer como id, mas hoje e so prosa e comentario.

Fora destes tres, a maioria dos 11 nomes (9) **vive**: estao no enumerado de producao e em clausulas `with` na arvore actual.

## Os racios (um por denominador)

| Denominador | Total | Reconhecidos (29 ids) | Racio |
|---|---:|---:|---:|
| D1 (with X actual) | 11 | 9 (GUM, Uncertainty nao reconhecidos; os outros 9 sim) | **9/11 ≈ 82%** |
| D2 (founding b6d03ae18a) | 7 nomes presentes (GUM, Uncertainty, Epistemic, Prob, Learn, Temporal, Audit) | 5 reconhecidos (Epistemic, Prob, Learn, Temporal, Audit); GUM e Uncertainty nao | **5/7 ≈ 71%** |
| D3 (prosa atual) | 11 | 9 | **9/11 ≈ 82%** |

Para a visao completa: dos 11 nomes, 10 tem `with X` actual ou tem prosa fundadora (Epistemic, Observe, Witness, Prob, Learn, Temporal, ZD, NonAssoc, Audit + um dos GUM/Uncertainty via D2). O racio **"esta no enumerado de producao"** sobe para **10/11 ≈ 91%** se aceitarmos que "tentado" inclui "no commit fundador" alem de "numa clausula `with` actual". Mas esse racio depende de uma decisao de leitura (o que conta como tentativa) que o despacho propos deixar em aberto.

## Claims-Forbidden (nenhum denominador e a verdade)

- **NENHUM destes tres denominadores e "os efeitos que o founder desenhou".** Sao substitutos, nao a lista original. A lista original continua por escrever.
- **A razao 9/11 ≠ "9 em cada 11 efeitos do founder foram reconhecidos".** O denominador e "efeitos que aparecem em codigo ou prosa rastreavel" — uma fracao desconhecida do universo real.
- **A descoberta de que `with GUM` e `with NaoExisteIsto` sao identicos nao prova que GUM e NaoExisteIsto sao identicos.** Prova que o parser nao distingue. O compilador pode ainda dar semantica a GUM num momento posterior; hoje, da perspectiva do type checker, nao da.
- **A leitura "tentado e falhou a ultima aresta" para GUM/Uncertainty e uma leitura de D2 + D1.** Nao exclui a leitura alternativa "nascido em prosa, nunca chegou a ser declarado de verdade mesmo no fundador" — a fronteira entre prosa do fundador e declaracao de efeito do fundador depende do que se considera declaracao.
- **INDETERMINADO** (per precedent do `Mod` phase 2b da minimax-cli2): se um leitor razoavel nao conseguir decidir em qual dos tres baldes um nome cai, **essa decisao fica em aberto** e nao se inventa um quarto balde para forcar caber.
- **Nao se acrescenta nada ao enum** (regra do despacho). Este documento e medicao; nao modifica `self-hosted/check/effects.sio`.

## O que esta medicao NAO diz

- Nao diz qual dos 9 nomes "reconhecidos" foi desenhado pelo founder e qual foi adicionado depois. A tabela de ids em `self-hosted/check/effects.sio` tem historico proprio (#1963 e o commit mais recente que adicionou seis extras) e cada id tem a sua propria primeira ocorrencia — que nao foi rastreada aqui.
- Nao diz nada sobre a semantica dos 29 ids. Medicao de presenca != medicao de uso significativo.
- Nao diz nada sobre efeitos em lanes nao mescladas. Ha branches (ver `docs/audit/BRANCH_AUDIT_2026-08-15.md`) com declaracoes `with X` que nao estao em `main`; esses estao fora deste cenario.

## Coordenacao

- Lane branch: `lane/minimax-cli3/gum-uncertainty-tail-20260819-v2` (sem commits novos — so este ficheiro)
- Coordination bus: `artifacts/omega/agent_handoff.log.md` (NOTIFY pendente apos push)
- PR comment no #1947 (handoff da medicao): pendente
- Coordenacao pedida: grok-cli5 tem o vocabulario de efeitos; esta medicao cruzou com o reconciliado da fase 1 deles (4 listas, GUM/Uncertainty ausentes em todas) sem contradizer
- Regra do founder em vigor: NAO se reverte nada. Commits candidatos `6f23dfe1da` (#1935) e `7be969ed05` (#1939) sob analise da grok-cli3, nao desta lane.

## Anexo: ficheiros do instrumento

- `/tmp/find_with_names.py` — instrumento strict (so declaracoes `fn NAME(...)[with X, Y, Z]`). Validado contra Epistemic.
- `/tmp/find_with_prose.py` — instrumento loose (apanha `with X` em qualquer contexto). Produziu os 164497 tokens / D1 table.
- `/tmp/discrim_{1,2,3}.sio` — 3 programas de discriminacao de efeito (Epistemic exigido em `f()`; `main()` com GUM / NaoExisteIsto / IO). Todos retornaram E035.
- `/tmp/gum_run2.sio`, `/tmp/unc_run2.sio`, `/tmp/nao_run2.sio`, `/tmp/epi_run.sio` — 4 programas de aceitacao de parser (todos compilaram e correram identicamente).
- `/tmp/founding_tree/` — extracao via `git archive b6d03ae18a | tar -x` para inventario D2.
