# Status de Qualificação — Dissertação de Mestrado (PUC-SP)

**Título:** *GUM-Native Pharmacokinetic Simulation via Epistemic Gradual Compilation (Rapamycin PBPK)*
**Instituição:** PUC-SP · **Defesa:** 22 de setembro de 2026
**Snapshot:** branch `fix/dissertation-confidence-gate` (off `origin/main`), 2026-06-13.

Documento de status para a qualificação. Toda afirmação aqui é ancorada num *gate* executável e reprodutível. Segue o vocabulário do `pbpk_claim_truth_table.md` (`repo-backed` / `experimental` / `future-work`). A disciplina é a mesma da tese: **usar a afirmação mais estreita que os arquivos e gates de fato sustentam.**

## 1. Prontidão — as três contribuições estão `repo-backed` e VERDES

As seis *gates* de dissertação — o backbone de evidência reprodutível da tese — **todas saem com `rc=0` (PASS)** neste snapshot. **Precisão:** cinco gates passam com 100% dos casos; a `pbpk_suite` passa com **50/53 casos ATIVOS verdes + 3 itens PENDING transparentes** (não são falhas — são lacunas conhecidas, declaradas no próprio output da gate; ver §4). Reprodutibilidade verificável: compilador self-host com **fixed-point gen2==gen3 (md5 `f8c51c07`)**; commits desta restauração: `92b66c3b4 697102e21 7a99702db 8ac4c993a 3069f3774 1633536d3 4e2f55c11`.

| Gate | Estado | Sustenta |
|---|---|---|
| `dissertation_pbpk_suite_gate` | ✅ PASS rc=0 (50/53 ativos + 3 PENDING declarados) | Contribuições 1–3 em 4 classes de fármaco |
| `dissertation_pbpk28_parity_gate` | ✅ PASS (9/9, <1% RMSE) | PBPK28 permeabilidade-limitada (Node↔Sounio) |
| `dissertation_pbpk_hessian_gate` | ✅ PASS | GUM de 2ª ordem (Hessiana), Contribuição 3 |
| `dissertation_frontend_parity_gate` | ✅ PASS (14/14 compartimentos) | PBPK14 well-stirred |
| `dissertation_confidence_gate_gate` | ✅ PASS (3/3) | **Contribuição 2** (gates de confiança em compilação) |
| `dissertation_dossier_gate` | ✅ PASS | Geração de artefatos (CSV/markdown) |

- **Contribuição 1 — GUM-through-ODE:** `repo-backed`. Propagação de 1ª ordem no integrador, validada em rapamicina, vancomicina, tacrolimo, tirzepatida.
- **Contribuição 2 — Gates de confiança em tempo de compilação:** `repo-backed`. `with Epistemic(N)` **rejeita em compilação** uma função de dosagem que afirma mais confiança que seu pior prior suporta (over-claim ε=0.65 < N=950 → sem binário). E **aceita** o teto honesto (ε=0.65 ≥ Epistemic(400)). Verificado contra falso-positivo.
- **Contribuição 3 — Orçamentos ISO de incerteza (1ª + 2ª ordem):** `repo-backed`. Hessiana + cumulantes (4ª ordem). O achado cross-droga **não** é a tautologia "fármacos diferem", e sim *qual* fonte domina por classe — CL (rapamicina, lipofílica/hepática), F_oral (tacrolimo, limitada por absorção — confirma e quantifica via GUM o reportado por Staatz & Tett 2004), MIC (vancomicina) — com a **hipótese** prática (a validar, repousando hoje em GUM de 1ª ordem): uma estratégia única de incerteza *pode* ser clinicamente inadequada por classe. A demonstração de que os termos de 2ª ordem mudam *materialmente* o orçamento vs 1ª ordem — e que a diferença afeta decisão — é validação numérica pendente (`future-work`); a implicação clínica NÃO é uma conclusão estabelecida.

Os priors do PBPK28 são `Knowledge[f64]` de fonte única (valor+variância+ε+proveniência), type-enforced — não prosa.

## 2. Casos clínicos — líder + segundo caso aprofundado

- **Caso-líder — Vancomicina (TDM epistêmico):** `repo-backed` + paper em preparação. Caso de paciente real (65a, SCr 1.40±0.14), priors de literatura (Roberts 2011, Matzke 1984), GUM verificado em compilação. Achado: a ponto-estimativa AUC=450 esconde — **sob o modelo Gaussiano de propagação GUM (demonstração metodológica, não teste estatístico nem coorte)** — uma cauda **P(AUC<400)≈13%** que todo sistema de TDM atual reporta como "terapêutico". ⚠️ **Segurança:** este valor é ilustrativo-metodológico (um único caso, sem pré-registro nem correção de multiplicidade) e **NÃO deve ser usado para decisão de dosagem individual** — o ponto é demonstrar que a incerteza é decisão-afetante, não recomendar conduta. Roda (`examples/dissertation_vancomycin_demo.sio`). Paper em **rascunho** (não submetido): referências completadas (Vancouver, verificadas via PubMed) com flags do autor pendentes; submissão é ação do autor. **Não afirmar** "publicado" nem "validado clinicamente".
- **Segundo caso (profundidade) — Tacrolimo:** `repo-backed`. Validação mais rica da tese: contra Jusko 1995 / Kershner 1996 / Staatz 2004 (GMFE≤3.0, Vd_ss≈1300 L, t½≈30 h), DDI tacrolimo+sirolimo (Undre 1999, piso epistêmico Knightiano), e o achado de **dominância de F_oral** (a classe é limitada por absorção, não por depuração — distinto da rapamicina). Provas Lean de monotonicidade existem como *statements* (descarga = trabalho futuro, ver §4).

## 3. ⚠️ Nota de reprodutibilidade — regressões de merge (material de tese)

Achado relevante para o capítulo de honestidade/reprodutibilidade: o backbone de gates **apodreceu na `main`** entre 2026-05-21 e este snapshot, por *merge churn* — o claim histórico "6/6 PASS" estava **stale** (1/6 + suite 0/53). Quatro regressões-keystone foram localizadas (arqueologia de git) e restauradas forensemente:

1. Construtor `Knowledge(v, ε=, prov=)` — dropado por `a55cc97ff` (merge "spine stabilization", −727 linhas); restaurado de `18855a505`.
2. Detecção de over-claim do `EpistemicComplete` — dropada pelo mesmo `a55cc97ff`; o loop de enforcement estava intacto mas faminto (tokens caíam para confiança 1000).
3. Suporte a `[Knowledge[f64]; N]` (Knowledge como elemento de array) — dropado pelo mesmo `a55cc97ff` em 5 sítios do checker.
4. `bin/souc` deveria ser um *wrapper* (dispatch `run`/`check`), sobrescrito por um ELF cru em `ba02961ed` → `souc run` quebrado → todas as gates falhando.

**Lição metodológica (defensável na banca):** os gates não eram "falhas de design" — eram regressões silenciosas de integração, invisíveis porque o próprio runner (`souc run`) estava quebrado. O *processo* de localização foi arqueologia de git manual (não automatizado); o *resultado* é byte-verificável (fixed-point self-host gen2==gen3 a cada passo, md5 `f8c51c07`). Isto fortalece, não enfraquece, o argumento central da tese: *a evidência precisa ser executável e continuamente verificada* — e quando o próprio runner quebra silenciosamente, gates "verdes" históricos podem mascarar regressões.

## 4. Escopo honesto — trabalho futuro (não bloqueia a qualificação)

Declarado explicitamente, ancorado onde a tese já rastreia:

- **Descarga das provas Lean** (tacrolimo/DDI/vancomicina): existem como *statements*; a descarga algébrica é trabalho futuro. **NÃO afirmar** "as provas Lean estão completas".
- **Incerteza de model-form (PBPK14 vs PBPK28):** `model_form_uc.csv` é rastreado mas **não usado no orçamento GUM** — o RK4 do PBPK14 é numericamente instável (dt·λ≈3.03 fora da fronteira de estabilidade; autovalor cerebral λ≈303). Rastreado como **G-α-δ pending**; fix correto = integrador stiff/A-estável. `future-work`.
- **GPU:** K-AXI Phase Y valida uma testemunha de 2 compartimentos; PBPK14 Tsit5 em GPU é `future-work`. **NÃO afirmar** "PBPK14 GPU validado".
- **Validação clínica de coorte:** os casos são literatura-ancorados / caso único; coorte prospectiva/retrospectiva é `future-work`.
- **Análise de identificabilidade** dos parâmetros (Kp, PS, fu): `future-work`.
- **`rapamycin_kaxi_fuse_prior` (testemunha de fusão K-AXI):** marcado **PENDING** na suite — depende do subsistema `Seq<T>`, dropado por outro merge (`5f1e397a2`, −9600 linhas, dois backends); restauração = trabalho de Tier-2 separado. A suite registra o pendente honestamente (não mascara).

## 5. Veredito

A qualificação é **defensável** neste snapshot: as três contribuições estão `repo-backed` com 6/6 gates verdes, o caso-líder (vancomicina) roda e tem paper em preparação, o segundo caso (tacrolimo) dá profundidade de validação, e o escopo de trabalho futuro é declarado com honestidade calibrada e rastreado em gate. A barra de qualificação acordada (construtor `Knowledge<T>` + 3 módulos PBPK28 + paper) está cumprida.
