<!-- docs:meta
topic_id: repo.docs.dissertation.qualification-status-2026-06-13
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.qualification-status-2026-06-13
-->

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
- **Incerteza de model-form (PBPK14 vs PBPK28):** a discrepância well-stirred-14 vs permeability-limited-28 é agora **registrada no ledger de risco epistêmico** `stdlib/epistemic/roi.sio` (`risk_model_form` → `ledger_add_risk`, em **bits** via `log2(1+r)`). ⚠️ **Precisão de terminologia:** isto é uma **métrica de risco/discrepância informacional (bits)**, e **NÃO** uma incerteza-padrão GUM (propagação de variância). Não é "orçamento GUM" no sentido de combinação de variâncias (Contrib. 1/3); é o ledger de débito-de-entropia da Contrib. 2 aplicado a uma fonte de model-form. A escolha da função de agregação é uma decisão de modelagem (não derivada de uma perda decisão-teórica formal), por isso reportamos **duas** agregações e seus valores exatos, sem afirmar invariância:
  - Método A — AUC-ponderada `Σ|AUC28−AUC14|/ΣAUC28` = resíduo 2.798 → **1.925 bits**
  - Método B — janela-de-pico `t≤6h` (exposição-ponderada) = resíduo 1.004 → **1.003 bits**

  Os dois métodos são **estimativas alternativas da MESMA discrepância** de model-form — portanto **NÃO são somadas** (somá-las dupla-contaria uma única fonte). O ledger carrega **um** termo (a estimativa **menor**, logo conservadora, B = janela-de-pico, `risk_total_bits = 1.003`, `risk_count = 1`); os dois métodos **enquadram** sua magnitude num intervalo **[1.0, 1.9] bits**. Os métodos **diferem por ~2,8×** (resíduos 2.798 vs 1.004); só a *existência* de um termo de model-form não-desprezível é estável entre agregações — o *valor* é dependente da agregação, não invariante. A AUC-ponderada integra **todos** os 12 tempos amostrais, incluindo tempos tardios onde a divergência 14/28 cresce e os modelos extrapolam além da janela com dados — por isso ela é maior; a janela-de-pico (`t≤6h`, clinicamente relevante) é a leitura conservadora. Recomputa AUCs trapezoidais das colunas de concentração `c_pbpk28_lit`/`c_pbpk14_wellstirred` — **não** do `delta_pct` (normalizado pelo modelo reduzido); a cauda de **concentração** quase-nula (C~1e-6 em t=30h, onde os 1246% de delta *relativo* só refletem um denominador minúsculo, não exposição clinicamente relevante) é guardada por ponderação de exposição **absoluta** (numerador e denominador ~1e-6 ali, contribuição desprezível vs. os tempos iniciais O(1e-3)). Teste: `tests/run-pass/dissertation_pbpk14_model_form_uc.sio` verde (`bin/souc run`, **6/6, exit 0**, valores conferidos bit-a-bit contra referência Python); revisão matemática xai/grok [OK] nas fórmulas; revisão devil's-advocate (deepseek) endereçada (terminologia GUM→ledger, faixa explícita, caveat de extrapolação). O resíduo >100% reflete divergência real de model-form (o well-stirred de 14 comp. super-prediz AUC tecidual em ~2–5×/órgão), não pequena incerteza.

  **Nota de correção (o motivo antigo era obsoleto):** a redação anterior citava "RK4 do PBPK14 numericamente instável (dt·λ≈3.03; λ≈303)" como bloqueio — **já resolvido**: o solver de **referência** (que gera `model_form_uc.csv`) é Crank-Nicolson A-estável desde o commit `0a6ecac66`, `dissertation_frontend_parity_gate` verde 14/14. O bloqueio real era esta ligação CSV→ledger + a decisão de agregação; o que permanece `future-work` é uma derivação decisão-teórica formal da função de agregação (não um bloqueio de qualificação).
- **GPU:** K-AXI Phase Y valida uma testemunha de 2 compartimentos; PBPK14 Tsit5 em GPU é `future-work`. **NÃO afirmar** "PBPK14 GPU validado".
- **Validação clínica de coorte:** os casos são literatura-ancorados / caso único; coorte prospectiva/retrospectiva é `future-work`.
- **Análise de identificabilidade** dos parâmetros (Kp, PS, fu): `future-work`.
- **`rapamycin_kaxi_fuse_prior` (testemunha de fusão K-AXI):** marcado **PENDING** na suite — depende do subsistema `Seq<T>`, dropado por outro merge (`5f1e397a2`, −9600 linhas, dois backends); restauração = trabalho de Tier-2 separado. A suite registra o pendente honestamente (não mascara).

## 5. Veredito

A qualificação é **defensável** neste snapshot: as três contribuições estão `repo-backed` com 6/6 gates verdes, o caso-líder (vancomicina) roda e tem paper em preparação, o segundo caso (tacrolimo) dá profundidade de validação, e o escopo de trabalho futuro é declarado com honestidade calibrada e rastreado em gate. A barra de qualificação acordada (construtor `Knowledge<T>` + 3 módulos PBPK28 + paper) está cumprida.
