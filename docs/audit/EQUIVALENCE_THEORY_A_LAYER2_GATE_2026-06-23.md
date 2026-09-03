<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-a-layer2-gate-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-a-layer2-gate-2026-06-23
-->

# Equivalence Theory — Feature A-layer-2: the chaos gate (Lyapunov-gated reassociation)
## Teoria da Equivalência — Recurso A-camada-2: a porta do caos (reassociação condicionada por Lyapunov)

*Date / Data:* 2026-06-23 · *Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis
*Builds on:* `EQUIVALENCE_THEORY_A_LAYER2_SUBSTRATE_2026-06-23.md` (the Lyapunov estimator + chaos manifest) and `..._A1_EXACTNESS_GATE_2026-06-22.md` (the A-1 exactness gate).

---

## EN — Summary

A-1 added the **representation-exactness** axis: f64 reassociation is forbidden by default and permitted only on the opt-in GUM-guided precision-preserving pass (which trades bits for a lower-uncertainty evaluation order). A-layer-2 adds the **chaos axis on top**: on a path that integrates a chaotic vector field, a reassociation-induced rounding difference grows ~ `e^{λt}` (largest Lyapunov exponent `λ > 0`) over the integration horizon, so **no tolerance can contain it** — reassociation is refused **even when the precision-preserving opt-in is enabled**. The chaos gate dominates.

### Mechanism (source → codegen, end to end)

1. **`with Chaotic` effect** (new, effect id 22, `self-hosted/check/effects.sio`) — a function that integrates a chaotic vector field declares it. The tag is *justified* by the chaos-sensitivity manifest (λ > 0), and that λ is *derivable* via `stdlib/math/lyapunov.sio`.
2. **`lower.sio`** maps a `with Chaotic` function to the new **`IR_STRATEGY_PRECISION_PRESERVING_CHAOTIC`** strategy (`ir.sio`).
3. **`opt_cleanup.sio`** runs the epistemic e-graph pass for that strategy (as for plain precision-preserving) but flags its `EgSmallContext.chaotic = true`.
4. **`egraph.sio` `eg_small_saturate_float`** returns immediately (0 merges) when `ctx.chaotic` — refusing ALL float reassociation, overriding `allow_inexact_reassoc`.

The three axes are now orthogonal and composable: *reassociate iff (value-algebra permits) AND (carrier exact OR precision opt-in) AND (NOT chaotic)*.

### Verification

- **Unit test T143d** (in `run_compiler_main_self_tests`, real Madaros codegen): with `allow_inexact_reassoc = true` (the A-1 opt-in that normally permits reassociation) **and** `chaotic = true`, `eg_small_saturate_float` yields **0 merges**. T143b (default blocks) and T143c (opt-in allows) still pass — so the chaos gate is exactly the additional refusal, nothing more.
- **`tests/run-pass/chaotic_effect.sio`**: a `with Chaotic` function type-checks (the effect is recognised and lowers to the chaotic strategy).
- **No regression:** 0 differences vs the checked-in baseline over 14 run-pass files; B/A-1 type-checking tests unchanged. Normal functions never reach the chaotic path (`ctx.chaotic` is set only for the chaotic strategy).

### Honest accounting — "no claims X, delivers Y"

- **The chaos tag is applied manually, justified by the substrate — not auto-read from the manifest.** The compiler does not read the TSV at compile time (that would break determinism). The developer consults the manifest (or runs `lyap_*` to get λ) and annotates `with Chaotic`; the manifest + estimator make that decision *derivable*, not guessed. Auto-tagging from the manifest is a possible future tool, not done here.
- **The gate is verified at the e-graph unit level + effect acceptance.** That "a chaotic function's f64 ops are not reassociated" is not directly observable from a source program (reassociation is internal), so the proof is the unit test on `eg_small_saturate_float` plus the strategy-plumbing being a straight-line mapping. The `gen2 == gen3` fixed point is unaffected (lean_single untouched).
- **Epistemic flag (§8) carried:** using a Lyapunov exponent as a compiler optimisation parameter is a *claim to verify*, not established practice. Stated in the effect comment, the strategy comment, and the manifest.

### Files

- `self-hosted/check/effects.sio`, `self-hosted/ir/ir.sio`, `self-hosted/ir/lower.sio`, `self-hosted/ir/opt_cleanup.sio`, `self-hosted/ir/egraph.sio`, `self-hosted/compiler/main.sio`, `tests/run-pass/chaotic_effect.sio`.

---

## PT — Resumo

A-1 adicionou o eixo de **exatidão de representação**: reassociação f64 proibida por padrão, permitida só no passo opcional preservador de precisão (guiado por GUM). A-camada-2 adiciona o **eixo do caos por cima**: num caminho que integra um campo vetorial caótico, a diferença de arredondamento induzida pela reassociação cresce ~ `e^{λt}` (`λ > 0`), então **nenhuma tolerância a contém** — a reassociação é recusada **mesmo com a adesão preservadora de precisão ativa**. A porta do caos domina.

### Mecanismo (fonte → geração de código)

1. **Efeito `with Chaotic`** (novo, id 22) — declarado por função que integra campo caótico; justificado pelo manifesto de caos (λ > 0), derivável via `stdlib/math/lyapunov.sio`.
2. **`lower.sio`** mapeia `with Chaotic` à nova estratégia **`IR_STRATEGY_PRECISION_PRESERVING_CHAOTIC`**.
3. **`opt_cleanup.sio`** roda o passo epistêmico para essa estratégia mas marca `EgSmallContext.chaotic = true`.
4. **`egraph.sio`** `eg_small_saturate_float` retorna 0 quando `ctx.chaotic` — recusa TODA reassociação, sobrepondo `allow_inexact_reassoc`.

Três eixos ortogonais: *reassociar sse (álgebra permite) E (portador exato OU adesão de precisão) E (NÃO caótico)*.

### Verificação

Teste unitário **T143d** (geração real do Madaros): com `allow_inexact_reassoc = true` E `chaotic = true`, `eg_small_saturate_float` dá **0 fusões**; T143b/T143c seguem passando. `tests/run-pass/chaotic_effect.sio`: função `with Chaotic` passa no check. Sem regressão: 0 diferenças vs baseline em 14 arquivos.

### Prestação de contas honesta — "não prometer X e entregar Y"

- **A etiqueta de caos é aplicada manualmente, justificada pelo substrato — não lida automaticamente do manifesto.** O compilador não lê o TSV em tempo de compilação (quebraria determinismo). O desenvolvedor consulta o manifesto (ou roda `lyap_*` para obter λ) e anota `with Chaotic`; o substrato torna a decisão *derivável*. Auto-tagging é trabalho futuro.
- **A porta é verificada no nível unitário do e-graph + aceitação do efeito.** Que "as operações f64 de uma função caótica não são reassociadas" não é diretamente observável de um programa-fonte; a prova é o teste unitário + o mapeamento direto da estratégia. O ponto fixo `gen2 == gen3` é intocado.
- **Flag epistêmico (§8):** usar expoente de Lyapunov como parâmetro de otimização é *afirmação a verificar*.

### Arquivos

- `self-hosted/check/effects.sio`, `self-hosted/ir/ir.sio`, `self-hosted/ir/lower.sio`, `self-hosted/ir/opt_cleanup.sio`, `self-hosted/ir/egraph.sio`, `self-hosted/compiler/main.sio`, `tests/run-pass/chaotic_effect.sio`.

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; code, tests, and this bilingual note were AI-drafted and human-reviewed. The gate is backed by a unit test running in real Madaros codegen (T143d) and a baseline-diff regression proof. / Desenvolvido com assistência de IA sob direção humana; a porta tem respaldo em teste unitário na geração real do Madaros (T143d) e prova de regressão por diferença com o baseline.
