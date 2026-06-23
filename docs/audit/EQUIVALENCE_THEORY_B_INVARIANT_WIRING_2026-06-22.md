<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-b-invariant-wiring-2026-06-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-b-invariant-wiring-2026-06-22
-->

# Equivalence Theory — Feature B wiring: `Invariant<T, G>` enforced in the type-checker
## Teoria da Equivalência — integração do Recurso B: `Invariant<T, G>` aplicado no verificador de tipos

*Date / Data:* 2026-06-22 · *Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis
*Follows:* `EQUIVALENCE_THEORY_B_INVARIANT_SCAFFOLD_2026-06-22.md` (the decidable core).

---

## EN — Summary

The decidable core (`check::invariance`) is now **enforced by the type-checker**: `Invariant<T, G>` observables that are not comparable under their invariance groups are a compile-time type error.

### What was wired (`self-hosted/check/check.sio`)

- **Lowering.** `checker_lower_named_type_with_args_mut` recognises `Invariant<T, G>` (G one of `DiffInf, Diff1, BiLip, Homeo`) and lowers it to a `TyNamed` whose name encodes the group rank — `"Invariant1".."Invariant4"` — with `inner = T`. **No `TypeEntry` field was added**, so the 131-site struct-literal churn is entirely avoided (the same judgement the escape-analysis work made). An unrecognised group name is a type error.
- **Enforcement.** `checker_check_binary_with_operand_types_inplace` decodes both operands' ranks and, when **both** are `Invariant`-tagged, rejects `+ - * / % == != < <= > >=` unless `invariant_comparable_default_ranks(ga, gb)` holds — i.e. unless `frames_comparable(ga, gb, Homeo)`. All lattice semantics stay in `check::invariance` (no duplication); the checker only decodes names and calls it. The diagnostic is the existing E004 *"these types cannot be combined with this operator."*

### Verification (madaros built from this source; `madaros --check`)

| Program | Result |
|---|---|
| `Invariant<f64, Diff1> == Invariant<f64, Diff1>` | **reject** (E004) — Lyapunov-based depths, §4.4 case 1 |
| `Invariant<f64, Homeo> == Invariant<f64, Homeo>` | **accept** — entropy-based depths, §4.4 case 2 |
| `Invariant<f64, Diff1> == Invariant<f64, Homeo>` | **reject** (cross-group) |
| `Invariant<f64, BiLip> == Invariant<f64, BiLip>` | **reject** |
| `f64 + f64` (normal code) | **accept** — no regression |
| `i64 + bool` | **reject** — pre-existing path, unchanged |

Regression sweep: six existing files that fail on the new build fail with the **identical** error codes on the checked-in baseline `bin/madaros-linux-x86_64`, proving the B wiring is **inert for non-`Invariant` code**. Tests added: `tests/compile-fail/invariant_diff1_not_comparable.sio`, `tests/run-pass/invariant_homeo_comparable.sio`.

### Honest accounting — "no claims X, delivers Y"

- **Delivered:** end-to-end type-checker enforcement of group-indexed comparability for `Invariant<T, G>`, reproducing the prompt's §4.4 discrimination, verified in real Madaros codegen, with no regression to ordinary type-checking.
- **Conservative-default semantics (intentional, documented):** with no value-level `FrameId` surfaced yet, the relating transform is taken as the **Homeo default** for *all* cross-value comparisons. Consequence: only `Homeo`-group (topological) observables are comparable; `Diff1`/`BiLip` observables are rejected. This is the safe direction (rejects unproven comparisons) and is exactly the prompt's §4.4 result, but it is a *floor*, not the full frame-relation rule — a true `FrameId` consulting `FrameRegistry` per value is **deferred**.
- **Known wart:** the error message shows the mangled `Invariant2`/`Invariant4` rather than `Invariant<f64, Diff1>`. Cosmetic; a pretty-printer mapping is a follow-up.
- **No Lean obligation emitted/discharged** (the "transform ∈ class" claim) — export surface orphaned; held pending the per-gap decision.

### Follow-up fix (same branch, `compat.sio`) — arithmetic ratio + inner-type soundness

Adversarial review surfaced that the first wiring verified only comparison (`==`); the prompt's headline operation is a **ratio** (`/`). Two defects were found and fixed in `compat.sio` (one decoder, `compat_invariant_rank`, shared with the checker):

1. **§4.4 ratio false-reject (fixed).** An opaque `TyNamed` is not numeric, so `binary_result_type` returned `ty_error()` for `+ - * /` on `Invariant` operands — `Invariant<f64,Homeo> / Invariant<f64,Homeo>` (the §4.4 entropy-ratio) would not type-check. `binary_result_type` now unwraps two `Invariant` operands to their inner `T` (the comparability gate still fires first in `check.sio` for the reject cases).
2. **Inner-`T` erasure / over-acceptance (fixed).** `types_compatible` for `TyNamed` was `name_eq` only, so `Invariant<f64,G>` and `Invariant<i64,G>` (same mangled name) were wrongly compatible. It now also requires the inner `T` to match.

Full verified matrix (madaros built from this source, `madaros --check`): `Homeo / Homeo` and `Homeo - Homeo` → **accept**; `Diff1 / Diff1` → **reject**; `Invariant<f64,Homeo> == Invariant<i64,Homeo>` → **reject** (inner mismatch); `Homeo == Homeo` and normal `f64 + f64` → **accept**. Regression sweep over 12 run-pass files: **0** differences vs the checked-in baseline (the `compat.sio` edits only trigger when **both** operands are `Invariant`-tagged). The "mangled `Invariant2` in the message" wart remains.

---

## PT — Resumo

O núcleo decidível (`check::invariance`) agora é **aplicado pelo verificador de tipos**: observáveis `Invariant<T, G>` não comparáveis sob seus grupos de invariância são erro de tipo em tempo de compilação.

### O que foi integrado (`self-hosted/check/check.sio`)

- **Lowering.** `checker_lower_named_type_with_args_mut` reconhece `Invariant<T, G>` (G ∈ `DiffInf, Diff1, BiLip, Homeo`) e o reduz a um `TyNamed` cujo nome codifica o posto do grupo — `"Invariant1".."Invariant4"` — com `inner = T`. **Nenhum campo foi adicionado a `TypeEntry`**, evitando totalmente a alteração de 131 literais. Nome de grupo não reconhecido é erro de tipo.
- **Aplicação.** `checker_check_binary_with_operand_types_inplace` decodifica os postos de ambos os operandos e, quando **ambos** são `Invariant`, rejeita `+ - * / % == != < <= > >=` a menos que `invariant_comparable_default_ranks(ga, gb)` valha — i.e. `frames_comparable(ga, gb, Homeo)`. Toda a semântica do reticulado permanece em `check::invariance` (sem duplicação). Diagnóstico: E004 *"these types cannot be combined with this operator."*

### Verificação (madaros compilado deste código; `madaros --check`)

Diff1/Diff1 → rejeita; Homeo/Homeo → aceita; Diff1/Homeo → rejeita; BiLip/BiLip → rejeita; `f64 + f64` → aceita (sem regressão); `i64 + bool` → rejeita (caminho pré-existente). Varredura de regressão: seis arquivos que falham na nova build falham com códigos de erro **idênticos** no baseline `bin/madaros-linux-x86_64`, provando que a integração é **inerte para código não-`Invariant`**. Testes: `tests/compile-fail/invariant_diff1_not_comparable.sio`, `tests/run-pass/invariant_homeo_comparable.sio`.

### Prestação de contas honesta — "não prometer X e entregar Y"

- **Entregue:** aplicação completa, no verificador, da comparabilidade indexada por grupo para `Invariant<T, G>`, reproduzindo a discriminação do §4.4, verificada em geração de código real, sem regressão.
- **Semântica conservadora (intencional, documentada):** sem um `FrameId` em nível de valor ainda exposto, a transformação relacionadora é tomada como o **default Homeo** para todas as comparações entre valores. Consequência: só observáveis do grupo `Homeo` (topológicos) são comparáveis; `Diff1`/`BiLip` são rejeitados — direção segura e exatamente o resultado do §4.4, mas é um *piso*, não a regra completa de relação entre referenciais. Um `FrameId` real consultando `FrameRegistry` por valor está **adiado**.
- **Verruga conhecida:** a mensagem de erro mostra `Invariant2`/`Invariant4` mangleados, não `Invariant<f64, Diff1>`. Cosmético; mapeamento de impressão é trabalho futuro.
- **Nenhuma obrigação Lean emitida/descarregada** — superfície órfã; adiado.

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; the wiring, tests, and this bilingual note were AI-drafted and human-reviewed. Behaviour is backed by re-runnable `madaros --check` evidence and a baseline-diff regression proof. / Desenvolvido com assistência de IA sob direção humana; o comportamento tem respaldo em evidência reexecutável `madaros --check` e prova de regressão por diferença com o baseline.
