<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-b-frameid-prettyprint-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-b-frameid-prettyprint-2026-06-23
-->

# Equivalence Theory — Feature B follow-ups: value-level FrameId + error pretty-printer
## Teoria da Equivalência — sequências do Recurso B: FrameId em nível de valor + impressão amigável de erros

*Date / Data:* 2026-06-23 · *Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis
*Follows:* `EQUIVALENCE_THEORY_B_INVARIANT_WIRING_2026-06-22.md`.

---

## EN — Summary

Closes two of the three items the B wiring left deferred (the third, Lean obligation emission, stays held pending the per-gap decision — its export surface is orphaned).

### 1. FrameId — `Invariant<T, G, F>` (makes stronger-than-Homeo invariants usable)

The wiring shipped with a **conservative Homeo default**: with no frame information, every cross-value comparison assumed a merely-topological relation, so only `Homeo`-group observables could ever be combined and `Diff1`/`BiLip` invariants — though constructible — could never be compared. That made them nearly useless.

`Invariant<T, G, F>` adds an optional **frame** type-argument `F` (any nominal tag). The rule (faithful to the prompt's §4.3 `frame: FrameId`):
- **Same frame** (matching `F`) ⇒ the relating transform is the **identity**, which lies in every group ⇒ comparable **at any group**. So two `Diff1` Lyapunov exponents measured in the same frame can now be compared.
- **Different or absent frame** ⇒ fall back to the conservative cross-frame rule (only `Homeo`-group comparable).

Implementation (no new `TypeEntry` field — name-mangling, as before): `Invariant<T,G,F>` lowers to `TyNamed "Invariant<rank>$<frame>"`. `compat.sio` decodes the rank (now length-tolerant) and a new `compat_invariant_same_frame` compares the `$`-suffix. Crucially, `types_compatible` for two Invariants now compares by **group rank + inner T, ignoring the frame**, so two same-group different-frame invariants stay *compatible* and the binop comparability gate alone decides accept/reject from the frame relation. Fully backward-compatible: 2-arg `Invariant<T,G>` is byte-identical to before.

### 2. Error pretty-printer

`print_type_name` now renders the internal mangled carrier as the user wrote it. The diagnostic for a rejected comparison shows `Invariant<f64, Diff1, SubjectA>` (or `Invariant<f64, Diff1>` for the 2-arg form) instead of the opaque `Invariant2` — closing the cosmetic wart the wiring note flagged.

### Verification (`madaros --check`, built from this source)

| Case | Result |
|---|---|
| `Invariant<f64, Diff1, F>` op `Invariant<f64, Diff1, F>` (same frame) | **accept** |
| `Invariant<f64, Diff1, FA>` op `Invariant<f64, Diff1, FB>` (different frame) | **reject** |
| `Invariant<f64, BiLip, F> / …F` (same frame, division) | **accept** |
| `Invariant<f64, Homeo, FA>` op `…FB` (different frame, both Homeo) | **accept** |
| 2-arg regressions: Diff1 reject, Homeo accept, ratio accept, inner-`T` mismatch reject, normal code accept | **all unchanged** |
| Error message | `= expected Invariant<f64, Diff1, SubjectA>` (0 raw `Invariant2` leaks) |

Regression sweep over 14 run-pass files: **0** differences vs the checked-in baseline (the `compat`/`check` edits only act on `Invariant`-tagged types). Tests added: `tests/run-pass/invariant_same_frame_comparable.sio`, `tests/compile-fail/invariant_cross_frame_diff1.sio`.

### Honest accounting

- The frame is a **type-level nominal tag** (a third type-argument), not the value-level `frame: FrameId` field of §4.3's sketch — the checker cannot read runtime values, so frame identity must live in the type to be enforceable. Same-frame comparability and the conservative cross-frame default are exact; a populated `FrameRegistry` of *named cross-frame transform classes* (so e.g. two frames known to be `BiLip`-related allow `BiLip` comparison) remains future work, as does the Lean obligation that would discharge such a class.
- No `TypeEntry` field was added (the 131-site churn is still avoided).

### Files

- `self-hosted/check/compat.sio`, `self-hosted/check/check.sio`, `tests/run-pass/invariant_same_frame_comparable.sio`, `tests/compile-fail/invariant_cross_frame_diff1.sio`.

---

## PT — Resumo

Fecha dois dos três itens que a integração do B deixou adiados (o terceiro, emissão de obrigação Lean, segue retido pela decisão por-lacuna — superfície órfã).

### 1. FrameId — `Invariant<T, G, F>` (torna utilizáveis invariantes mais fortes que Homeo)

A integração saiu com um **default conservador Homeo**: sem informação de referencial, toda comparação entre valores assumia relação meramente topológica, então só observáveis do grupo `Homeo` podiam ser combinados e invariantes `Diff1`/`BiLip` — embora construíveis — jamais comparados. `Invariant<T, G, F>` adiciona um **referencial** opcional `F` (etiqueta nominal). Regra (fiel ao `frame: FrameId` do §4.3): **mesmo referencial** ⇒ transformação identidade (em todo grupo) ⇒ comparável **em qualquer grupo**; **referencial diferente/ausente** ⇒ regra conservadora (só `Homeo`).

Implementação (sem novo campo em `TypeEntry` — mangling de nome): `Invariant<T,G,F>` reduz a `TyNamed "Invariant<rank>$<frame>"`. `types_compatible` para dois Invariant passa a comparar por **posto de grupo + T interno, ignorando o referencial**, de modo que dois invariantes de mesmo grupo em referenciais diferentes seguem *compatíveis* e só a porta de comparabilidade do operador decide. Totalmente retrocompatível: `Invariant<T,G>` (2 args) é idêntico ao anterior.

### 2. Impressão amigável de erros

`print_type_name` agora renderiza o portador como o usuário escreveu: `Invariant<f64, Diff1, SubjectA>` (ou `Invariant<f64, Diff1>`) em vez do opaco `Invariant2` — fechando a verruga cosmética sinalizada.

### Verificação (`madaros --check`)

Mesmo-referencial Diff1 → aceita; referencial-diferente Diff1 → rejeita; mesmo-referencial BiLip `/` → aceita; referencial-diferente Homeo → aceita; regressões de 2 args inalteradas; mensagem mostra `Invariant<f64, Diff1, SubjectA>` (0 vazamentos de `Invariant2`). Varredura de regressão em 14 arquivos: **0** diferenças vs baseline. Testes: `tests/run-pass/invariant_same_frame_comparable.sio`, `tests/compile-fail/invariant_cross_frame_diff1.sio`.

### Prestação de contas honesta

O referencial é uma **etiqueta nominal em nível de tipo** (terceiro argumento), não o campo de valor `frame: FrameId` do §4.3 — o verificador não lê valores em runtime. Comparabilidade de mesmo-referencial e o default conservador são exatos; um `FrameRegistry` populado com *classes de transformação entre referenciais nomeados* (e a obrigação Lean que as descarregaria) segue como trabalho futuro. Nenhum campo foi adicionado a `TypeEntry`.

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; code, tests, and this bilingual note were AI-drafted and human-reviewed. Behaviour is backed by re-runnable `madaros --check` evidence and a baseline-diff regression proof. / Desenvolvido com assistência de IA sob direção humana; comportamento com respaldo em evidência reexecutável `madaros --check` e prova de regressão por diferença com o baseline.
