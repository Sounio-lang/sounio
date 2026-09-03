<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-lean-persite-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-lean-persite-2026-06-23
-->

# Equivalence Theory — per-call-site Lean obligation emission
## Teoria da Equivalência — emissão de obrigações Lean por local de chamada

*Date / Data:* 2026-06-23 · *Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis
*Refines:* `EQUIVALENCE_THEORY_LEAN_AUTOEMIT_2026-06-23.md` (the lexical, per-feature emitter).

---

## EN — Summary

The first auto-emitter scanned the source lexically and emitted one obligation per *feature*. This refines it to **per call-site**: the obligations are now recorded by the **checker** itself — each `Invariant` comparison's concrete groups + frame relation + outcome, and each `with Chaotic` function — so the emitted set reflects exactly what the program does.

### Mechanism

- **Recording side-table** in `check/effects.sio` (module globals + `pub` accessors): `equiv_begin_recording` / `equiv_end_recording`, `equiv_record_invariant(rank_l, rank_r, same_frame, comparable)` (deduplicated), `equiv_set_chaotic`, and `equiv_emit_body` (builds the Mathlib-free `.lean` text). Recording is gated by `EQUIV_RECORDING`, set only during `emit-lean-obligations`.
- **Checker hooks** in `check.sio`: the binary-op Invariant rule records each comparison's `(inv_l, inv_r, same_frame, comparable)`; `checker_check_fn_item_inplace` records a `with Chaotic` (effect id 22) function.
- **Emit mode** in `main.sio`: `emit-lean-obligations` now runs the checker with recording enabled (instead of a lexical scan), then writes `equiv_emit_body`.

### What it emits

For each distinct call-site configuration, a per-site obligation referencing the discharged definitions in `formal/EquivalenceTheory.lean`:
- different/absent frame → `comparableDefault Group.<gl> Group.<gr> = <bool>`
- same frame → `framesComparable Group.<gl> Group.<gr> Group.DiffInf = <bool>`
- any `with Chaotic` function → `reassocAllowed true true true = false`

On `examples/equivalence_theory/lean_obligation_demo.sio` (cross-frame Diff1, same-frame Diff1, Homeo, BiLip, and a chaotic step):

```lean
import EquivalenceTheory
open Sounio.EquivalenceTheory

theorem obl_site_0 : comparableDefault Group.Diff1 Group.Diff1 = false := by decide
theorem obl_site_1 : framesComparable Group.Diff1 Group.Diff1 Group.DiffInf = true := by decide
theorem obl_site_2 : comparableDefault Group.Homeo Group.Homeo = true := by decide
theorem obl_site_3 : comparableDefault Group.BiLip Group.BiLip = false := by decide
theorem obl_chaos : reassocAllowed true true true = false := by decide
```

### Verification

- The demo emits exactly the four distinct Invariant sites (cross-frame Diff1 reject, same-frame Diff1 accept, Homeo accept, BiLip reject) plus the chaos obligation. Each emitted statement is independently confirmed **true**.
- A program with no Equivalence Theory features records **0** obligations.
- **NO MATHLIB:** `grep -E "import Mathlib|Mathlib\."` returns **0** (the only "Mathlib" string is the "Mathlib-free" comment).

### Honest accounting — "no claims X, delivers Y"

- **Behaviour-neutral when not emitting.** The checker hooks only record into the side-table and are gated by `EQUIV_RECORDING`, which is `0` during normal `check`/`build`; `equiv_record_invariant`/`equiv_set_chaotic` return immediately. The reject diagnostics are unchanged (the demo still reports E004 for the rejected sites during emission — the recording happens regardless of the report). A baseline-diff sweep (65 non-feature files: 40 run-pass + 25 compile-fail) found **0 differences** vs the checked-in baseline, and all six Equivalence Theory tests behave correctly on the new build (the three compile-fail tests reject with exit 1, the three run-pass tests accept with exit 0) — confirming the change is behaviour-neutral for existing code while preserving correct feature behaviour.
- **Per-module.** Recording fires for the compiled module's call-sites; imported modules' sites are captured when those modules are the emit target. The dedup is on `(groups, frame, outcome)`, so distinct configurations are kept and repeats collapsed.
- **Proof-checking is by the project's Lean CI** (no `lean`/`lake` toolchain here); the obligation statements are confirmed true.

### Files

- `self-hosted/check/effects.sio`, `self-hosted/check/check.sio`, `self-hosted/compiler/main.sio`, `examples/equivalence_theory/lean_obligation_demo.sio`.

---

## PT — Resumo

O primeiro auto-emissor varria a fonte lexicalmente e emitia uma obrigação por *recurso*. Isto refina para **por local de chamada**: as obrigações são agora registradas pelo **verificador** — os grupos concretos + relação de referencial + resultado de cada comparação `Invariant`, e cada função `with Chaotic` — então o conjunto emitido reflete exatamente o que o programa faz.

### Mecanismo

- **Tabela de registro** em `check/effects.sio` (globais de módulo + acessores `pub`): `equiv_begin_recording`/`equiv_end_recording`, `equiv_record_invariant` (deduplicado), `equiv_set_chaotic`, `equiv_emit_body`. Gatilho por `EQUIV_RECORDING`, ativo só durante `emit-lean-obligations`.
- **Ganchos do verificador** em `check.sio`: a regra de operador binário registra cada comparação; `checker_check_fn_item_inplace` registra função `with Chaotic` (efeito id 22).
- **Modo de emissão** em `main.sio`: roda o verificador com registro ativo, depois escreve `equiv_emit_body`.

### Verificação

O demo emite exatamente os quatro locais Invariant distintos + obrigação de caos; cada afirmação é confirmada **verdadeira** independentemente. Programa sem recursos registra **0**. **SEM MATHLIB:** `grep -E "import Mathlib|Mathlib\."` retorna **0**.

### Prestação de contas honesta

- **Neutro em comportamento quando não emite.** Os ganchos só registram na tabela, com gatilho `EQUIV_RECORDING` (`0` em `check`/`build` normais). Os diagnósticos de rejeição não mudam. Uma varredura baseline-diff (65 arquivos sem o recurso) encontrou **0 diferenças** vs o baseline, e os seis testes da Teoria da Equivalência comportam-se corretamente (os três compile-fail rejeitam com exit 1; os três run-pass aceitam com exit 0).
- **Por módulo.** O registro dispara para os locais do módulo compilado; dedup por `(grupos, referencial, resultado)`.
- **A verificação de prova é pela CI Lean do projeto** (sem toolchain local).

### Arquivos

- `self-hosted/check/effects.sio`, `self-hosted/check/check.sio`, `self-hosted/compiler/main.sio`, `examples/equivalence_theory/lean_obligation_demo.sio`.

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; the recording, hooks, emit mode, and this bilingual note were AI-drafted and human-reviewed. The per-site emission is backed by a re-runnable end-to-end demo (compiler built from source); emitted obligations are confirmed true and Mathlib-free. / Desenvolvido com assistência de IA sob direção humana; a emissão por local tem respaldo em demo reexecutável; as obrigações são confirmadas verdadeiras e livres de Mathlib.
