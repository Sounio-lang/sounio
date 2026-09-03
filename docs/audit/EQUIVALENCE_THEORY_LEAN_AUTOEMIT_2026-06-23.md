<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-lean-autoemit-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-lean-autoemit-2026-06-23
-->

# Equivalence Theory — compiler auto-emission of Lean obligations
## Teoria da Equivalência — auto-emissão de obrigações Lean pelo compilador

*Date / Data:* 2026-06-23 · *Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis
*Follows:* `EQUIVALENCE_THEORY_LEAN_OBLIGATIONS_2026-06-23.md` (the discharged proofs).

---

## EN — Summary

The previous step discharged the decidable Equivalence Theory obligations in `formal/EquivalenceTheory.lean`, but those were hand-mirrored — the compiler did not emit them. This step closes the §7 "emitted as an obligation" half: **the compiler now auto-generates the obligation `.lean` from a program.**

### What was added

A new compiler mode in `self-hosted/compiler/main.sio`:

```
madaros emit-lean-obligations <src.sio> <out.lean>
```

It reads the program, scans it for Equivalence Theory feature usage, and writes the corresponding **Mathlib-free, decidable** Lean 4 obligations — each `by decide`, each referencing the discharged definitions in `formal/EquivalenceTheory.lean`. Example, on `examples/equivalence_theory/lean_obligation_demo.sio` (which uses `Invariant<f64, Diff1>`, `Invariant<f64, Homeo>`, and `with Chaotic`):

```lean
-- Auto-emitted Equivalence Theory obligations (Mathlib-free).
-- Source: examples/equivalence_theory/lean_obligation_demo.sio
-- Decidable obligations; discharged in formal/EquivalenceTheory.lean.
import EquivalenceTheory
open Sounio.EquivalenceTheory

theorem obl_diff1 : comparableDefault Group.Diff1 Group.Diff1 = false := by decide
theorem obl_homeo : comparableDefault Group.Homeo Group.Homeo = true := by decide
theorem obl_chaos : reassocAllowed true true true = false := by decide
```

### Verification

- **End to end** (compiler built from this source): the demo emits exactly the `Diff1`, `Homeo`, and chaos obligations; `DiffInf`/`BiLip` (not used) are correctly absent.
- **Discrimination:** a program with no Equivalence Theory features emits the header and **zero** obligations.
- **The emitted obligations are true:** independently re-evaluated (`comparableDefault Diff1 Diff1 = false`, `Homeo Homeo = true`, `reassocAllowed true true true = false`) — they match the discharged general lemmas.
- **NO MATHLIB:** the emitted file imports only `EquivalenceTheory` and uses only `by decide`; a grep for `Mathlib` returns **0**. (`formal/EquivalenceTheory.lean` itself has zero imports.)

### Honest accounting — "no claims X, delivers Y"

- **The scan is lexical** (whole-token presence of the group names / `Invariant` / `Chaotic` in the source), not a full checker/AST traversal. So the emission is **per-feature** (one obligation per group/feature used), not **per-call-site**. A precise per-site emitter — recording each `Invariant` comparison's concrete groups during checking — is the natural next refinement; the lexical version is documented as such and is sufficient to certify that a program's features rest on discharged theorems.
- **Mathlib-free by design.** The emitted obligations are decidable so they need no Mathlib; this is also why Feature C-a's geodesic identities (which would require Mathlib's `Real.arccosh`) are kept **numerical** (`tests/run-pass/hyperbolic_geodesic.sio`) rather than emitted as Lean — formalising them would breach the no-Mathlib policy.
- **Proof-checking is by the project's Lean CI** (no `lean`/`lake` toolchain in this environment); the obligation statements are confirmed true here.

### Files

- `self-hosted/compiler/main.sio` (new `emit-lean-obligations` mode), `examples/equivalence_theory/lean_obligation_demo.sio` (fixture).

---

## PT — Resumo

O passo anterior descarregou as obrigações decidíveis em `formal/EquivalenceTheory.lean`, mas espelhadas à mão — o compilador não as emitia. Este passo fecha a metade "emitido como obrigação" do §7: **o compilador agora gera o `.lean` de obrigações a partir de um programa.**

### O que foi adicionado

Novo modo em `self-hosted/compiler/main.sio`:

```
madaros emit-lean-obligations <src.sio> <out.lean>
```

Lê o programa, varre o uso de recursos da Teoria da Equivalência e escreve as obrigações Lean 4 correspondentes — **sem Mathlib, decidíveis** (cada `by decide`), referenciando as definições descarregadas em `formal/EquivalenceTheory.lean`.

### Verificação

Ponta a ponta: o demo emite exatamente as obrigações de `Diff1`, `Homeo` e caos; `DiffInf`/`BiLip` (não usados) ausentes. Discriminação: programa sem recursos emite cabeçalho e **zero** obrigações. As obrigações emitidas são verdadeiras (reavaliadas independentemente). **SEM MATHLIB:** o arquivo importa só `EquivalenceTheory` e usa só `by decide`; grep por `Mathlib` retorna **0**.

### Prestação de contas honesta

- **A varredura é lexical** (presença de token inteiro dos nomes de grupo / `Invariant` / `Chaotic`), não percurso de AST. A emissão é **por recurso**, não **por local de chamada**. Um emissor preciso por local — registrando os grupos concretos de cada comparação `Invariant` durante a checagem — é o refinamento natural seguinte.
- **Sem Mathlib por desenho.** As obrigações são decidíveis; é também por isso que as identidades geodésicas do Recurso C-a (que exigiriam `Real.arccosh` do Mathlib) ficam **numéricas**, não emitidas em Lean — formalizá-las violaria a política sem-Mathlib.
- **A verificação de prova é pela CI Lean do projeto** (sem toolchain local); as afirmações das obrigações são confirmadas verdadeiras aqui.

### Arquivos

- `self-hosted/compiler/main.sio` (modo `emit-lean-obligations`), `examples/equivalence_theory/lean_obligation_demo.sio` (fixture).

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; the emitter, fixture, and this bilingual note were AI-drafted and human-reviewed. The auto-emission is backed by a re-runnable end-to-end demo (compiler built from source) and the emitted obligations are confirmed true; they are Mathlib-free. / Desenvolvido com assistência de IA sob direção humana; a auto-emissão tem respaldo em demo reexecutável ponta a ponta e as obrigações emitidas são confirmadas verdadeiras; são livres de Mathlib.
