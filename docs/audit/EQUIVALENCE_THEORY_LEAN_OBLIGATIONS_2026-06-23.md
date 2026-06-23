<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-lean-obligations-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-lean-obligations-2026-06-23
-->

# Equivalence Theory — resolving the Lean gap: discharged obligations for the decidable cores
## Teoria da Equivalência — resolvendo a lacuna Lean: obrigações descarregadas dos núcleos decidíveis

*Date / Data:* 2026-06-23 · *Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis

---

## EN — Summary

The prompt's §7 asks that every "rewrite sound in theory X", "transform ∈ group class C", and geodesic identity be **emitted as a Lean obligation and discharged**. The Lean surface was a gap: `self-hosted/compiler/epistemic_lean_export.sio` is orphaned (wired into no driver) and `formal/*.lean` proves the *Rust* compiler, not these features. This commit discharges the **decidable, load-bearing** obligations in machine-checkable Lean 4.

### What was discharged — `formal/EquivalenceTheory.lean` (Mathlib-free, added to the `formal` lakefile roots)

**Feature B — invariance-group lattice** (mirrors `stdlib/check/invariance.sio`):
- `groupRank_determines` — the four groups are pairwise rank-distinct (the lattice is a strict total order).
- `groupIntersect_isMin`, `groupIntersect_comm_rank` — `group_intersect` is exactly the rank-minimum (the algebraic content of "intersection").
- `framesComparable_iff` — the comparability rule holds **iff** the relating transform's rank lies within `ga ∩ gb` (the rule is sound and complete w.r.t. its specification).
- `sameFrame_comparable` — same frame ⇒ comparable at **any** group (identity transform); the theorem behind the FrameId follow-up that makes Diff1/BiLip invariants usable.
- `lyapunov_crossHomeo_not_comparable`, `entropy_crossHomeo_comparable`, `comparableDefault_onlyHomeo` — the prompt's §4.4 discrimination, proved: under the conservative default only Homeo-group observables are comparable.

**Features A-1 / A-layer-2 — e-graph float-reassociation gate** (mirrors `eg_small_saturate_float`):
- `chaos_dominates` — a chaotic path refuses reassociation regardless of the other axes (the theorem behind unit test **T143d**).
- `inexact_default_blocks` — an inexact (f64) carrier that has not opted in is never reassociated (A-1 default).
- `valueAlgebra_necessary` — the Cayley-Dickson/Fano axis is independently necessary.
- `permit_when_all_ok`, `reassoc_composition` — reassociation fires **iff** `(value-algebra permits) ∧ (inexact opt-in) ∧ ¬chaotic` — so the gates remove only unsound rewrites, not all of them.

All proofs are by exhaustive `cases`/`decide` over the four-element `Group` enum and the Boolean axes — finite, decidable, needing no Mathlib (the most robust kind of Lean proof).

### Verification status — read honestly

- **The obligations target true statements.** Every theorem was independently re-evaluated (an out-of-Lean check of the same definitions): all 16 hold. The corresponding Sounio behaviour is also already verified empirically — the §4.4 discrimination via `madaros --check`, the chaos gate via unit test T143d.
- **Proof-checking is by the project's Lean CI, not this environment.** No `lean`/`lake` toolchain is available here, so the proofs are not machine-checked locally. They are written in conservative, decidable style (`cases <;> decide`) and added to `formal/lakefile.lean`'s roots so the existing Lean build checks them. This is stated plainly rather than claimed as locally verified.

### Honest scope — "no claims X, delivers Y"

- **Discharged: the decidable, load-bearing type-system obligations** (B's lattice/comparability soundness; the A gates' composition). These are the "theorems in theory X" the checker relies on.
- **The obligations are hand-mirrored from the Sounio definitions** (each Lean theorem names its Sounio source). **Automatic emission from the compiler** — wiring `epistemic_lean_export.sio` so a `.sio` compile generates these obligation statements — remains future work.
- **Feature C-a's geodesic identities** (`d = arccosh(-⟨x,y⟩_M)`, exp/log round-trip) are real-analysis statements needing Mathlib's `Real`; they are verified **numerically** in `tests/run-pass/hyperbolic_geodesic.sio` and their Lean formalisation is deliberately deferred.
- **Per-frame-pair "transform ∈ class" obligations** (a populated `FrameRegistry`) are per-instance facts discharged when a developer registers a concrete frame relation; the **rule that consumes them is now proved** (`framesComparable_iff`).

### Files

- `formal/EquivalenceTheory.lean` (new), `formal/lakefile.lean` (added to roots).

---

## PT — Resumo

O §7 do enunciado pede que todo "reescrita sólida na teoria X", "transformação ∈ classe de grupo C" e identidade geodésica seja **emitido como obrigação Lean e descarregado**. A superfície Lean era uma lacuna: `epistemic_lean_export.sio` está órfão e `formal/*.lean` prova o compilador *Rust*, não estes recursos. Este commit descarrega as obrigações **decidíveis e centrais** em Lean 4 verificável por máquina.

### O que foi descarregado — `formal/EquivalenceTheory.lean` (sem Mathlib, adicionado às raízes do lakefile)

**Recurso B — reticulado de grupos de invariância** (espelha `stdlib/check/invariance.sio`): `groupRank_determines` (ordem total estrita), `groupIntersect_isMin`/`groupIntersect_comm_rank` (interseção = mínimo de posto), `framesComparable_iff` (a regra de comparabilidade é sólida e completa quanto à sua especificação), `sameFrame_comparable` (mesmo referencial ⇒ comparável em qualquer grupo), e a discriminação do §4.4 provada (`lyapunov_crossHomeo_not_comparable`, `entropy_crossHomeo_comparable`, `comparableDefault_onlyHomeo`).

**Recursos A-1 / A-camada-2 — porta de reassociação f64** (espelha `eg_small_saturate_float`): `chaos_dominates` (teorema por trás de T143d), `inexact_default_blocks` (padrão A-1), `valueAlgebra_necessary`, `permit_when_all_ok`, `reassoc_composition` (reassocia sse `(álgebra permite) ∧ (adesão inexata) ∧ ¬caótico`).

Todas as provas por `cases`/`decide` exaustivo sobre o enum `Group` de 4 elementos e os eixos booleanos — finitas, decidíveis, sem Mathlib.

### Estado de verificação — honesto

- **As obrigações miram afirmações verdadeiras.** Cada teorema foi reavaliado independentemente (fora do Lean): todos os 16 valem. O comportamento Sounio correspondente também já está verificado empiricamente (§4.4 via `madaros --check`; porta do caos via T143d).
- **A verificação de prova é pela CI Lean do projeto, não por este ambiente.** Não há toolchain `lean`/`lake` aqui; as provas não são verificadas por máquina localmente. Escritas em estilo decidível conservador (`cases <;> decide`) e adicionadas às raízes do lakefile. Isto é declarado claramente, não alegado como verificado localmente.

### Escopo honesto — "não prometer X e entregar Y"

- **Descarregado: as obrigações decidíveis e centrais do sistema de tipos.** **Emissão automática pelo compilador** (conectar `epistemic_lean_export.sio`) segue como trabalho futuro. **As identidades geodésicas do Recurso C-a** precisam de Mathlib (`Real`); verificadas **numericamente** em `tests/run-pass/hyperbolic_geodesic.sio`, formalização Lean adiada. **Obrigações "transformação ∈ classe" por par de referenciais** (um `FrameRegistry` populado) são fatos por instância; a **regra que as consome está agora provada** (`framesComparable_iff`).

### Arquivos

- `formal/EquivalenceTheory.lean` (novo), `formal/lakefile.lean` (adicionado às raízes).

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; the Lean proofs and this bilingual note were AI-drafted and human-reviewed. The theorem statements are independently confirmed true; machine proof-checking is by the project's Lean CI (no local toolchain in this environment). / Desenvolvido com assistência de IA sob direção humana; as afirmações dos teoremas são confirmadas verdadeiras independentemente; a verificação de prova por máquina é pela CI Lean do projeto (sem toolchain local neste ambiente).
