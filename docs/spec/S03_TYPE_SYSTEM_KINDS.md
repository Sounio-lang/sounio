<!-- docs:meta
topic_id: repo.docs.spec.s03-type-system-kinds
authority: repo_only
audience: users
last_validated: 2026-08-20
validated_by: codex-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.s03-type-system-kinds
-->

---
title: S03 — Type system: kinds and theorem vocabulary
status: measured
date: 2026-08-20
last_validated: 2026-08-20
engines: Madaros v0.80.0 (default), lean_single
---

# S03 — Type system: kinds and theorem vocabulary

## 3.1 Normative

*Awaiting founder rulings — see §3.4. Nothing in this section is normative yet.*

## 3.2 What is measured today

Measurements in this section use tree SHA `67aa2aec12`. Compiler probes use the
checked-in Madaros v0.80.0 and `lean_single`; they do not claim a source-built
fixed point for Madaros.

### 3.2.1 `stdlib/theorem` is large, but much of its vocabulary is generated or copied

The versioned directory is **67,904 lines in 36 files**, with **76 type
declarations** and **3,281 function declarations**. The earlier count of 37
files is not reproduced by `git ls-files` or `find` at the measured SHA; the
line and declaration counts are reproduced exactly.

Its dominant function-name families are not propositions or lemmas:

| name evidence | count |
|---|---:|
| `solver_*` functions | 1,627 |
| `portfolio_*` functions | 367 |
| functions ending `_fingerprint` | 1,478 |
| functions ending `_valid` | 183 |
| functions ending `_mask` | 162 |
| functions ending `_proof` | 41 |
| functions ending `_obligation` | 40 |
| functions ending `_witness` | 15 |

The vocabulary therefore lives in three different forms:

- **ordinary structs and arenas:** `Prop`, `Proof`, `ProofStore`,
  `ProofContext`, `Goal`, `Decidable`, `EpistemicProof`, `DSProof`,
  `DecayProof`, and `FusionProof`;
- **ordinary functions:** `assume`, `trusted`, `verify`, natural-deduction
  introduction/elimination functions, SAT/SMT/LRAT/RUP checkers, and numerical
  certificate builders;
- **identifier suffixes:** especially `_fingerprint`, `_valid`, `_mask`,
  `_proof`, `_obligation`, `_artifact`, `_preflight`, and `_witness`.

The apparent kernel is duplicated. The first 900 lines of `logic.sio`,
`nat.sio`, `real.sio`, `tactics.sio`, `units.sio`, and `epistemic.sio` have the
same SHA-256. Consequently the directory declares `Proof` eight times and
`ProofContext` eight times. These are file-local ordinary structs, not one
shared nominal theorem type.

### 3.2.2 The library proof types are disconnected from the compiler proof kinds

The parser and checker name three generic kinds: `Axiom<T>`, `Lemma<T>`, and
`Proof<T>`. `stdlib/theorem` uses those spellings **zero times**; all of
`stdlib/` also uses them zero times. The non-archived, non-bootstrap versioned
corpus names them in **11 files** (14 if `archive/` and `bootstrap/` are
included). The library's `struct Proof` is therefore not an inhabitant of the
compiler's `Proof<T>`.

Madaros does distinguish the three kinds on the by-value type path: assigning
an `Axiom<i64>` to a `Proof<i64>` parameter produces `E009`. It also refuses a
raw `f64` assigned to each kind with `E001`. That is only negative evidence.
No source-level constructor or positive inhabitant was found, and the three
fixtures marked `run-pass` (`annotation_axiom_basic.sio`,
`annotation_lemma_basic.sio`, and `annotation_proof_basic.sio`) all fail today
with `E001`.

The parameter-lowering path is weaker. `checker_lower_type_expr_mut` does not
have arms for these kinds; they enter its silent `_ =>` branch, increment an
internal type-error count, and return `ty_error()`. A live caller passing an
`i64` to each of `Proof<i64>`, `Lemma<i64>`, and `Axiom<i64>` produces the same
`E009` as the invented control `Zorblex<i64>` on Madaros. On `lean_single`, all
four live-caller probes pass. This is the twenty-kind silent-spine class tracked
by `silent_type_spine_ratchet_gate.sh`, not proof semantics.

### 3.2.3 The central library types do not yet have a compiler-enforced refusal witness

A refusal probe must construct, pass, or consume the value. An unused type
annotation was not counted. The attempted witnesses used confidence outside
`[0,1]`, negative proof and proposition identifiers, contradictory `Decidable`
branches, negative goal sizes, negative decay rates, and a fusion result with
no sources.

| type | declarations / references in `stdlib/theorem` | invalid witness result |
|---|---:|---|
| `Proof` | 8 / 302 | The public `theorem::search::Proof` accepts `confidence = 2.0`, negative identifiers, emits an ELF under `lean_single`, and exits 0. Madaros reaches existing `E038` errors in `theorem::search` before the witness. |
| `ProofContext` | 8 / 459 | Counts and arena relationships are ordinary fields. Capacity is checked by selected functions at runtime; no type-level refusal was found. The owning files fail before an invalid direct-literal witness is reached. |
| `EpistemicProof` | 1 / 25 | `epistemic_valid` checks confidence and uncertainty only when called. The struct is private; its owning file fails on both engines before the injected invalid value is reached (`E035` first on Madaros). |
| `DSProof` | 1 / 3 | No validator was found. Beliefs and conflict are `f64` fields; the owning file fails before the injected out-of-range witness is reached. |
| `DecayProof` | 1 / 3 | No validator was found. Negative rate/time and out-of-range confidence are representable fields; the owning file fails before the witness is reached. |
| `FusionProof` | 1 / 3 | No validator was found. Negative source count and uncertainty are representable fields; the owning file fails before the witness is reached. |
| `Goal` | 1 / 8 | `goal_assume` bounds additions at runtime, but the type has no invariant. `tactics.sio` fails first with `E035` on both engines. |
| `Decidable` | 1 / 22 | The constructors choose one branch by convention; no validator excludes two invalid branches or invalid proof IDs. `logic.sio` fails first with `E035` on both engines. |

This table does **not** turn baseline failures into evidence that malformed
values are accepted. It records a sharper result: except for the public
`search::Proof` control, the current compiler cannot reach the proposed
semantic refusal because the owning module already fails. Those rows remain
unmeasured at the semantic boundary.

### 3.2.4 The module is imported heavily, but chiefly as generated solver data

There are **239 import statements in 234 distinct files**, not 235 distinct
importers at this SHA. Of those files, 204 are tests, 19 are examples, and 11
are other stdlib modules. Only 30 are outside `tests/`.

| imported submodule | import statements |
|---|---:|
| `portfolio` | 174 |
| `smt` | 27 |
| `div_witness` | 13 |
| `lrat` | 4 |
| `pb` | 3 |
| all other submodules combined | 18 |

Representative live importers of `portfolio`, `smt`, `div_witness`, and
`lrat` check successfully on Madaros. By contrast,
`tests/stdlib/theorem/test_theorem_kernel.sio` fails with `E175`. Imports prove
that selected public surfaces are live; they do not prove that the copied
kernel, epistemic proof records, or compiler kinds are connected to those
surfaces.

## 3.3 What this does not claim

This section measures names, reachability, checking, and refusal. It does not
assess the mathematical correctness of SAT, SMT, LRAT, Farkas, Dempster–Shafer,
GUM, decay, or fusion algorithms. It does not infer that a private type accepts
an invalid value when its owning module fails before the witness. It does not
infer that high importer count makes the module a sound proof kernel, nor that
copied code is wrong merely because it is copied.

## 3.4 Rulings owed

1. **Which proof vocabulary is the language surface?** The founder must decide
   whether `Axiom<T>`/`Lemma<T>`/`Proof<T>`, the library's ordinary structs, or
   an explicit bridge is authoritative.
2. **Are proof kinds constructible or Reserved?** They currently have negative
   distinctions but no positive source inhabitant. If deliberately Reserved,
   refusal needs a named diagnostic; if executable, each needs a constructor
   and a positive witness.
3. **Is there one theorem kernel or several file-local copies?** Six identical
   900-line prefixes and eight separate `Proof` declarations require an
   ownership ruling before consolidation or divergence can be judged.
4. **Which invariants belong to types?** Confidence ranges, valid identifiers,
   mutually exclusive decidability branches, non-negative source counts, and
   proof-context consistency are conventions or optional runtime checks today.
5. **What does an import certify?** Most importers consume generated portfolio
   and solver surfaces. A ruling is owed on whether those are theorem evidence,
   checked certificates, or ordinary data APIs.
6. **Which engine defines refusal?** Madaros distinguishes some proof kinds but
   cannot check the central modules cleanly; `lean_single` accepts the invalid
   public `Proof` witness and all four parameter probes. A conformance decision
   cannot be derived from divergence alone.

## Claims Forbidden

What this section does not license anyone to say:

- Not that `stdlib/theorem` is a verified theorem prover. No such end-to-end
  witness was run, and its central owning files do not check cleanly today.
- Not that `Axiom<T>`, `Lemma<T>`, or `Proof<T>` are implemented merely because
  the parser and checker name them. No positive inhabitant was found.
- Not that the library `struct Proof` implements the language `Proof<T>`. The
  corpus contains no bridge between them.
- Not that 234 importers validate the proof semantics. Most import generated
  portfolio surfaces, and 204 are tests.
- Not that the private central types accept malformed values. Baseline failures
  prevented those refusal witnesses from reaching the semantic boundary.
- Not that any mathematical claim in the module is false. This survey did not
  review the mathematics.
