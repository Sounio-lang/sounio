<!-- docs:meta
topic_id: repo.docs.spec.s08-epistemic-values
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.s08-epistemic-values
-->

# §8 — Epistemic Values

Spec-Section: `SOUNIO-SPEC-08`
Frame: `docs/spec/E2E_SPECIFICATION_FRAME.md`

Status: **Hypothesis.** The normative statement below is a founder ruling of
2026-08-19. No conformance test exists yet; the section reaches `Executable`
only when one runs on **both** engines with a negative control
(`SOUNIO-GATING-ENGINE`, `SOUNIO-NO-VERSUS-UNKNOWN`).

## 8.1 Normative

> **`Knowledge<T>` is a type with invariants, not a record with three fields.**

Founder ruling. The distinction is the whole section: a record exposes fields,
and a type exposes **operations that preserve its invariants**. What is written
`k.value` is therefore an *operation on an epistemic value*, not a field access
that happens to be spelled with a dot.

Everything else in this section follows from that sentence rather than being
stipulated beside it.

## 8.2 What is measured today

`origin/main`, 2026-08-19. **There is no single epistemic value.** The name
denotes three structurally different objects, and they disagree about what an
epistemic value even carries.

| # | Definition site | Members | Provenance | Confidence |
|---|---|---|---|---|
| 1 | Compiler, `self-hosted/parser/ast.sio:492` (`KnowledgeTypeInfo`) | `inner_type`, `epsilon`, `validity`, `provenance`, `proof_constraints` | **member** | **absent** |
| 2 | `stdlib/epistemic/knowledge.sio:63` (`Epistemic`) | `val: f64`, `variance: f64`, `confidence: i64` | absent | 0–1000 |
| 3 | `examples/` ×4 (`Epistemic`) | `value: f64`, `variance: f64`, `label: i64` | **as a tag** | absent |

Shape 3 is not a stray: it is what `examples/science/darwin_epistemic_pbpk.sio`
uses — the dissertation surface. The four occurrences are independent
re-declarations of the same name in separate files, not one imported type.

Three consequences follow directly, and each is a defect rather than an
observation:

- **The same name, the same field position, opposite meanings.** In shape 3
  `label: 0` annotates `measured` — the *strongest* provenance. In shape 2
  `confidence: 0` denotes *no confidence* — the weakest possible claim. A value
  moving between the two shapes by position, or a reader carrying a habit from
  one file to another, inverts the epistemic claim without any diagnostic.
- **`Knowledge<T>` as the compiler knows it has neither `variance` nor
  `confidence`.** It carries `epsilon`, `validity`, `provenance` and
  `proof_constraints`. The 0–1000 scale — including the endpoint ruled on in
  8.3.5 — exists only in library and example code, and the compiler's own
  epistemic type has no field for it to occupy.
- **Provenance is a member in two of the three and absent from the third**, and
  the third is the one carrying confidence. Whatever 8.5 decides about where
  provenance lives, the repository currently answers it two ways at once.

### 8.2.1 The three are not three representations. They are two layers and a copy.

Reading the compiler type's members closes the question the table appears to
open. Every member of `KnowledgeTypeInfo` is `Option<...>`, and none is a
number:

| Member | What it is |
|---|---|
| `epsilon: Option<EpsilonBound>` | a **predicate**: `EpsilonBound { op: CompareOp, value: f64 }`, `op` ∈ {`<`, `≤`, `=`, `≥`, `>`} |
| `validity: Option<ValidityCondition>` | a predicate with an optional expression |
| `provenance: Option<AstProvenanceKind>` | a six-way enum: `Derived`, `Source`, `Computed`, `Literature`, `Measured`, `Input` |
| `proof_constraints` | a list of constraints |

So the compiler's `Knowledge<T>` is a **type-level claim** — predicates and
provenance, discharged at compile time. The stdlib's `Epistemic` is a
**value-level representation** — the numbers that flow at run time. They are not
competing encodings of the same thing; they sit at different levels, and the
relation between them is that `epsilon` is a **predicate the runtime variance
must satisfy**. Bridging the two needs a coverage convention, which is what GUM
already standardises and what `gum_k95` already computes.

Two things follow, and they point in opposite directions.

**In the compiler's favour.** Every member being `Option<...>` is
`SOUNIO-NO-VERSUS-UNKNOWN` done correctly: *no epsilon was declared* is `None`,
structurally distinct from every declared bound. The checker reinforces it with
a sentinel outside the domain — `epistemic_epsilon: 0.0 - 1.0` (that is, `-1.0`)
at six sites in `self-hosted/check/check.sio`. A negative epsilon is a value no
correct bound can take, which is exactly the sentinel the concept sanctions. The
compiler already solves for `epsilon` the problem the stdlib leaves open for
`confidence`.

**Against the examples.** Shape 3's `label: i64` is documented in-file as
`0=measured, 1=asserted, 2=constant` — **three** tags. `AstProvenanceKind` has
**six**, and two of the three names (`asserted`, `constant`) do not appear in it
at all. Shape 3 is therefore not a simplification of the compiler's provenance;
it is an **independent, smaller, differently-named provenance vocabulary**, and
it is the vocabulary the dissertation surface uses. A value labelled `constant`
in `darwin_epistemic_pbpk.sio` has no image in the provenance the compiler can
reason about.

### 8.2.2 The cost of withdrawing the `label` vocabulary

Measured on `origin/main`, 2026-08-19, because the withdrawal proposed in 8.5(c)
touches the dissertation surface and its cost must precede its ruling.

**Footprint.** Four files, thirteen constructions, **two reads**. The four files
each carry an identical copy-pasted three-constructor preamble (`label: 0`,
`label: 1`, `label: 2`); twelve of the thirteen constructions are inside those
helpers. The only reads are `examples/epistemic_quantum_vqe.sio:247-248`, two
equality comparisons. **`darwin_epistemic_pbpk.sio` — the dissertation surface —
never reads `.label` at all.** There, the tag is write-only metadata.

**The tag conflates two axes, and one of them is already recorded elsewhere.**
Every one of the four `label: 2` sites is character-for-character the same
literal:

    Epistemic { value: value, variance: 0.0, label: 2 }

`2 = constant` is not a provenance. It is a claim about the variance, and the
claim is already made in the adjacent field of the same literal. Only `0 =
measured` and `1 = asserted` carry provenance.

**Therefore the withdrawal is not lossy — it removes a redundancy.**

| tag | carries | image under withdrawal |
|---|---|---|
| `0` measured | provenance | `AstProvMeasured` — exact |
| `1` asserted | provenance | `AstProvInput` or `AstProvSource` — **owed**, the only real decision |
| `2` constant | *variance*, not provenance | already expressed by `variance: 0.0`; no provenance image needed |

The residue is one mapping decision — where an *asserted* value sits among
`Input` and `Source` — plus two equality comparisons in a file that is not the
dissertation. `label: 2` needs no image at all, because it never carried
provenance to lose.

### 8.2.3 Three of the six provenance kinds cannot be written

`AstProvenanceKind` declares six kinds. `self-hosted/parser/types.sio:1107-1117`
matches exactly three token kinds — `TokenKind::Derived`, `TokenKind::Computed`,
`TokenKind::Measured` — and constructs the corresponding three. `AstProvSource`,
`AstProvLiterature` and `AstProvInput` occur **once each** in the whole of
`self-hosted/parser/`: their own declaration. They are declared and unwritable.

This is not `Reserved` in the sense of `MATURITY_LADDER`. A reserved name is
refused with a named diagnostic. These three are not refused — there is simply
no syntax that reaches them, and nothing says so.

**The parse loop's fallthrough makes it silent.** The same block ends:

    } else {
        // Unknown component — skip
        p = p.advance()
    }

An unrecognised component inside a `Knowledge<...>` annotation is **discarded
without a diagnostic**. Writing a provenance the compiler declares but cannot
parse yields a value with *no* provenance and no error — which is
`SOUNIO-NO-VERSUS-UNKNOWN` at the point where provenance is claimed. Silent,
reachable, and ungated: the `SOUNIO-S-G-R` criterion is met in full and a gate is
required regardless of any ruling in this section.

### 8.2.4 Correction: there are five declarations, not three, and the dissertation matches none

Re-measured 2026-08-19, later the same day. The table in 8.2 lists three shapes
plus a two-line fixture. **It is incomplete.**

| # | site | members |
|---|---|---|
| 4 | `ecosystem/shared/epistemic_types.sio:9` `struct Knowledge[T]` | `value: T`, **`ε: f64`**, `prov: string`, `metadata: KnowledgeMetadata` |
| 5 | `self-hosted/test_knowledge.sio` | a second `struct Knowledge[T]` |

Shape 4 matters for two reasons.

**It is the richest one, and `ε` means something else in it.** Its own comment
reads `ε: f64  // Confidence (0.0 = no confidence, 1.0 = certain)`. So `ε` here
is a **confidence in [0,1]**, while in the compiler `EpsilonBound` is a
**predicate on the error**. The same symbol carries two unrelated meanings across
two layers, and a third convention — the stdlib's integer 0–1000, whose endpoint
was ruled to denote certainty in 8.3.5 — makes three.

**The bracket form is widespread.** `Knowledge[...]` occurs in **49** versioned
`.sio` files, against the angle-bracket form the earlier revision assumed.

### 8.2.4-bis Five was a floor, not a count: eighteen shapes over forty-six sites

§8.2 lists five declarations. That list was assembled from the sites this section
already had reason to visit, and it is **materially incomplete** — recorded here
rather than silently widened, because the canonical-shape decision cannot be
taken against a list missing most of its candidates.

Census: versioned `.sio` outside `archive/`, `bootstrap/`, `docs/`; declarations
whose *name* is the epistemic value itself (`Knowledge`, `Knowledge<T>`,
`Knowledge[T]`, `KnowledgeF64`, `KnowledgeI64`, `Epistemic`, `Epistemic<T>`,
`EpistemicValue`, `EpistemicVal`). Domain structs that merely begin with
`Epistemic`/`Knowledge` — `EpistemicPtxConfig`, `KnowledgeARIMA`,
`EpistemicMCTSNode` and 127 others — are **excluded**; counting them gives 173
and is the same overcount this document has already made once.

**46 declaration sites, 18 distinct field shapes:**

| sites | members |
|---:|---|
| **14** | `value, uncertainty, confidence` |
| 9 | `value, variance, conf_alpha, conf_beta` |
| 4 | `value, variance, label` |
| 2 | `value, uncertainty, confidence, provenance` |
| 2 | `value, uncertainty` |
| 2 | `val, variance, confidence` |
| 1 | `value, variance, confidence, provenance` |
| 1 | `value, variance, confidence` |
| 1 | `value, variance, alpha, beta` |
| 1 | `value, variance` |
| 1 | `value, uncert, conf, source` |
| 1 | `value, uncert, conf, provenance_id` |
| 1 | `value, uncert, conf, provenance_count, debt_bits` |
| 1 | `value, prov, metadata` |
| 1 | `value, epsilon, reducible_by_n_samples` |
| 1 | `value, confidence, provenance` |
| 1 | `value, confidence, knightian` |
| 1 | `provenance` |

Two things fall out.

**The plurality shape is not in §8.2's list.** `value, uncertainty, confidence`
holds 14 of 46 sites — three times any other — and §8.2 names none of them. The
shape §8.2 calls the one the stdlib exposes, `val, variance, confidence`, holds
**2**.

**The disagreement is lexical before it is structural.** The same slot is spelled
`value`/`val`; the second is `uncertainty`/`variance`/`uncert`/`epsilon`; the
third is `confidence`/`conf`/`label`/`knightian`, or split into
`conf_alpha, conf_beta`. Several of these shapes are the same idea under
different names, and a canonicalisation decision has to say which spelling wins
before it can say which shape does.

`docs/spec/LANGUAGE_SPECIFICATION.md:402` documents a sixth-and-different one —
`struct Knowledge<T>` with `confidence: BetaConfidence` — and it is **not
fictional**: `examples/alphageozero_final.sio:83` declares exactly that, and
`BetaConfidence` is declared in three files. It appears in **zero** files under
`self-hosted/`, so the compiler has never heard of it. The same document's
`Knowledge::exact` (:422) has **one** call site tree-wide, and `ep_exact` — which
§8.4 already records as non-existent — remains at **zero**.

### 8.2.5 The dissertation surface writes a shape that is declared nowhere

`stdlib/darwin_pbpk/epistemic_pbpk28.sio:292` contains, verified **not** inside a
block comment:

    var kn: [Knowledge[f64]; 8] = [Knowledge(0.0, ε=1.0, prov="unused"); 8]
    kn[0] = Knowledge { value: m[0], variance: v[0], epsilon: c[0],
                        provenance: "Jiao2009_popPK_CV38|CHEBI:9168" }

The literal's members are `value, variance, epsilon, provenance`. **No declared
shape has that set** — shape 4 has `value, ε, prov, metadata`. And the file
imports only `darwin_pbpk::core::pbpk28_params` and `darwin_pbpk::tsit5_pbpk28`,
so it imports no `Knowledge` declaration at all. Four versioned files write this
literal.

The comment immediately above it states the arrangement plainly:

> *"The f64 arrays above are their numeric projection for the GUM kernel."*

So the numbers flow through plain `f64` arrays (`m`, `v`, `c`) and the
`Knowledge` block sits beside them. **Re-verified, 2026-08-19** (`docs/audit/`, `#2024`, both engines on Slurm):

- **The literal is ACCEPTED.** Both Madaros and lean_single `check` the file with
  `rc=0` and an `E200` count of **zero**. An earlier draft of this correction
  cited an `E200` refusal *from memory*; the memory was wrong, and it is recorded
  here rather than quietly dropped.
- **`Knowledge` here is the compiler builtin** (`TypeKind::TyKnowledge`), not any
  of the five declared structs. That is why it resolves without an import.
- **The block is the GUM carrier, not decoration.** Stripping the seven `kn[i] =`
  assignments zeros `sens[0]` and fails TEST 5; the unmodified file passes 9/9
  under lean_single. The numbers pass through it.

Six further facts fall out of the same measurement, and each is a defect:

- **The field written is not the field read.** The literal writes `epsilon:`;
  under Madaros `.epsilon` reads **0.0** while `.confidence` reads 0.65.
- **The cause is positional filling, not a swapped alias.** Madaros fills the
  builtin `Knowledge` literal **by position**, ignoring the written names.
  lean_single resolves **by name** and aliases `epsilon` to `confidence`. Both
  therefore land `c[i]` in the confidence slot *under the field order this file
  happens to use* — Madaros by arithmetic, lean_single by meaning.
- **Madaros does not validate constructor field names at all.** Negative control
  (`docs/audit/repro/epsilon/neg_epsilom.sio`): a literal writing the invented
  name `epsilom: 0.42` is accepted by Madaros with `check: OK`, rc=0, and **no
  diagnostic**; lean_single emits `warning: unknown field in Knowledge literal`
  and drops the write. An earlier draft of this section stated the opposite —
  that *Madaros refuses with `E012`* — and that is **false for the builtin**.
  `E012` is real, and it fires: on a **declared** struct, `P { a: 1.0, zz: 9.0 }`
  gives `error[E012] ... this type has no field named`, rc=1, and lean_single
  gives two errors. The claim was a true measurement quoted from the wrong path.
  It is corrected here rather than quietly dropped, because it credited the
  silent engine with the rigour of the noisy one.
- **The builtin is exempt from the checker the language already has.** Field-name
  validation exists, works, and is enforced on every declared struct. The one
  type that carries the dissertation's numbers is the one that does not get it.
  This is `SOUNIO-TYPE-INTERROGATION` failure type 3 in a new form: not a missing
  check, but a **privileged type routed around an existing one**.
- **The correctness is order-dependent and nothing records the dependence.**
  Under Madaros, moving `epsilon:` ahead of `variance:` — a reordering no
  diagnostic objects to — puts `c[i]` in the variance slot and zeroes
  `.confidence`. The dissertation's GUM numbers are right because of the
  sequence the literal was typed in, not because the names were honoured.
- **The gate that covers this file is unreachable.**
  `scripts/ci/dissertation_pbpk_suite_gate.sh` lists it, and no workflow reaches
  that gate.

Other measured facts about shape 2, which is the shape the stdlib exposes:

- **Not linear.** Dropping one requires nothing.
- `confidence` is an integer **0–1000**, clamped at construction
  (`ep_clamp_conf`, `stdlib/epistemic/knowledge.sio:52`).
- The scale carries meaning nothing records: `ep_certain` (**not** `ep_exact`,
  which does not exist) constructs `variance: 0.0, confidence: 1000`;
  `ep_measured` constructs `confidence: 900`. Neither 1000 nor 900 is derived
  anywhere — 900 is a chosen constant with no stated basis.
- `.value` occurs **2,278 times** across `stdlib/` and `examples/` — but against
  shape 3 and the compiler type, since shape 2 spells the member `val`.

So the implementation is not merely "a record where the ruling wants a type".
It is **three records that do not agree**, none of which is the ruled type. That
gap is the section's work.

> **Correction, 2026-08-19.** An earlier revision of this section stated the
> measured type as `struct Knowledge<T> { value: T, variance: f64, confidence:
> i64 }` and named its constructor `ep_exact`. Neither exists. The struct was a
> merge of shapes 2 and 3 that is declared nowhere, and the constructor is
> `ep_certain`. The error is recorded rather than silently overwritten because
> this section's own subject is what happens when a claim is separated from what
> justifies it.

## 8.2.6 RULING — the epistemic value is TWO things, and they are not the same thing

Founder ruling, 2026-08-20: **the refinement in the type and the record in the
value are two layers, and each gets its own name.**

### What forced the question

`self-hosted/check/types.sio`:

    pub fn ty_knowledge(inner: TypeEntry, epsilon: f64) -> TypeEntry

The compiler's `Knowledge` carries **an inner type and one `f64`**. No value, no
variance, no confidence, no provenance. It is a **refinement type** — `T` with a
bound — not a record. The corpus writes records: 26 of 45 declaration sites are
`value, uncertainty, confidence` (§8.2.4-ter).

Four separately-recorded defects are one defect under this reading:

| observed | explained by |
|---|---|
| the builtin literal fills **by position** (§8.2.5) | there are no fields to match — there is `inner` and `knowledge_epsilon` |
| `.epsilon` reads `0.0` while `.confidence` reads the number | the reader expects record fields on a thing that never had any |
| ε has opposite polarities in the two engines (#2030) | a bound on a refinement **can only** be an error: satisfying it means being **below** |
| the clinical surface writes `ε >= 0.82` meaning confidence | a **measurement written into a constraint slot** |

**Madaros is not wrong about ε.** It implements the only reading a
`knowledge_epsilon: f64` on a `TypeEntry` can support.

### The ruling

1. **ε is the type-level error bound.** Doubt, compile-time, lower is better.
   `Knowledge[T, ε < 0.05]` reads *"this value's error is under 0.05"*.
   `epsilon_subsumes(a, b) = a <= b` (`check/epistemic.sio:595`) is **correct** and
   stays.
2. **Belief is a value-layer property and takes its own name.** It is not ε, it
   is not a bound, and it does not belong in the type's constraint slot.
3. **Writing belief as an ε bound is a layer error**, not a polarity difference.
   The five engine-divergent `compile-fail` tests frozen by
   `scripts/ci/epsilon_engine_parity_gate.sh` are written in the wrong layer.

### What this costs, stated plainly

**The vancomycin ε guarantee does not currently hold in either layer.** The
clinical surface wrote it into the type, and the type does not speak that
language. §13 of `KNOWN_LIMITATIONS` records it as engine-dependent; under this
ruling it is layer-absent, which is worse and more honest.

### The first task is smaller than the ruling

The value layer already has a syntax. Five files write `with Epistemic(950)` and
`with Epistemic(400)`; `parser/types.sio:812` parses it:

    fn parse_effect_payload(self) -> Parser {

It returns **only the parser**. It counts parentheses and advances past
everything, so `950` is read, balanced and **discarded**. `Epistemic(950)` and
`Epistemic(400)` are **identical to the compiler**.

That is why `tests/compile-fail/dissertation_pbpk28_overclaim.sio` — which demands
95% confidence over a 65% prior — is one of the five silent passes. It was never
a polarity failure. The belief gate is syntax with no semantics.

**So the ruling's first implementation step is: make the effect payload carry its
number.** The syntax is already parsed; only the slot is missing.

## 8.3 Invariants entailed by the ruling

These follow from 8.1. Each is normative; none is implemented.

1. **An epistemic value is inseparable from its uncertainty.** `value` and
   `variance` are not independently meaningful members. Obtaining one without
   the other is an operation, and the operation is not silent.
2. **Projection is an operation with a rule.** `k.value` yields a value that
   carries the mark of having been separated (`SOUNIO-EPISTEMIC-ERASURE`).
   The mark is inferred and propagates; the programmer never writes it.
3. **Re-attachment requires an act.** Uncertainty is restored only by
   `attest(v, uncertainty:, because:)`, whose floor is a discharged proof
   obligation (`SOUNIO-JUSTIFICATION`). There is no coercion.
4. **Decisions read the invariant, not the number.** `Admissible<T>` requires
   support that has not been degraded without justification
   (`SOUNIO-ADMISSIBILITY`). Deciding is the fifth sink.
6. **`epsilon` is a predicate over the runtime variance, not an alternative to
   it.** Founder ruling, 2026-08-19. The compiler's `Knowledge<T>` states a
   bound; the value-level representation computes a variance; the bound is
   something the variance is **checked against**. The two levels therefore do
   not compete for the same role, and no unification of the two declarations is
   required by this section.

7. **The predicate is discharged by the GUM coverage convention.** Founder
   ruling, 2026-08-19. `k` is the bridge from a computed variance to the
   expanded uncertainty the bound compares against, and `gum_k95` already
   computes it. Which `k` a given `EpsilonBound` implies is a matter for the
   conformance test, not a further ruling; what is settled is that the bridge is
   GUM's and not an invention of this specification.

5. **`confidence = 1000` denotes certainty.** Founder ruling, 2026-08-19. It is
   not "maximum representable" and not "no claim made". `ep_exact` constructing
   an exact value with `variance: 0.0, confidence: 1000` is therefore correct
   rather than incidental. `0` is the opposite endpoint: no confidence.

## 8.4 What the ruling buys

Before it, four registered concepts were four independent decisions, each
separately contestable and separately forgettable. After it they are
**consequences of one definition**:

| concept | becomes |
|---|---|
| `SOUNIO-EPISTEMIC-ERASURE` | the rule of the projection operation |
| `SOUNIO-JUSTIFICATION` | the sole re-entry, with its floor |
| `SOUNIO-PROVENANCE` | a member the invariant requires, not an addition |
| `SOUNIO-ADMISSIBILITY` | a reader of the invariant at the point of action |

A rule can be argued away one at a time. A definition has to be replaced whole.

## 8.5 Undefined — rulings owed

- **(c) Withdrawal of the `label` vocabulary — ruled, and BLOCKED on a language
  change.** Founder ruling, 2026-08-19: withdraw the vocabulary; `asserted`
  becomes `Input`. The ruling is right and cannot be executed as written:
  `AstProvInput` has **no surface syntax** (8.2.3). Executing it requires giving
  `input` a keyword, which is a language change and a separate decision.
  Blocked-On: provenance keyword coverage.

- **Where "no confidence claim made" lives.** Ruled 2026-08-19: `1000` is
  **certainty** and `0` is **no confidence**, so the scale `0..1000` is fully
  occupied by meanings. A value constructed without anyone having assessed
  confidence must still carry a number — and every number in range is an
  assertion nobody made. `SOUNIO-NO-VERSUS-UNKNOWN` names the exit: *the defect
  is the collision, not the sentinel; a sentinel no correct value can occupy is
  fine.* The sentinel must therefore live **outside** `0..1000`, or the type
  must make an unassessed value unconstructible. Which of the two is **owed**.
- **Whether `variance = 0.0` is legitimate.** An exact value has no variance; a
  degraded value reports none. The two currently print identically. Whether the
  type admits a genuine zero, or reserves it, is owed.
- **Where provenance lives.** `SOUNIO-PROVENANCE` rules class-in-the-type and
  instance-as-id; neither exists in the struct.
- **Linearity.** Whether `Knowledge<T>` is affine (dropping is an act) is not
  settled; the erasure ruling addressed projection, not discard.
- **`T`'s obligations.** What a type must satisfy to be carried — whether any
  `T` may be, or only those with defined arithmetic — is unstated.

## 8.6 Conformance

The section is `Executable` when, on **both** engines:

- a programme that constructs, propagates and reads uncertainty through a call
  produces the specified variance, and
- a programme that projects and then reads uncertainty is **refused** with a
  named diagnostic, and
- the negative control shows the refusal firing for the stated reason and not
  from name-ignorance.

It is **not** `Claim-ready` on one engine. The current state is the reason: the
FO matrix gives `ADD3 = 0.000000` on Madaros and passing tests on lean_single
for the same source (#1964).

## Claims Forbidden

- Do not read 8.1 as a description of the implementation. The struct is a
  record today; the ruling is what it must become.
- Do not treat 8.3 as implemented. None of the five invariants is enforced.
- Do not fill 8.5 by inference. Those are rulings owed, and a plausible answer
  written there is the failure this corpus exists to prevent.
- Do not cite the `confidence` endpoints as meaning anything until 8.5 is ruled.
