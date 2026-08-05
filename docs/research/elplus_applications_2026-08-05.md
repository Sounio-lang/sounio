<!-- docs:meta
topic_id: repo.docs.research.elplus-applications-2026-08-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.elplus-applications-2026-08-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# EL+ role-aware closure — application integrations (2026-08-05)

Lane: `kimi-cli1 / elplus-apps-20260805`.  Branch:
`research/zd-fiber-antisymmetry-lemma-20260731`.

This note records where `stdlib/ontology/elplus.sio` (the executable
mirror of `formal/OntologyELPlusClosureComplete.lean`) is applied
*beyond* the ontology-frontiers prototypes, and the state of each
integration.

## 1. SNOMED mini-closure adapter — `stdlib/ontology/biomedical/snomed.sio`

The `SNOMEDElplus` struct interns SNOMED relationship triples into the
dense EL+ concept table, closes them with `elplus_fixpoint_packed`, and
exposes O(1) `snomed_subsumes_elplus` / `snomed_subsumes_ex_elplus` /
`snomed_role_targets_elplus` queries.

**State: partially working, repair in flight (not this lane).**
`add_isa` / `add_rel` / staging / stated-edge subsumption work, but
`snomed_elplus_close()` runs the cross-module fixpoint **inertly** from
the impl-method frame on the current Madaros lane: the fixpoint reports
`rounds=1` and derives nothing (no transitivity, no role edges), so only
stated + seeded cells are visible to queries.  Bisect probes (this lane,
2026-08-05):

- `elplus_fixpoint_packed` is exact when called from `main` or from a
  free function of an importer module — even for an imported module
  defining a big-struct adapter with an impl method (probe v4 shape).
- The same call from the real `SNOMEDElplus.close()` in `snomed.sio` is
  inert.  A same-file minimal method-frame probe also lost *stated*
  edges when the local `s` was initialised by copying a large bool
  struct field (`self.s[k]`) in a loop; staging via i64 side arrays
  (the shipped design) avoids that specific failure but not the inert
  fixpoint.

Lane `elplus-optimize-20260805` has an uncommitted `snomed.sio` rewrite
(module-level `CLOSE_S`/`CLOSE_R` scratch + free-function trampoline,
working tree 2026-08-05) targeting this; as of this writing the probe
still shows `rounds=1`.  A handoff message with the probe results was
sent via `bin/sounio-coord`.

**Gate / acceptance test:**
`tests/stdlib/ontology/test_snomed_elplus_adapter.sio` (auto-discovered
by `scripts/run_sio_test_suite.sh`).  It drives the full public adapter
API on the classic EL+ heart fragment and asserts the role-aware
derivations (`finding_site ⊑ part_of`, `part_of ∘ part_of ≤ part_of`).
It is currently annotated `//@ known-failure` with the reason; when the
close() repair lands it flips to XPASS and the annotation must be
removed.

The older local-array demo
`examples/ontology/biomedical/snomed_elplus_demo.sio` does **not**
exercise the adapter struct and passes under both engines today.

## 2. Clinical safety: role-aware DDI smoke demo — `examples/clinical/ddi_elplus_demo.sio`

New integration (this lane).  Bridges `stdlib/chemistry/ontology.sio`
(ChEBI-grounded drug IRIs — including a newly exported
`ketoconazole_chebi()`, CHEBI:48339) with `elplus_fixpoint` +
`elplus_derive_conflicts`: drug classes are keyed by their ChEBI ids
(`midazolam_chebi()` etc.), a mini pharmacological TBox states
`metabolized_by` / `inhibits` / `transported_by` / `part_of`
existentials, and the role layer adds what a plain is-a walk cannot:

- role hierarchy: `inhibits ⊑ alters_activity_of`
- composition chains: `part_of ∘ part_of ≤ part_of`,
  `metabolized_by ∘ part_of ≤ metabolized_by` (Galen-style)

A stated disjointness
(`CYP3A4_substrate_therapy` ⊥ `strong_CYP3A4_inhibitor_therapy`) plus
`elplus_derive_conflicts` then **derives** the patient-level interaction
flags midazolam × ketoconazole and tacrolimus × ketoconazole, while the
P-gp-only digoxin pairs stay clear and warfarin (CYP2C9) never reaches
the CYP3A4 classes; the tacrolimus pair additionally derives an
NTI-based high-severity flag.  The header carries an explicit claims
split (logical fidelity via the Lean anchor vs pharmacological
adequacy as an unproven external assumption) written in response to the
mandatory hostile offload review (xai/grok-4.5, 2026-08-05, logged in
`.claude/llm_offload_log.md`): the output is a *potential
pharmacokinetic interaction flag*, not a clinical contraindication.

Engine: **lean_single only** (`stdlib/chemistry/ontology.sio` is a
lean_single-lane module; its `&str`/`&string` workaround predicates do
not type-check under the current Madaros lane).  The elplus engine
itself runs under both engines.

Gate:

```bash
bash scripts/clinical_ddi_elplus_gate.sh   # pins SOUNIO_SOUC_ENGINE=lean_single
```

prints `CLINICAL_DDI_ELPLUS_GATE_OK` on success.

Scope (`claims_not_made`): no quantitative AUCR prediction — that stays
with the `darwin_pbpk::ddi` lane
(`tests/stdlib/clinical/test_midazolam_ddi_e2e.sio`); no DrugBank-scale
coverage; not a bedside alerting product.

## 3. Clinical safety: PGx diplotype → phenotype → safety demo — `examples/clinical/pgx_elplus_demo.sio`

New integration (PGx lane off the same branch).  PGx is *gene-drug*,
not drug-drug: the knowledge that `stdlib/darwin_pbpk/pgx/*.sio` today
hand-codes as per-drug activity-score thresholds is stated as a mini
EL+ TBox (20 base concepts: 5 drugs, 4 enzymes, 6 allele-function
classes, 4 phenotypes + 1 interned conjunction; 3 roles) and closed
with `elplus_fixpoint` **once per patient diplotype extension**:

- allele hierarchy: `*4 ⊑ NullAllele ⊑ NoFunction`,
  `*10 ⊑ ReducedFunction`
- phenotype definitions: `PM ⊑ ∃hasVariant.(NoFunction ⊓ NoFunction)`
  (conjunction filler — honestly noted as logically ≡ `NoFunction` in
  EL+; it exercises conjElim/conjIntro), `IM ⊑ ∃hasVariant.ReducedFunction`
- ONE role chain: `metabolizedBy ∘ hasVariant ⊑ contraindicated`
  (read as *contraindicatedAtStandardDose* — a review flag, not a
  dosing instruction)

For a CYP2D6 `*4/*4` (PM) patient the closure derives
`haloperidol/venlafaxine/risperidone ⊑ ∃contraindicated.NoFunction`
while clopidogrel (CYP2C19, retained only as a negative control —
prodrug direction is explicitly out of scope) and midazolam
(CYP3A4/3A5) stay clear; `*4/*10` (IM) derives only the weaker
`∃contraindicated.ReducedFunction`; `*1/*4` (NM per the numeric lane's
thresholds) derives nothing.  A chain-disabled (nchain = 0) regression
run proves the flag is chain-derived, not seeded.  The boolean
`elplus_subsumes_dense(drug, ∃contraindicated.NoFunction)` then GATES
the existing numeric lane
(`darwin_pbpk::pgx::cyp2d6_haloperidol::halo_cyp2d6_adjusted_cl`): a
hard flag withholds the standard-dose scaling (sentinel), a clear gate
applies the phenotype multiplier, and each patient asserts the two
layers agree (`gate == (activity-score phenotype == PM)`).

Engine: **default (Madaros)** — passes under BOTH engines (no
lean_single-only import; contrast §2).  The demo had to be written in
the SNOMED-demo shape (`let` ids inside `main`, separate concept
arrays, 14-parameter `elplus_fixpoint` called directly from `main`, no
const-referencing helpers): a first draft using module-level `const`
ids referenced from helper fns segfaulted on Madaros while passing
under lean_single — now filed as **D6** in
`docs/compiler/KNOWN_LIMITATIONS.md`.

Gate:

```bash
bash scripts/clinical_pgx_elplus_gate.sh   # default engine
```

prints `CLINICAL_PGX_ELPLUS_GATE_OK` on success.

Mandatory clinical-pathway offload review (xai/grok-4.5, 2026-08-05 —
deepseek was out of balance, provider fallback per policy; logged in
`.claude/llm_offload_log.md`): 13 findings; applied fixes were the
codeine → risperidone swap (prodrug direction off the positively
checked contraindication path), EL+/numeric layer-consistency
assertions, the chain-disabled regression, and header honesty notes
(PM conjunction equivalence, delegated diplotype → phenotype
classification, contraindicatedAtStandardDose role semantics, no
per-TBox Lean instantiation).  Left as documented scope: toy allele
coverage (only *4/*10 interned), no UM patient, magic dense strides
(house ABI).

## Conventions worth knowing (all three integrations)

- Named imports only (`use ontology::elplus::{...}`); the qualified
  form miscompiles.
- The role-hierarchy matrix `rclos` has stride `ELPLUS_MAXR = 8` and
  role indices must be `0..nr-1`; state role inclusions *before*
  calling `elplus_role_closure` so saturation sees them.
- `exid[role*64 + filler]` must be `-1` for non-interned pairs: the
  fixpoint's roleSub/roleComp/RtoS rules only fire onto interned
  existentials (the `conceptUniv` restriction of the Lean
  development), so every existential you want to *query* must be
  interned up front.
- If a conjunction filler must propagate through Rmono/roleComp, its
  concept id must lie inside the base range `0..nb-1` (fillers iterate
  base concepts only); intern conjunctions before the existentials.
- Module-level `const` ids must be referenced ONLY from `main` on the
  Madaros lane (D6, `docs/compiler/KNOWN_LIMITATIONS.md`): use `let`
  bindings inside `main`, or pass values as parameters to helpers.
