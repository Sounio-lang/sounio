<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-178-extended.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-178-extended.readme
-->

# SIO-178: exact unbounded rational interval addition and subtraction

This increment implements and proves real algorithms for addition, negation and
subtraction of exact rational interval sets, including finite intervals,
one-sided unbounded spans, entire and empty. It does not implement a Sounio
compiler/backend interval primitive or finish general multiplication/division.
No hardware, IEEE, signed-zero encoding or universal certification claim follows.

The isolated base is remote main8eca8e0c086bbd074485bbfd8716accc4063ef1c.
The reviewed finite rational dependency is [PR3118](https://github.com/Sounio-lang/sounio/pull/3118),
sourcec1eb0bbb918d39cde71cdbdc05a4795161228dba. Its three imported files are
byte-identical:

- SounioIntervalFinite.lean SHA17cf45617ddf5aacbcafe8a553cc6b6cd02bb5bfa12eb14103950438fddc172b.
- SounioIntervalFiniteAxiomAudit.lean SHAfb2abaecb5c45f6110195ef03a2414199d21e8269431c68bcbbca205f7ea75a6.
- interval_finite_axiom_inventory.mjs SHA68b816784f992312bec5ad27235a0f8c36797ba5e0b84cfa3af2603d08ea0200.

The reviewed3118 Node-only global inventory port is also imported byte-identically
(scripts/dev/gen_axiom_inventory.sh and scripts/ci/axiom_inventory_source.mjs).
Its seven scanner controls pass. Running that exact port against the untouched
main316-file list reproduces the original inventory byte-for-byte; the candidate
changes only the file census316 to320, retaining59 axioms,741 historical
native_decide uses and0 source proof holes. No Python was executed.
No generated governance snapshot, compiler source or workflow file is changed.
Four additive default Lake targets register the exact finite dependency and
the new model/audit.

## Set semantics and proof boundaries

ExtendedRat has negative infinity, finite Rat and positive infinity constructors.
An Interval is either explicit Empty or a proper span with optional finite lower
and upper bounds. An absent lower bound denotes negative infinity; an absent upper
bound denotes positive infinity. The proper-span proof and member-existence
lemma ensure Empty is the only uninhabited representable interval set.

Membership quantifies over Rat values. Infinities are endpoints and are never
members, including through ContainsExtended. Empty exposes endpoints [+inf,-inf]
and Entire exposes [-inf,+inf]. This is an exact rational model, not a Lean Real
theorem or a bit-level floating-point model.

The internal make constructor interprets computed reversed finite bounds as the
empty set. It is not the runtime raw-input validator: the ADR distinction between
invalid-input refusal and a valid Empty remains an external runtime requirement.

Addition computes the two sum bounds, propagating infinity only from an actually
unbounded corresponding input side. Negation swaps and negates the bounds;
subtraction calls addition with that negated interval. Empty propagation and
nonempty addition are proved. Finite addition agrees with the imported canonical
finite engine. Half-line witnesses reject the invalid fallback of returning
Entire for all nonfinite operands.

Decorations use ill/trv/def/dac/com order. Combining operands takes the weaker
input decoration; Empty caps at trv and a nonempty unbounded result caps at dac.
The proofs establish those represented order constraints and non-strengthening;
they are not a hardware-decoration validation.

Outward addition/subtraction use the unchanged finite model Rounding parameter.
Its rd(q)<=q<=ru(q) fields are explicit hypotheses, not new global axioms. Their
binding to actual IEEE directed operations is explicitly NOT ESTABLISHED.

## Actual qualification

Heavy builds run only in Slurm UID1000 on cpu-ops, one requested CPU and4GiB.
Lean4.33.0 commitd8b18978322de05a8f3dba51ef03cf5461676c17 is reused in a new
isolated worker prefix from the previously verified official toolchain; no
system package or production configuration was changed.

- Slurm17542: preserved development failure. The cap_rank proof split the wrong
  match shape; the algorithm was unchanged in its proof-only correction.
- Slurm17544: COMPLETED0:0 in9s. Scoped Lake build of all four targets, fresh source
  elaboration, and fresh #print axioms for40 new theorems passed. The original32
  finite theorems and their audit remain unchanged.
- The final expanded gate checks full toolchain/source identity, all theorem
  footprints, administrative guard controls and four actual semantic mutations.
  Slurm17545 completed0:0 in13s with all19 gate phases successful.

The first lean_provar consultation returned Mathlib-only tactics. The candidate
was not accepted; this Std-only model was actually checked by Lean.

## Pending programme work

General unbounded interval multiplication, reciprocal/division with pole and
zero-set cases, their decorations, and corresponding enclosure proofs remain
open. So do compiler primitive/IR/literal/optimization integration and x86/RISC-V/
RVV/PTX cross-target qualification. Finite rational proofs and test-only C++ ISA
witnesses remain distinct from those tasks.

Tracking: [Notion SIO-178](https://app.notion.com/p/4033f37f93718110967acf47f65cbe51).
No main merge, auto-merge, enqueue, quota increase or historical artifact deletion.

## Final receipt and replay

Slurm17545 confirms40 extended theorem rows and32 original finite rows. Both
fresh source elaborations and both fresh axiom audits pass. Allowed footprints
are only propext, Classical.choice and Quot.sound. The ten administrative guard
controls and nine unchanged finite-guard controls pass. Four separately mutated
Lean sources fail in the required mathematical theorem ranges: changing the
sum bound to subtraction, changing an infinite upper endpoint, admitting infinity
as a member, and strengthening an empty decoration. None passes through an
import/parser failure.

Sources, exact Lean/lake binaries, complete toolchain libraries and Node hashes
verify after execution. The compact extended04-text.tar.gz is808795 bytes,
SHA256cc4f74bee8bc262bb90dedc9b136321186aff1d3ff054c52092588421bad4e80.
Its internal manifest retains both source versions, original failure, successful
source/audit outputs, all real mutants, job scripts, Slurm states and hash checks.
Executables, compiled proof objects and the toolchain stay outside Git.

Original worker root:

    /scratch/sounio-stdlib-inference-scale-20261024/SIO178-extended

The isolated reusable prefix is toolchain/lean-4.33.0-linux below that root.
Run scripts/ci/interval_extended_lean_gate.sh with a fresh absolute output path
in an allocated Slurm job, explicit LEAN_PREFIX and the retained Node on PATH.
The gate refuses a different Lean/lake binary or changed finite dependency.
