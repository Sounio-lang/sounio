<!-- docs:meta
topic_id: repo.docs.decisions.adr-009-verified-foreign-reference-oracles
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-009-verified-foreign-reference-oracles
-->

# ADR-009: Verified Foreign Reference Oracles

**Status**: proposed
**Date**: 2026-09-04
**Supersedes in part**: ADR-008 (adds a new `oracle_class`; does not reopen
Python/mpmath/SciPy to claim authority)
**Related**: ADR-008 (claim-oracle semantic clock), ADR-006 (fixed-point
seed scope), `scripts/dev/claim_oracle_inventory.sh`

---

## Authority paragraph (binding)

ADR-008 correctly identified that a dynamically-typed, unverified foreign
runtime (Python/mpmath/SciPy) must never be the sole judge of a Sounio
language or library claim, and correctly proposed `sounio_closed_form_twin`
(two independent Sounio implementations agreeing) as the default
replacement. In practice, self-twin has a gap: two Sounio implementations
of the same routine can share the same misunderstanding of the underlying
math, effect semantics, or hardware model. Self-agreement is not
independence.

This ADR adds one new oracle class, **`verified_foreign_reference`**, for
foreign implementations rigorous enough to carry independent authority —
distinct from, and strictly stronger than, `external_corroboration_only`.
Python does not qualify and is not reclassified by this ADR; its role
(measurement, corroboration, bug-hunting, research harnesses) is unchanged
from ADR-008.

## Decision

### 1. `verified_foreign_reference` — admission criteria

A foreign reference implementation may be classified
`verified_foreign_reference` (may hard-fail CI for the claim it covers)
only if **all** of the following hold:

1. **Independently authored** — written from the specification/paper, not
   derived from or transliterated line-by-line from Sounio source.
2. **Statically typed, pure, or proof-carrying** — the language must
   structurally rule out the failure classes Python cannot (implicit
   mutation, untyped numeric coercion, unchecked partiality), or must
   carry a machine-checked proof of the property in question.
3. **Pinned exact toolchain version** — compiler/prover version recorded
   in the gate script and in `artifacts/audit/claim_oracle_inventory.tsv`.
4. **Documented insufficiency rationale** — a comment or linked note
   explaining why `sounio_closed_form_twin` is not enough for this
   specific claim (e.g. "both Sounio paths reuse the same series
   expansion" or "no second Sounio implementation exists yet for this
   effect handler").

Gates that do not meet all four criteria stay `external_corroboration_only`
or `research_harness` per ADR-008, regardless of implementation language.

### 2. Domain → language mapping

This is guidance for which language to reach for, not a hard requirement
to use all five. Pick the narrowest one that satisfies the four criteria
above for the claim at hand.

| Domain | Oracle language | Why |
|---|---|---|
| GPU / data-parallel numeric kernels | **Futhark** | Pure functional, compiles to real GPU code; no hidden imperative state to diverge from Sounio's own GPU codegen (`self-hosted/gpu/`). Matches existing CUDA `MATERIAL_PARITY` kernel work — Futhark can replace the ad-hoc CUDA/C++ digest generator as the independent ground truth. |
| Formal / provable properties (invariants currently only asserted, not proven) | **F\*** | Dependently-typed, proof-carrying; used in production for verified crypto/parsers (Project Everest). Appropriate where the claim is "this property always holds," not just "these examples agree." |
| Effect-system semantics (if/when Sounio's effect typing needs an external check) | **Koka** | Reference implementation of algebraic effect handlers; checks Sounio's effect semantics against the language that originated the theory, not against an ad-hoc reimplementation. |
| General functional / algebraic semantics | **F#** | Typed, ML-family, immutable-by-default; catches the class of bugs Python's implicit mutability hides, without the proof-engineering cost of F*. |
| Bootstrap / performance-contract integrity | **C++23** (existing, via Poseidon Stage 0) | Already the bootstrap oracle per `RUSTLESS_COMPLETE.md`; this ADR does not change bootstrap_integrity classification (ADR-006 territory), only notes that C++23 is the existing SOTA choice there and no swap is needed. |

### 3. Interaction with the pilot demotion table (ADR-008 §Pilot demotion)

Rows in ADR-008's pilot demotion table stay `external_corroboration_only`
(Python soft/report-only) **unless** a `verified_foreign_reference`
replacement is authored and passes the four admission criteria. Priority
order, highest-value first, based on current repo state:

1. `MATERIAL_PARITY` GPU kernel digests (currently CUDA/C++ ad-hoc,
   see `tools/pireus/dgx_ptx_shfl_material_parity.cu`) → Futhark.
2. `special_scipy_parity_gate.sh` / `linalg_parity` / `stats_dist_parity`
   → candidate for F* where a provable bound exists, otherwise stays
   `sounio_closed_form_twin` per ADR-008.
3. Sedenion / bigrat gates → likely stay `sounio_closed_form_twin`;
   revisit only if drift incidents recur.

### 4. New work rule

New claim-bearing gates may use `verified_foreign_reference` from day one
if a suitable independent implementation already exists and meets the
four criteria; otherwise default to `sounio_closed_form_twin` per ADR-008,
not to a foreign runtime.

## Consequences

- Python/mpmath/SciPy remain `external_corroboration_only` /
  `research_harness`. This ADR does not restore their authority.
- `sounio_closed_form_twin` remains the default when no independent,
  sufficiently rigorous foreign implementation exists — this ADR does not
  mandate rewriting every gate in a foreign language.
- New CI toolchains are introduced only as needed: Futhark first (GPU
  kernel parity has an immediate, concrete target), the others on demand.
- `claim_oracle_inventory.sh` / the TSV schema need a new `oracle_class`
  value; scanner update is a prerequisite for CI enforcement.

## Grounded in

- ADR-008 (claim-oracle semantic clock) — this ADR extends, not reverses,
  its authority paragraph.
- Existing GPU codegen work: `self-hosted/gpu/hlir_to_gpu.sio`,
  `self-hosted/gpu/lower_to_ptx.sio`, and the CUDA
  `dgx_ptx_shfl_material_parity.cu` MATERIAL_PARITY probe.
- Prior art: Project Everest (F* verified crypto/parsers), Koka's effect
  handler semantics, Futhark's pure functional GPU compilation model.
