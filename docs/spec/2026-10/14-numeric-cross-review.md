<!-- docs:meta
topic_id: repo.docs.spec.2026-10.14-numeric-cross-review
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.14-numeric-cross-review
-->

# Independent scientific-information cross-review of chapter 05

Review date: 2026-10-16. Worktree:
`/tmp/sounio-main-root-cleanup-20261016`. Source HEAD:
`f5b2c5850be9f3d3995719ffcade44a6d851658a`.

Reviewed chapter SHA-256:
`9d62e97088e774a68ccb985bf5e60c14f5113ddf6c5b2fe208f5676351a6e435`.
Reviewed `05-evidence.tsv` SHA-256:
`f1576014cf6cb477d17b946b4d5e6b908d61fb522176070e9f95c16115ddf91c`.
Line references below identify that inspected snapshot; paths and symbols remain
the primary references if the chapter changes.

Scope: chapter 05, its companion evidence catalog, and matching compiler,
stdlib, fixture, and gate sources. This review did not read or modify other
chapters. Only this report is written. No compiler, gate, numerical oracle,
adversarial executable, or Lean build was run. Findings are source/contract
discrepancies and a consequential support-boundary omission, not newly measured
runtime failures. No numerical calculation was used as a verification oracle.
Canonical external xai/kimi/offload review remains the parent lane's separate
checkpoint; this report does not duplicate or certify it.

## Verdict

No BLOCKER identified within this inspection. Three MAJOR findings need
chapter/evidence qualifications before the chapter's affected statements are
used as a scientific-information contract. The first two are direct mismatches
between prose and inspected executable source. The third is a missing critical
scope boundary, not an assertion that the chapter explicitly promises wide
Knowledge support.

| Finding | Severity | Affected rules | Principal source |
|---|---|---|---|
| NCR-01 | MAJOR | `NUM-022` and its evidence/gate wording | `scripts/ci/madaros_f128_f256_v0e41_fail_closed_lower_gate.sh`, negative-compile classification |
| NCR-02 | MAJOR | `ALG-011`, in the context of `ALG-010` | `stdlib/eisa/core_v2.sio::egate2` |
| NCR-03 | MAJOR | `NUM-020`, `NUM-026`, and wide-float/epistemic boundary coverage | `self-hosted/ir/lower.sio::lower_type_expr_knowledge_value_float` and `Lowerer.lower_measure_call_ref` |

## NCR-01 — The alleged stronger refusal gate does not always establish nonzero exit and no ELF

**Rule and prose.** `05-numerics-units-and-algebra.md:437`, `NUM-022`, says
the V0-E.4.1 gate checks a negative compile result and absence of an emitted ELF,
then recommends it as the stronger gate for refusal before artifact emission.

**Current source.** In
`scripts/ci/madaros_f128_f256_v0e41_fail_closed_lower_gate.sh:138`:

- The first success branch requires the sentinel and `! -s "$OUT"`, but does
  not require `c_rc` to be nonzero. A zero-byte artifact also satisfies the
  size test, so this is not literal file absence.
- The second success branch at line 140 requires a nonzero compiler exit and
  a broad related-diagnostic expression, but does not check output absence or
  size. It precedes the emitted-artifact failure branch.
- The later stale-E249 success branch is another explicitly different
  classification and must not be equated with current-source lowering refusal.

**Why it matters.** A compiler returning nonzero with a matching diagnostic
and a nonempty output can take the second success branch before the script
examines the artifact. Thus the gate's aggregate success marker does not imply
the conjunction claimed in the chapter. The issue is not whether such an
artifact is currently produced; the gate fails to establish its absence on
every admitted success path. This is a provenance/evidence-strength problem
even if the current compiler normally refuses correctly.

**Required chapter correction.** Describe the exact classified outcomes and
limit any no-output claim to the branch/receipt that actually checks it. Do not
recommend aggregate V0-E.4.1 success as proof of both a negative exit and absence
of an ELF. The corresponding `NUM-022` catalog gap should identify these
weaker branches rather than only contrasting V0-F.5 with V0-E.4.1.

**Discriminating future gate check, not executed.** Isolate classification and
provide a compiler-result stub with a nonzero status, a related `f256`
diagnostic, and a nonempty output artifact. The gate must reject that outcome
before advertising no-artifact enforcement. Also distinguish nonzero status,
sentinel specificity, nonexistent output, zero-byte output, and stale-frontend
refusal. No compiler or gate repair is performed by this review.

## NCR-02 — EISA v2's gate tests the leading correction component, not the full correction magnitude

**Rule and prose.** `05-numerics-units-and-algebra.md:911`, `ALG-011`, groups
`egate` and `egate2` as comparing correction magnitude with `u`, and says a
nonpositive budget returns okay only for zero correction.

**Current source.** `stdlib/eisa/core.sio:196`, `egate`, obtains its scalar
test quantity through `dd_to_f64(dd_abs(x.err))`.
`stdlib/eisa/core_v2.sio:160`, `egate2`, instead assigns
`roundoff = core_v2_abs_f64(x.err.x0)`. It does not inspect, sum, bound, or
validate `x1`, `x2`, and `x3`. Its zero-budget branch tests that scalar alone.
The preceding source comment explicitly describes the leading-component
collapse. In contrast, `core_v2_zero_flags_raw` at line 78 checks every
correction component for nonzero status.

**Why it matters.** Chapter 05 correctly treats the corrected qd quantity as
distinct from nominal binary64 and correctly warns about renormalization
assumptions in `ALG-010`. The gate paragraph must retain that distinction too:
the predicate `abs(err.x0)` is not the definition of the magnitude of the
four-component correction. Even for well-formed expansions, dropping small
components is an approximation requiring a threshold/rounding qualification;
the inspected gate establishes no exact full-expansion classification theorem.

For externally assembled/noncanonical `EReg2` values, the source distinction is
more immediate: a zero leading component with a nonzero trailing component is
invisible to this gate's correction read but visible to the zero flags. This
is a structural branch observation, not an executed numerical counterexample
or a claim that such an expansion satisfies the library's normalization model.
The gate itself does not enforce that model as an admission check.

**Required chapter correction.** Name the exact quantities separately:
v0's rounded scalar collapse of its double-double correction, versus v2's
absolute leading component. Replace “only for zero correction” with the
implemented zero-test predicate, and state the normalization assumptions and
lack of a whole-expansion threshold guarantee. Decide separately whether
noncanonical input is prohibited, validated, or merely an unchecked caller
obligation. The `ALG-011` evidence gap should record the omitted components.

**Discriminating future checks, not executed.** Compare gate classification
with a complete correction representation near threshold boundaries under a
declared canonical-expansion model. Separately test a fabricated trailing-only
correction against the public gate and zero flags, with the intended admission
policy fixed in advance. A nonfinite-input test alone does not exercise this
information-loss boundary.

## NCR-03 — The wide numeric support map omits Knowledge payload and uncertainty precision boundaries

**Affected scope.** `NUM-020` discusses wide refusal boundaries;
`NUM-026` at `05-numerics-units-and-algebra.md:503` maps f128 scalar/aggregate
support and language-f256 refusal. Neither supplies a `Knowledge<f128>` /
`Knowledge<f256>` / wide-GUM row or exclusion. The few general `Knowledge`
mentions elsewhere do not resolve the representation and uncertainty-width
questions.

**Current source.** In `self-hosted/ir/lower.sio:9392`,
`lower_type_expr_knowledge_value_float` handles f64/f32 payloads, integer
payloads, and a fallback intended for nominal brands. That fallback returns
the float marker; it is not a distinct binary128 or binary256 payload contract.
`ir_register_knowledge_layout` at line 2707 supplies the synthetic variance and
confidence fields with the ordinary float marker.
`Lowerer.lower_measure_call_ref` at line 28124 computes measurement variance
through explicitly ordinary floating multiplication and seeds the existing
scalar channel system. None of those inspected branches establishes a
wide-softfloat variance/covariance propagation route merely because the nominal
argument or its annotation uses a wide type.

Most importantly, `tests/compile-fail/f128_nested_knowledge_reserved.sio:2`
is ignored. Its own header explains that the integer initializer confounds
the old reserved-type diagnostic, that E249 is no longer applicable, and that
the fixture cannot discriminate a wide-epistemic refusal. It is not a usable
negative witness for the current boundary.

**Why it matters.** Scalar f128 arithmetic, literal exactness, and selected
aggregate transport do not establish precision-preserving `Knowledge<f128>` or
GUM over that format. Likewise, the standalone language-f256 negative does not
prove uniformly fail-closed behavior for every nested epistemic shape. A
scientific reader needs to know separately the nominal payload representation,
variance representation, derivative representation, and unsupported-wrapper
admission policy. This is a material omission in a numeric support map for an
epistemic language, rather than a counterexample to the chapter's carefully
bounded scalar claims.

**Required chapter correction.** Explicitly exclude unestablished
wide-`Knowledge` and wide-uncertainty propagation from the scalar support rows.
Record that the ignored historic fixture supplies neither a dedicated current
diagnostic nor an execution guarantee. Do not assert that the wrappers are
uniformly refused without a new distinguishing source path and witness. An
explicit cross-reference to the corresponding epistemic chapter can explain
the details after the parent checks consistency; this review did not read that
chapter as evidence.

**Discriminating future checks, not executed.** Use valid nominal wide
arguments rather than an unrelated integer-initializer mismatch. Inspect
value bits/limbs and uncertainty representation independently at construction,
arithmetic, extraction, return, and call boundaries; include explicit refusal
tests where the route is unsupported. Positive scalar f128 output and a
language-f256 refusal test are not substitutes for these witnesses.

## Claims checked without a new MAJOR finding

- `NUM-011` correctly identifies the native f32 cast as a declared
  binary64-retention residual; its bit-pattern fixture tests that residual,
  not binary32 conformance.
- `NUM-017` accurately transcribes the current format descriptors. The
  `F256Bits` add/sub library and executable language-f256 paths remain
  correctly distinguished in `NUM-021`; the concern above is gate strength
  and wrapper scope, not the existence of the limb library.
- `NUM-025` correctly identifies the rounded-quotient/unit-peeling remainder
  algorithm, zero-divisor infinity branch, and finite iteration cap as
  implementation defects/limits rather than fmod semantics. No runtime
  remainder counterexample was executed.
- `UNIT-004` through `UNIT-008` accurately preserve the distinction among
  dimensions, stored rational scales, runtime magnitudes, cast acceptance,
  cancellation, and table exhaustion. Current source supports their stated
  qualifications; no additional MAJOR unit-formula mismatch was established.
- `ALG-006` and `ALG-007` correctly separate bounded exact sparse-product
  tokens from caller-supplied measured-norm receipts. `ALG-009` correctly
  retains the deliberate subtraction envelope and the loss of origin IDs at
  the instruction-machine boundary. Those findings are not new objections.

Minor terminology follow-up: `UNIT-001` repeats the source label “luminosity”
for the seventh SI-style dimension. The physical base dimension is luminous
intensity; retain the source spelling as an implementation label if needed,
without silently treating it as physical luminosity. This is not classified
as an additional MAJOR finding here.

## Review limitations and handoff

This is a static scientific-information review, not an exhaustive arithmetic
audit, proof review, independent numerical oracle, or compiler qualification.
Suggested discriminants remain unrun. No rule is marked passed, and no source
defect is claimed fixed. The parent owns chapter 05 edits, evidence-catalog
updates, external reviewer reconciliation, and any subsequent executable work.
