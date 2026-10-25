<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-178.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-178.readme
-->

# SIO-178 finite rational enclosure proof increment

Status: **in progress**. This is the first finite proof frame of the accepted
[ADR-019](../../decisions/adr-019-certified-intervals.md), not completion of its
extended interval model, backend qualification or hardware certification.

The existing formal/lean4 Lake project now builds SounioIntervalFinite and its
separate SounioIntervalFiniteAxiomAudit target. It uses core Lean4/Std/Rat, with
no Mathlib dependency or new compiler/backend/Sounio interval API changes.

## Exact mathematical scope

Interval carries rational lo,hi endpoints and a proof lo<=hi. It is finite,
NaN-free and nonempty by construction. Contains is the ordinary conjunction
lo<=x and x<=hi. Rational zero identifies signed zeros: no binary sign-of-zero
property is represented.

The kernel checks universal inclusion theorems for all four operations:

| Operation | Endpoints / domain | Main theorem |
|---|---|---|
| Addition | lo+lo, hi+hi | add_contains |
| Subtraction | lo-hi, hi-lo | sub_contains |
| Multiplication | min/max of all four endpoint products | mul_contains |
| Division | min/max of all four endpoint quotients; denominator strictly positive or strictly negative | div_contains |

The product proof splits on scalar signs and composes primitive rational
multiplication/order laws. It does not assume nonnegative inputs. Reciprocal
reverses order on both allowed divisor domains; denominator_member_ne_zero
explicitly proves every divisor member is nonzero. Definitions expose the four
quotient corners in reciprocal endpoint order. Closed kernel checks cover mixed
sign multiplication and positive/negative division; they also prove that a
zero-crossing or zero-touching denominator is outside this proof frame.

This establishes enclosure over exact rationals, not tight binary bounds,
a floating error certificate or a proof of a particular Sounio implementation.

## Conditional primitive rounding, not an IEEE instance

Rounding is an explicit parameter with two functions and exactly these two
primitive inequality premises:

~~~lean
rd : Rat -> Rat
ru : Rat -> Rat
rd_le : forall q : Rat, rd q <= q
le_ru : forall q : Rat, q <= ru q
~~~

These assumptions are **NOT ESTABLISHED against IEEE bits or hardware**.
The four rounded inclusion theorems are conditional on a supplied Rounding
value. There is no global RD/RU axiom and no assumption of the interval
enclosure conclusion. Exact inclusion is proved first.

Addition/subtraction apply RD/RU to their exact scalar endpoint results.
Multiplication/division round EACH of the four scalar corners individually,
then select min/max. The proof derives componentwise min/max inequalities from
rd_le/le_ru; it assumes no monotonicity, idempotence, representability, tightness,
or exactness property of the rounding maps. Definition theorems expose the
individually rounded product/quotient lower corners. The two scalar premises are
also printed separately in the audit because parameters are not global axioms.

The carrier is an **abstract unbounded rational carrier** with total finite
rational rounding maps. It cannot be instantiated globally by a finite IEEE
format in the presence of overflow. A hardware binding must establish a
restricted local finite supported-corner domain with suitable premises, or use
a future correctly proved extended-rational model admitting infinities.
The identity maps witness mathematical consistency of the present premises;
they are explicitly not IEEE directed-rounding instructions.

## Axiom inventory and actual checking

The separate audit prints every theorem's transitive axiom inventory and checks
both rounding-premise field signatures. All32 theorems in the final source use
only the standard Lean foundations:

- propext
- Classical.choice
- Quot.sound

No sorryAx, Lean.trustCompiler, custom arithmetic/enclosure axiom, admit or
native_decide is accepted. The closed rational checks use decide +kernel.
check_axiom_inventory.mjs validates TEXT of the actual Lean output, exact theorem
coverage and the explicit primitive-premise inventory; Lean checks the proofs.
Its nine negative controls reject missing/duplicate rows, omitted premise
inventory, forbidden/custom axioms and source escape constructs.

The first lean_provar request supplied the complete four-corner rational theorem
with a proof hole for tactic search. It returned incomplete prose and no Lean
proof; none of that response was accepted as a proof. All accepted source was
subsequently checked by the actual pinned Lean kernel.

## Toolchain and custody

The package pins leanprover/lean4:v4.33.0. The workspace had an elan launcher but
no installed toolchain. After recording829GiB available on the workspace PVC,
the official pinned Linux artifact was provisioned in an isolated directory;
no global elan installation, shared quota or other agent artifact changed.

- Official [release v4.33.0](https://github.com/leanprover/lean4/releases/tag/v4.33.0),
  release367714334, published2026-08-10.
- Asset508371049, lean-4.33.0-linux.tar.zst,574882764 bytes.
- GitHub API digest and actual archive SHA256:
  4b3fb03c29a1e0a253fb1d11f9bae3725f19a0dc6fc09b3ea16d2c9df3349e2c.
- Archive checksum verified BEFORE extraction/execution.
- Actual Lean version4.33.0, commit
  d8b18978322de05a8f3dba51ef03cf5461676c17, x86_64 release.
- Actual Lean binary SHA256:
  e8baaa71855a616dc351028f3ad2200051b0671f423a1696a100e809302d5550.
- Actual Lake binary SHA256:
  60330ab6f07dce20f3fa9ebb08e8b984ea9549eac172afeb15d9d2227060e2b3.

Long work runs in remote tmux. The official archive uses548MiB compressed and the
isolated unpacked toolchain2.9GiB; it is retained once outside Git. The local
Lake build tree is2.7MiB. No Python/Rust/Julia scientific reference is used.
Node performs archive decompression, hashing and text-inventory checks only.

Final build07 passes: actual scoped build2.327s, fresh source kernel check1.693s,
all32 theorem inventories and nine text-guard controls. It records scoped Lake
build, fresh source kernel checking,
separate audit invocation, all source/toolchain/artifact identities and the
nine-control guard. Scoped builds do not claim that every other default target
in this large formal project has been rebuilt or accepted. Initial scaffolding
errors are retained separately: build01 did not invoke Lake because this
workspace lacks /usr/bin/time; build02 exposed missing core Rat theorem/instance
names; build05 reached successful Lean checks but lacked Node on its bounded
PATH. These are not theorem passes. The final script selects the exact binaries.

## Reproduction and boundaries still open

With the verified toolchain's bin directory on PATH, from formal/lean4:

~~~sh
lake build SounioIntervalFinite SounioIntervalFiniteAxiomAudit
lake env lean SounioIntervalFinite.lean
lake env lean SounioIntervalFiniteAxiomAudit.lean > AUDIT_STDOUT
~~~

Then, from the repository root, use a Node runtime for text checks only:

~~~sh
node docs/evidence/stdlib-SIO-178/check_axiom_inventory.mjs --selftest
node docs/evidence/stdlib-SIO-178/check_axiom_inventory.mjs \
  formal/lean4/SounioIntervalFinite.lean AUDIT_STDOUT
~~~

The source basis is reviewed SIO-177 aec65f8285953dce86a56dc27a633ac4587d1b04,
with main7ce merged into the OWN SIO-178 branch, preserving the177 graph.
The merge added only the upstream workflow and internal release fragment.

Still open: the Rat±infinity model; empty/entire encodings;0*infinity and infinite
differences; binary signed zeros; decorations; disconnected division/zero-crossing
hulls; IEEE f32/f64 bit semantics; overflow/subnormal behavior; concrete RD/RU
instruction correctness; FP environment barriers; and the one-million-case
per-format/per-operation cross-backend hardware qualification. This increment
does not change those obligations or complete SIO-178.
