<!-- docs:meta
topic_id: repo.docs.architecture.madaros-v2-eisa-semantic-ir
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.madaros-v2-eisa-semantic-ir
-->

# Madaros v2 EISA/METRON Semantic IR

Status: normative requirements for a future implementation, not an
implementation claim

Issue: [#751](https://github.com/Sounio-lang/sounio/issues/751)

Target layer: `Semantic IR -> EpistemicNumericIR -> MIR`

Implementation note (2026-07-11): `E1-ENIR-SHADOW-FULL` now provides a
compiler-owned native ENIR model, strict canonical parser/printer, semantic
verifier, deterministic hash, and exact source-derived 30-program/39-observation
manifest under `self-hosted/enir/`. Its gate includes independent checking,
byte-identical roundtrip, nine invalid mutations, valid numeric hash tamper,
manifest tamper, and a zero-diff assertion over compiler lowering/codegen/ABI
surfaces. This is an implemented **shadow foundation**, not the
`E1-ENIR-CORPUS-FULL` lowering claimed below. The historical "no source
lowering/interpreter" boundary was superseded by E2A and E2B below; ENIR-to-MIR
and production code generation remain absent. Run
`scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`; inspect or replay artifacts
with `bin/madaros-enir emit|verify|roundtrip`.

Implementation note (2026-07-11): `E2A-ENIR-V0-STRAIGHT-LINE-FULL`, the first
closed tranche of the still-open `E2-ENIR-LOWERING-FULL` umbrella, implements the
complete source-authored EISA v0 straight-line slice of the canonical corpus:
`golden_mul`, `golden_add`, `golden_sqrt`, `golden_poison`, and
`e5_cancellation` (5/30 programs, 6/39 observations). The compiler-owned parser
lowers source text directly to stage-2 ENIR; it does not call `eisa::backend`,
construct `.eisax`, or consult the E1 fixture. Its supported grammar is an
`epistemic fn` with immutable `let` bindings over finite decimal literals,
prior identifiers, one binary `+ - * /` expression, or `sqrt(operand)`, plus
`gate` and `store [slot] <- value`. Unsupported control flow fails closed.

The compiler-owned ENIR interpreter independently implements the EISA v0 f64,
DD64, scalar GUM1, poison, gate-policy, and memory transitions. The E2 gate
extracts the five sources from `tools/eisa/eisa_evm_run.sio`, independently
reconstructs expected SSA/provenance in Python, checks canonical lowering and
roundtrip, replays all error words and statuses in an independent interpreter,
and compares the six source-observable events with a source-fresh Metron EVM.
It also requires causal source tamper, nine malformed/unsupported source
negatives, two artifact/canonicalization tampers, E1 regression, and zero diff
over production lowering/codegen/ABI/runtime and oracle sources. Run
`scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`, or use
`bin/madaros-enir lower <source>` and `bin/madaros-enir run <enir>`.

Implementation note (2026-07-11): `E2B-ENIR-V1-FINITE-CFG-FULL` adds the
finite v1 slice `v1_loop`, `v1_if_both`, `v1_i6`, `v1_highreg`, `v1e_frail`,
`v1e_emov_negzero`, `v1_arith_high`, and `v1_branch_high` (8 programs and 11
observations). Cumulative real lowering is therefore 13/30 programs and 17/39
observations. `lower-v1` emits explicit `ebr`, `ebrz`, `ebrn`, and `ehalt`
targets; the interpreter executes a dynamic PC and records taken, not-taken,
poisoned, and frail branch decisions. High-value-ID cases reach IDs 20, 23,
and 20, and negative-zero comparison uses raw bits rather than formatted zero.

The E2B SSA boundary is deliberate and executable. A `while` condition is an
immutable preheader value and its body must be empty. An `if` body may contain
`store` operations but may not define values or contain gates; conditional
gates require path-sensitive observation regions. The standalone verifier
recognizes the structured CFG shape, requires final `ehalt`, rejects
arbitrary targets and policies, and rejects conditional definitions. Values
carried across a backedge or merge require block arguments/phi and are rejected
by the schema-v1 path rather than represented as mutable pseudo-SSA. Explicit
`fuel` is likewise rejected by schema v1; E2C provides the separate schema-v2
path described below. The E2B
gate performs a three-way EISA VM/native ENIR/independent Python differential,
causal tamper, 14 source negatives, four artifact tampers, one canonicalization
tamper, E2A/E1 regression,
and a protected-surface zero diff. Run
`scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`, or use
`bin/madaros-enir lower-v1 <source>` followed by `run <enir>`.

Implementation note (2026-07-11): `E2C-ENIR-FUEL-BLOCKARGS-FULL` adds
`v1_fuel`, `v1e_fixedpoint`, and `v1_fuel_high`, bringing cumulative real
lowering to 16/30 programs and 20/39 observations. A canonical schema-v2
resource row carries the exact fuel limit. Four explicit basic blocks
(`entry`, `header`, `body`, `exit`), typed edges, and SSA block arguments model
loop-carried values; source `set` lowers through an alias-safe fresh expression
value and `emov`. The verifier proves canonical predecessor/argument ranges,
edge arity and types, reachability, dominance, and source-value availability.
For a fuel-only observation it additionally proves an immutable empty loop
with a known non-zero entry condition and identity backedge, so exhaustion is a
structural consequence rather than a corpus assumption.

The schema-v2 interpreter binds edge arguments simultaneously and debits fuel
before every cost-one operation or terminator, matching METRON instruction
accounting. The fixed-point witness reaches exactly `2.0` in 15 instructions
with 85/100 fuel left; the two non-terminating witnesses stop at zero fuel and
preserve their true last writes, including value ID 20. The FULL gate requires
a source-fresh EISA/METRON differential, an independent Python parser/lowerer/
interpreter, byte-identical roundtrip, 16 source negatives, 16 structural,
dominance, resource, and canonicalization tampers, E2B/E2A/E1 regression, and
zero diff over production codegen/ABI/runtime. Run
`scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`.

Implementation note (2026-07-11): `E2D-ENIR-V1-RUMP-DD-FULL` adds the
source-authored `v1_rump_dd` flagship, bringing cumulative lowering to 17/30
programs and 23/39 observations. The source contains the complete Rump 1988
DD64 graph as 26 SSA values, 29 ENIR operations, three ordered gates, and a
cost-one halt terminator. Schema v2 now also represents a fuel-bearing
single-block program with zero block arguments and zero edges; normal
termination consumes exactly 30/64 fuel units and leaves 34.

The E2D checker independently normalizes the destructive-register graph from
the frozen `rump_build` image into SSA and requires exact graph identity with
the source lowering. A Python DD64 replay checks both error words, while the
frozen METRON v1 receipts independently check value, formatted roundoff,
uncertainty, poison, frailty, and gate order. Its FULL gate adds 18 source
negatives, 16 structural/resource/observation/canonicalization tampers, a
causal source tamper, E2C/E2B/E2A/E1 regression, and zero diff over the frozen
oracle and production codegen/ABI/runtime. Run
`scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`.

Implementation note (2026-07-11): `E2E-ENIR-V2-QD128-ARITHMETIC-FULL`
adds `v2_const_gate`, `v2_add`, `v2_sub`, `v2_mul`, `v2_div`, and
`v2_sqrt`, bringing cumulative real lowering to 23/30 programs and 29/39
observations. `lower-v2` emits schema-v2 resource-bearing single-block ENIR
with an explicit qd128 error type. Runtime observations expose all four
roundoff words independently; none is reconstructed from a formatted scalar.

The compiler-owned interpreter uses the pinned finite-domain expansion
algorithms in `self-hosted/enir/qd.sio`, without importing `eisa::core_v2`,
`math::qd128`, the Metron VM, or the x86 bridge. Its receipt binds the qd
semantics source and independent Python checker hashes. The checker separately
parses each source, normalizes the corresponding frozen v2 image to SSA,
replays the Hida-Li-Bailey/Priest operation order, and requires exact value,
`error0..error3`, uncertainty, status, graph, fuel, and METRON receipt parity.
An additional `Fraction`/300-digit `Decimal` oracle checks the reconstructed
expansion against the exact rational operations or high-precision square root
under the declared finite qd relative-error bound; it does not reuse the qd
operation implementation.
The FULL gate includes 16 source negatives, 17 artifact/canonicalization
tampers, four runtime all-word receipt tampers, a poisoned divide-by-zero
witness, causal source tamper, E2D/E2C/E2B/E2A/E1 regression, and zero diff over
the shared qd runtime, frozen oracle, and production codegen/ABI/runtime. Run
`scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh`.

Implementation note (2026-07-11): `E2F-ENIR-V2-RUMP-QD-FULL` adds the
source-authored `v2_rump_qd` flagship, bringing cumulative real lowering to
24/30 programs and 32/39 observations. Its 26 SSA values, 29 operations, three
ordered gates, and fuel transition 64 -> 34 must normalize to the complete
frozen `rump_build` graph. The checker independently replays all four qd128
words and requires exact receipt-v3 parity with source-fresh METRON execution.

The reconstruction claim is deliberately narrower than "the final register is
the exact result." The first gated register is exactly -2, and adding the first
two gated true qd expansions reproduces all four words of the Rump target. The
single final register reproduces the first three target words but has a zero
fourth word where the target's fourth word is nonzero. A separate exact
`Fraction` oracle checks relative error at most `2^-210` for the four-word qd
target and at most `2^-162` for the actual single final register. The receipt
records both bounds and the single-register boundary instead of hiding it. The
FULL gate includes 18 source negatives, 21 structural, all-word, resource,
observation, and canonicalization tampers, four runtime receipt-word tampers,
a causal source tamper, E2E/E2D/E2C/E2B/E2A/E1 regression, and zero diff over
the qd semantics, frozen oracle, and production codegen/ABI/runtime. Run
`scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh`.

Implementation note (2026-07-11):
`E2G-ENIR-V2-FUEL-CONTROL-FRAIL-FULL` adds `v2_fuel`, `v2_loop`, and
`v2_frail`, bringing cumulative real lowering to 27/30 programs and 35/39
observations. Profile-v2 CFG is admitted only in the canonical four-block,
four-edge schema with an entry jump, zero-branch header, body backedge, and
cost-one halt. The existing dominance, block-argument, reachability, edge
ownership, and fuel-only nontermination verifier checks remain mandatory.

The CFG interpreter now computes branch frailty from qd128 `error.x0` for the
v2 profile instead of consulting the inactive DD64 lane. The frail witness has
value-lane zero, reconstructs the exact rational value one from its four-word
qd expansion, takes the zero edge, and increments the frail count before its
gate. The independent checker parses and lowers all three sources, normalizes
the frozen EISA images, replays qd arithmetic and CFG/fuel transitions, and
requires all-word METRON receipt parity. The FULL gate adds 18 source
negatives, 22 profile/CFG/dominance/resource/all-word/canonicalization tampers,
five runtime receipt tampers, a causal source tamper that fails closed before
an unreachable declared gate, E2F through E1 regression, and zero diff over
production codegen/ABI/runtime, qd semantics, and the frozen oracle. Run
`scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh`.

Implementation note (2026-07-11):
`E2H-ENIR-V2-MEMORY-MOVE-POISON-FULL` adds `v2_mem`, `v2_emov`, and
`v2_mem_poison`, completing the bounded E2 Source-to-ENIR corpus at 30/30
programs and 39/39 observations. `estore` and `eload` copy the complete qd128
epistemic product atomically: value, four roundoff words, uncertainty, and
poison status. In this deliberately single-block memory profile, a load is
valid only after a linearly prior store to the same slot, and its provenance
names that store operation. `emov` copies the same product; source-known and
arithmetic zero values are canonicalized to positive zero before storage,
move, or gate.

The independent checker parses the three sources, normalizes the frozen EISA
register/memory images to SSA, replays qd128 arithmetic, memory, move, poison,
fuel, and observations, and requires exact source-fresh METRON parity. The FULL
gate adds 16 source negatives, 25 descriptor/provenance/dominance/all-word/
canonicalization tampers, nine runtime receipt tampers, a two-store dominance
witness, causal source tamper, E2G through E1 regression, and zero diff over
production codegen/ABI/runtime, qd semantics, and the frozen oracle. Run
`scripts/dev/madaros_v2_e2h_enir_memory_move_poison_gate.sh`.

This completes only the declared, finite `E2-ENIR-LOWERING-FULL` corpus. By
itself E2 does **not** establish arbitrary nested/path-sensitive loops, general
exceptional-value algebra, `ENIR -> MIR`, ABI lowering, GPU lowering, or
production-codegen selection. The original E2H slice rejects multi-block memory
rather than assigning an unsound path-insensitive store provenance. E3C below
adds one separately gated canonical loop-memory profile; other multi-block
memory shapes still fail closed. There is no fallback through the shadow
fixture.

Implementation note (2026-07-11):
`E3A-ENIR-MIR-QD128-ARITHMETIC-FULL` implements the first translation-validated
`ENIR -> MIR` slice for `v2_const_gate`, `v2_add`, `v2_sub`, `v2_mul`, `v2_div`,
and `v2_sqrt`. The new semantic MIR is not the target-specific x86 MachineIR.
It retains ABI-independent logical epistemic products, SSA value identity,
source provenance, observation effects, explicit poison-on-invalid trap policy,
one semantic fuel tick per instruction, and an explicit halt terminator.

Every MIR artifact binds the canonical source ENIR hash. A compiler-owned
relation checker and a separately implemented Python checker both validate the
type, value, provenance, operation, effect, trap, observation, and fuel mapping.
The MIR interpreter is separate from the ENIR interpreter; an independent
Python replay checks the same artifact, and all six logical receipts must be
bit-identical across ENIR, MIR, and source-fresh METRON execution. The FULL gate
also includes a divide-by-zero poison witness, five fail-closed out-of-scope
ENIRs, 30 structural/relational/canonicalization tampers, eleven runtime receipt
tampers, cross-name and same-name source-hash rejection, causal source tamper,
and E2H through E1
regression. Run `scripts/dev/madaros_v2_e3a_enir_mir_qd128_gate.sh`.

E3A does **not** lower memory, `emov`, profile v0/v1, fuel-only programs, or
multi-block CFG. It does not choose ABI layout, MachineIR instructions, runtime
helpers, or production codegen. Those remain later E3 slices and fail closed.

Implementation note (2026-07-11):
`E3B-ENIR-MIR-QD128-MEMORY-MOVE-FULL` extends semantic MIR with explicit
`LOAD`, `STORE`, and `MOVE` instructions for the bounded single-block v2
profile. A memory instruction carries a logical slot and a load carries the
exact latest dominating store site. Memory read and write are distinct effects;
non-memory instructions must carry no memory metadata. A store and every
load/move copy the complete qd128 epistemic product atomically: value, four
roundoff limbs, uncertainty, and poison status.

The compiler-owned verifier rejects load-before-store, stale/future store
origins, slot/effect disagreement, provenance disagreement, undefined operands,
and memory metadata on `MOVE`. The relational validator independently checks
the one-to-one ENIR mapping. A separate Python validator reconstructs that
relation, replays logical memory, and requires exact ENIR == MIR == METRON
observations for `v2_mem`, `v2_emov`, and `v2_mem_poison`. Runtime memory
receipts bind source ENIR hash, slot, MIR/source site, and all product words.

The FULL gate includes a two-store latest-dominance witness, negative-zero
store/load/move, poison preservation, 38 artifact tampers, 32 runtime receipt
tampers, cross-source and same-name hash rejection, causal source tamper, and
E3A-through-E1 regression. Run
`scripts/dev/madaros_v2_e3b_enir_mir_memory_gate.sh`.

E3B is deliberately not Memory SSA. Its verifier and translation validator
check linear latest-store provenance only for one block. Phi-like memory
versions, joins, and loops belong to the E3C schema rather than being encoded as
E3B store-site metadata.

Implementation note (2026-07-11):
`E3C-CFG-MEMORY-SSA-FULL` introduces semantic CFG-MIR schema 3 as the explicit
multi-block successor to frozen E3A/E3B schema 2. It copies four canonical ENIR
blocks, block arguments, edges, branch conditions, terminators, and semantic
ticks into machine-checkable MIR rows. `STORE` defines a memory-version SSA
value; `LOAD` consumes one. The loop header defines a memory phi whose incoming
pairs are bound to the entry edge/store version and the backedge/body-store
version.

Two source-authored witnesses make the phi non-vacuous. `v2_mem_phi_zero`
takes the zero-trip edge and observes the entry-store product (`7.25`).
`v2_mem_phi_once` executes one body store, traverses the backedge, and observes
that version (`8.5`). On each edge into the header, the interpreter checks the
concrete current version against the declared incoming version before changing
it to the phi result. The exit load consumes that phi result, while the complete
qd128 epistemic product remains atomic.

Compiler-owned structural and relational verifiers are complemented by a
separate Python relation checker and CFG/Memory-SSA replay. The three existing
E2G control programs remain exact against source-fresh METRON; the two new
memory witnesses are checked ENIR == CFG-MIR == independent replay. The FULL
gate rejects three path-initialization/source shapes, 59 artifact tampers, and
54 runtime receipt tampers, and runs E3B through E1 as regressions. Run
`scripts/dev/madaros_v2_e3c_cfg_memory_ssa_gate.sh`.

The E3A/E3B regression scripts remain strict by default. E3C invokes their
documented `E3A_ALLOW_DOWNSTREAM_ENIR_EXTENSION=1` and
`E3B_ALLOW_DOWNSTREAM_ENIR_EXTENSION=1` modes only after E3C itself has checked
that production codegen, ABI/runtime, ENIR IR/interpreter, qd semantics, and the
frozen METRON oracle are unchanged. The opt-in excludes only the E3C source
lowering extension from the historical stage-local diff check.

E3C checks behavior only for the declared reducible four-block qd128 loop and
one logical memory slot. Its schema does not admit multiple or aliased slots,
arbitrary joins, irreducible CFG, calls, exceptions, Memory SSA optimisation,
ABI selection, MachineIR, or code generation; E3D below introduces a separate
bounded multi-slot join schema rather than widening those E3C claims.

Implementation note (2026-07-12):
`E3D-MULTIPRED-SCALAR-MEMORY-SSA-FULL` adds ENIR schema 3 and Join-MIR schema 4
for one acyclic four-block diamond. The entry condition selects one of two real
predecessor blocks. Each predecessor supplies a different scalar value to one
explicit join block argument and defines two independently named memory slots.
Join-MIR represents the scalar merge as one `jsphi` relation and the memory
merge as four store versions plus two result versions and two `jmphi` rows.

The source witnesses execute opposite predecessors. `v2_join_then` selects
scalar `2`, slot products `10` and `100`, and observes `202`.
`v2_join_else` selects scalar `3`, slot products `20` and `200`, and observes
`403`. The post-join computation is `selected + 10*slot0 + slot1`; therefore a
wrong predecessor, exchanged slot, missing phi, or path-insensitive latest
store changes source-observable bits.

The compiler-owned verifier checks exact diamond topology, dominance, scalar
phi edge/value pairs, per-arm slot definitions, six memory versions, two memory
phis, load consumers, effects, ticks, and canonical serialization. A separate
Python checker parses and evaluates the source, reconstructs ENIR-to-Join-MIR,
replays qd128 execution, and requires source == ENIR == Join-MIR == independent
observables. Runtime receipts expose control choice, scalar-phi choice, both
memory-phi choices, final slot products, and final execution state. The FULL
gate rejects 12 source shapes, 72 artifact mutations, and 60 runtime-receipt
mutations; checks cross-source binding and a causal source mutation; and reruns
E3C through E1. Run
`scripts/dev/madaros_v2_e3d_multipred_scalar_memory_ssa_gate.sh`.

E3A, E3B, and E3C remain strict when invoked directly. E3D invokes their
`*_ALLOW_DOWNSTREAM_ENIR_EXTENSION=1` modes only after protecting production
codegen, ABI/runtime, frozen E3C MIR, qd128 semantics, and the METRON oracle;
the historical witnesses are then rebuilt and executed through E1.

E3D is not general SSA construction. Its implemented FULL profile is exactly
one reducible acyclic diamond, one scalar join value, and two non-aliased
logical slots. N-way joins, multiple scalar phis, nested diamonds, loops in the
same schema, alias/pointer analysis, calls, exceptions, Memory SSA
optimisation, ABI selection, MachineIR, and production code generation remain
deferred and are not implied by this gate.

## 1. Decision

Any implementation claiming Madaros v2 ENIR conformance SHALL make EISA/METRON
semantics a compiler-owned intermediate language named **EpistemicNumericIR**
(ENIR). ENIR sits after typed Semantic IR and before control-flow/ABI lowering.
The current EISA reference runtime,
Metron VM, `.eisax` format, and x86-64 bridge remain frozen oracles until a new
path proves parity. They are not the future compiler interface.

These SHALL clauses define acceptance criteria for future work; they do not
assert that a corresponding pass or checker exists today. The first
implementation milestone is not one more opaque `EReg2` helper call.
It is a complete, source-observable lowering of the existing 30-program,
39-observation corpus through ENIR, with per-stage receipts and no silent
fallback. That bounded milestone is implemented by the explicitly gated
E2A/E2B/E2C/E2D/E2E/E2F/E2G/E2H slices. E3A arithmetic, E3B bounded
single-block memory/move, the E3C canonical loop CFG/Memory-SSA, and the E3D
acyclic two-predecessor scalar/multi-slot SSA `ENIR -> MIR` slices are now
implemented. General CFG/alias coverage and every later ABI/codegen stage
remain unproven.

This document is intentionally bolder than the historical `MetronIR` sketch.
It treats numerical class, roundoff trail, epistemic uncertainty, provenance,
decision fragility, resource fuel, and policy outcome as independent semantic
axes. A backend may lower those axes into aggregates and SRET calls, but it may
not erase or conflate them before validation.

## 2. Why A Separate IR Is Necessary

The current implementation has real **runtime conformance evidence**:

- `stdlib/eisa/core.sio` implements a three-lane value for v0/v1:
  `f64 value + dd64 roundoff + f64 first-order GUM uncertainty`.
- `stdlib/eisa/core_v2.sio` deepens the roundoff lane to qd128 while preserving
  the f64 value lane and GUM uncertainty lane.
- `stdlib/eisa/evm.sio` defines execution, memory, fuel, frail-branch counting,
  poison propagation, and receipt text.
- `tools/eisa/eisa_evm_run.sio` contains 30 programs producing 39 gate/fuel
  observations.
- `scripts/ci/eisa_madaros_native_conformance_gate.sh` proves those 39 lines are
  byte-identical between Metron VM and Madaros-generated ELF, checks a source
  tamper, rejects baked output, and forbids compact/full fallback.

That evidence establishes behavior for the named corpus and executors; it does
not establish a compiler-owned semantic stage or universal preservation. Today
the meaning is
distributed among runtime structs, handwritten dispatch, format versions,
bridge templates, stdout formatting, and gates. Optimizers see opaque calls or
ordinary fields and therefore cannot answer the questions that matter:

- Is reassociation legal for this value and this uncertainty policy?
- Did a lowering preserve a qd128 correction trail or only its leading word?
- Is a branch numerically frail, epistemically unsupported, poisoned, or all
  three?
- Which source measurement and which transform contributed to an observation?
- Is a precision downgrade validated or merely cheaper?

ENIR exists to make those questions typeable, verifiable, and rejectable.

## 3. Design Principles

1. **Orthogonal semantic axes.** IEEE class, value bits, roundoff, uncertainty,
   provenance, status, gate policy, and fuel are separate fields.
2. **No premature ABI.** ENIR values are logical products. Register splitting,
   stack layout, helper calls, and SRET belong to ABI IR.
3. **Exact when exact is promised.** Bit identity is required for deterministic
   observable bits. An explicit equivalence relation is required elsewhere.
4. **Receipts are checked evidence.** A receipt without a replay, proof,
   counterexample search, or source-observable witness is metadata only.
5. **Fail closed.** Unsupported semantics produce a classified compile error;
   they never select an unreceipted fallback.
6. **Oracle independence.** The Metron VM and native executor may share the
   specification, but the conformance gate must not compare two wrappers around
   the same implementation.
7. **Learning advises.** E-KAN may rank or propose decisions. A deterministic
   validator accepts or rejects them.

## 4. Position In Madaros v2

```text
Surface AST
  | parse/name/module receipts
Typed Semantic IR
  | types, effects, numeric intent, epistemic declarations
EpistemicNumericIR (this document)
  | operational numeric/epistemic semantics and obligations
MIR
  | SSA CFG, calls, explicit traps and effects, ABI-independent values
ABI IR
  | calling convention, SRET, pack/split, target data layout
LIR / MachineIR
  | instruction selection, registers, frames, target helpers
ELF / .eisax / future Device IR
  | link/runtime contracts and execution receipts
```

ENIR is not a universal replacement for MIR. Ordinary integer and control-flow
operations may pass through it unchanged. It owns only operations whose legal
lowering depends on numerical intent, error semantics, uncertainty semantics,
provenance, or observation policy.

## 5. Core Semantic Domains

### 5.1 Numeric value

```text
NumericBits := F16Bits | BF16Bits | F32Bits | F64Bits | F128Bits
             | I<N>Bits | U<N>Bits | DecimalBits<P,E>

FPClass := PositiveZero | NegativeZero | Subnormal | Normal
         | PositiveInfinity | NegativeInfinity | QuietNaN | SignalingNaN
```

`NumericBits` is an uninterpreted fixed-width bitvector plus a declared format.
`FPClass` is derived and checked, never supplied as trusted metadata. NaN sign,
payload, and signaling/quiet state remain available even when a receipt policy
chooses a canonical display form.

### 5.2 Roundoff trail

```text
ErrorTrack := Exact
            | DD64 { hi: F64Bits, lo: F64Bits }
            | QD128 { x0: F64Bits, x1: F64Bits,
                      x2: F64Bits, x3: F64Bits }
            | StaticBound { lo: Rational, hi: Rational, proof_ref: Hash }
            | Unknown { reason: ErrorReason }
```

`DD64` and `QD128` are semantic variants, not opaque structs. Their
renormalization and non-overlap invariants are verifier obligations. The names
describe expansion depth, not IEEE binary128 storage.

For the current EISA profiles:

```text
true_dd(x) = dd(x.value) + x.error
true_qd(x) = qd(x.value) + x.error
```

The v2 closure obligation required of a future conforming profile is:

```text
t       = round_qd(true_qd(x) op true_qd(y))
value_z = round_f64(x.value op y.value)
error_z = qd_sub(t, qd(value_z))
```

Here `qd`, `round_qd`, and `qd_sub` are not free mathematical functions. Every
semantic profile binds them to a content-addressed **expansion semantics
artifact** containing:

- component format and rounding mode;
- canonical zero representation;
- the exact `two_sum`, `two_prod`, renormalization, add, subtract, multiply,
  divide, and square-root algorithms used;
- component ordering and normalization postconditions;
- exceptional-input domain;
- implementation and independent checker hashes.

For `eisa_v2_finite`, a canonical qd value is a four-component finite f64
expansion whose real interpretation is the exact sum of its components in
order. The profile requires non-increasing component magnitude and checks the
documented Hida-Li-Bailey overlap precondition/postcondition used by the local
qd implementation. It does not silently equate an arbitrary four-word tuple
with a canonical qd value. `round_qd` means the deterministic output of the
pinned algorithm artifact; any stronger correctly-rounded claim requires a
separate proof certificate. `qd_sub(a,b)` means the pinned subtraction
algorithm applied to two canonical expansions followed by pinned
renormalization.

This is therefore a finite-precision, algorithm-indexed closure contract. It is
not a proof about exact real arithmetic beyond the declared qd algorithm,
domain, and assumptions. Before an implementation gate can cite the equation,
the expansion semantics artifact and an executable independent checker must
exist. E2E supplies that pair for the six finite arithmetic witnesses through
`self-hosted/enir/qd.sio` and its independently hashed Python replay. E2F adds
the complete finite Rump graph and explicitly distinguishes exact two-gate
reconstruction from the final single-register precision boundary. E2G extends
the bounded evidence to fuel and its canonical finite CFG; E2H extends it to
the declared memory, move, negative-zero, and poison witnesses. These results
do not extend the claim to general exceptional inputs, arbitrary control flow
or memory, MIR, ABI, or arbitrary qd programs; those remain blocked rather
than provisionally redefining the equation.

### 5.3 Epistemic uncertainty

```text
Uncertainty := ExactInput
             | GUM1 {
                 standard_u: F64Bits,
                 sensitivity: ProvenanceMap,
                 covariance_model: CovarianceModel,
                 assumptions: [AssumptionId]
               }
             | Interval { lo: NumericBits, hi: NumericBits }
             | PBox { lower_cdf: ArtifactRef, upper_cdf: ArtifactRef }
             | Unknown { reason: UncertaintyReason }
```

Only `ExactInput` and the scalar independent-input `GUM1` slice are current
EISA behavior. Interval and p-box variants reserve typed extension points; they
are not implemented claims. A lowering must reject a variant it cannot
preserve. It may not coerce p-box, interval, and GUM uncertainty to one scalar.

### 5.4 Provenance

```text
Provenance := {
  value_id: StableValueId,
  source_spans: NonEmptySet<SourceSpan>,
  input_origins: Set<InputOrigin>,
  transform_chain: [TransformId],
  assumption_ids: Set<AssumptionId>,
  policy_ids: Set<PolicyId>
}
```

Provenance is content-addressed. Arithmetic unions input origins and appends
the operation transform. Loads add the memory-version origin. Phi-like joins
retain all incoming origins plus the chosen control predicate. Hashes are not
provenance by themselves; the hashed object must be canonical and retrievable.

### 5.5 Status and observations

```text
NumericStatus := Clean
               | Poisoned { reasons: NonEmptySet<PoisonReason> }
               | Trapped { reason: TrapReason }
               | Unsupported { reason: UnsupportedReason }

GateClass := Ok | Marginal | Reject | Unknown

BranchSupport := Supported | Frail { band: F64Bits }
               | UnsupportedDecision { reason: BranchReason }
```

These are deliberately distinct:

- a NaN is an IEEE class, not automatically a compiler poison;
- poison is a semantic state with one or more reasons;
- `Reject` is an observation-policy result, not a numeric class;
- a frail branch records decision sensitivity and does not automatically poison
  the value under the current count-only policy;
- a trap is observable control behavior and cannot be rewritten into poison.

## 6. ENIR Types

The canonical logical type is:

```text
!enir.number<
  value = f64,
  error = exact | dd64 | qd128 | static_bound,
  uncertainty = exact | gum1 | interval | pbox,
  status = tracked,
  provenance = tracked,
  profile = eisa_v0 | eisa_v1 | eisa_v2 | general
>
```

Additional first-class types:

```text
!enir.error.dd64
!enir.error.qd128
!enir.uncertainty.gum1
!enir.fpclass
!enir.status
!enir.gate_result
!enir.branch_support
!enir.provenance
!enir.fuel
!enir.memory<Token, NumberType>
```

Every type has a canonical text and binary serialization. Stable IDs derive
from semantic content and predecessor IDs, not allocation order or addresses.

The ENIR verifier enforces operation-specific well-formedness. In particular,
`ebrz` and `ebrn` are well typed only when the selected profile defines a total
decision policy for the operand's uncertainty variant. Merely admitting
`interval` or `pbox` in `!enir.number` does not make such a value branchable.

## 7. Operations

### 7.1 Construction and classification

```text
%x = enir.econst bits(...) : !enir.number<...>
%class = enir.classify %x : !enir.fpclass
%x2 = enir.attach_uncertainty %x, %u
```

`econst` preserves exact source bits and spelling provenance. Decimal literals
carry both the source decimal and the selected correctly-rounded target bits.
The decimal-to-binary proof obligation belongs to the preceding Numeric IR
boundary and is referenced by ENIR.

### 7.2 Arithmetic

```text
%z = enir.eadd  %x, %y {rounding = nearest_even, profile = eisa_v2}
%z = enir.esub  %x, %y {rounding = nearest_even, profile = eisa_v2}
%z = enir.emul  %x, %y {rounding = nearest_even, profile = eisa_v2}
%z = enir.ediv  %x, %y {rounding = nearest_even, profile = eisa_v2}
%z = enir.esqrt %x     {rounding = nearest_even, profile = eisa_v2}
```

Each operation is a product transition:

```text
(value, class, error, uncertainty, status, provenance)
  -> (value', class', error', uncertainty', status', provenance')
```

The transition names all assumptions: IEEE rounding mode, FMA contraction,
subnormal mode, qd/dd algorithm version, covariance policy, and exceptional
case policy. Reassociation, contraction, and precision changes are illegal
unless a validator proves the whole product transition equivalent.

The current `eisa_v2` finite profile requires round-to-nearest-even, strict
subnormals, no FMA contraction, and no reassociation. The `eisa_v0/v1` profile
retains the documented first-order dd64 formulas and does not acquire stronger
claims by passing through ENIR.

### 7.3 Memory and movement

```text
%m1 = enir.estore %m0[%slot], %x
%x, %m2 = enir.eload %m1[%slot]
%y = enir.emov %x
```

Memory stores the entire logical product atomically. A backend split into
multiple words must prove there is no mixed-version read. `emov` preserves
bits, class, all error words, uncertainty, status, and provenance except for an
appended movement transform. Poison reasons may accumulate but never disappear.

### 7.4 Branches

```text
enir.ebr ^target
enir.ebrz %x, ^zero, ^nonzero {policy = count_only}
enir.ebrn %x, ^negative, ^nonnegative {policy = count_only}
```

For current EISA with `Uncertainty = ExactInput | GUM1`:

```text
u_band(x) = 0                                  if ExactInput
          = x.uncertainty.standard_u           if GUM1
band(x)   = max(u_band(x), abs(leading(x.error)))
frail(x) = band(x) != +0 and abs(x.value) <= band(x)
```

The branch decision still uses the value-lane f64 bits. The branch event also
records `BranchSupport`, predicate bits, chosen edge, and policy. Exact zero
with zero band is not frail. The current policy increments a counter only.
For `Interval`, `PBox`, or `Unknown` uncertainty, this scalar predicate is not
defined. The branch yields `UnsupportedDecision` and no successor edge unless
the profile names a separately validated decision policy for that uncertainty
variant. Future warn/reject/poison policies require distinct IDs and cannot
silently change existing images.

### 7.5 Gates and termination

```text
%g = enir.egate %x {policy = @eisa_10_100_v1}
enir.ehalt
enir.fuel_tick %fuel {cost = semantic_instruction}
enir.fuel_stop %last_value
```

Gate policy is data, not hardcoded control flow. The current policy computes
roundoff from the leading error component and compares it to scalar GUM
uncertainty at 10x and 100x thresholds. Its policy artifact states that choice
explicitly; it does not imply that lower qd components are semantically absent.

Fuel counts ENIR semantic instructions, not host cycles. Lowering one ENIR op
to many MIR operations must preserve one semantic tick. A scheduler or gas
model may define a separate target-cost metric keyed by profile and target.

## 8. Exceptional-Value Algebra

Current EISA v2 validates finite constants and turns computed non-finite lanes
into poison. ENIR v1 SHALL make the missing general algebra explicit instead
of pretending it already exists.

For every operation and profile, an exceptional-case table must classify:

- positive and negative zero;
- smallest/largest subnormal and normal values;
- positive and negative infinity;
- quiet NaN and signaling NaN, including payload policy;
- invalid, divide-by-zero, overflow, underflow, and inexact events;
- propagation into error and uncertainty lanes;
- trap versus status versus poison behavior.

The first FULL implementation may select `profile=eisa_v2_finite` and reject
source programs requiring the general table. It may not claim general NaN/Inf
semantics until all rows are implemented and gated. Negative tests must prove
that unsupported cases fail at the ENIR boundary and emit no ELF.

## 9. Static And Dynamic Receipts

One receipt format cannot honestly prove both compilation and execution. ENIR
therefore defines two linked artifacts.

### 9.1 Lowering receipt

```json
{
  "schema": "madaros.enir.lowering_receipt/0.1",
  "stage_from": "semantic_ir",
  "stage_to": "enir",
  "input_artifact_sha256": "...",
  "output_artifact_sha256": "...",
  "semantic_profile_sha256": "...",
  "operations_root_sha256": "...",
  "operation_count": 0,
  "accepted": true,
  "validator": {
    "kind": "replay|smt|proof_checked|differential",
    "version_sha256": "...",
    "obligations_root_sha256": "...",
    "result_sha256": "..."
  },
  "fallback": "none",
  "compiler_sha256": "..."
}
```

Each operation leaf commits to stable op ID, source span, operand/result type,
semantic profile, provenance transition, selected lowering, assumptions,
validator result, and predecessor/successor hashes.

### 9.2 Execution receipt

```json
{
  "schema": "madaros.enir.execution_receipt/0.1",
  "program_sha256": "...",
  "enir_sha256": "...",
  "executor_sha256": "...",
  "input_sha256": "...",
  "event_trace_root_sha256": "...",
  "event_count": 0,
  "gate_events": [],
  "final_state_commitment_sha256": "...",
  "stop": "halt|fuel|trap",
  "anti_vacuity_nonce_sha256": "..."
}
```

Per-operation dynamic events form a canonical Merkle trace. Normal CI stores
the root plus all gate events. A failing or sampled run carries inclusion
proofs for relevant operations. This gives per-operation accountability
without forcing every release receipt to contain a huge plaintext trace.

### 9.3 Evidence levels

```text
R0 structural: schemas parse and hashes link
R1 replay: an independent interpreter reproduces the concrete transition
R2 translation validation: solver or exhaustive finite check proves refinement
R3 proof checked: a small checker validates a formal certificate
```

R0 alone never authorizes a lowering. Current concrete corpus parity is R1
evidence. Bounded SMT may reach R2 for selected operations. R3 is reserved for
small stable kernels such as class decoding, dd/qd renormalization invariants,
and ABI pack/unpack lemmas where proof cost is justified.

## 10. Translation-Validation Relation

For adjacent stages `A -> B`, validation checks refinement, not file equality:

```text
for every admitted input state s:
  observe_A(exec_A(A, s)) == observe_B(exec_B(B, encode(s)))
```

The canonical observation signature is:

```text
Observation := {
  termination: Halt | Fuel | Trap(code),
  stdout: Bytes,
  values: [(StableObservableId, NumericBits, FPClass,
            ErrorTrack, Uncertainty, NumericStatus)],
  gates: [(GateEventId, StableObservableId, GateClass, PolicyId)],
  branches: [(BranchEventId, PredicateBits, BranchSupport, EdgeId)],
  semantic_fuel_consumed: U64,
  provenance: [(StableObservableId, CanonicalProvenanceHash)]
}
```

Lists are ordered by semantic event sequence; maps inside their elements use
canonical key order. Stage receipts carry a total `StableObservableId` map from
predecessor to successor. Provenance equivalence means equality after replacing
successor IDs with predecessor IDs through that map and canonicalizing sets; a
missing, duplicate, or non-total map is a validator failure. Internal values
not named by the stage's observability manifest are existentially hidden, but
their effects on every listed observation remain checked.

For every value listed in the observability manifest, **all** DD64/QD128 words
are observable and bit identity is required under the deterministic EISA CPU
profile. An internal value may be hidden only if the validator proves that no
listed value, gate, branch, status, fuel event, or I/O depends on an erased
distinction. Representation hiding never authorizes dropping a lower expansion
word that can influence a later observation.

`observe` includes:

- value bits and IEEE class;
- all dd64/qd128 words;
- uncertainty representation and declared assumptions;
- status, trap, poison reasons, and gate class;
- chosen control edge and frail count;
- semantic fuel consumed and stop reason;
- source-observable I/O and gate receipts;
- provenance commitment modulo the stage's declared ID mapping.

The serialized signature above is the R1 contract. An R2 validator must encode
the same fields and ID map in its solver theory; fields outside its supported
theory force `unknown`, not omission. Three verdicts are legal: `proved`,
`refuted(counterexample)`, and `unknown(reason)`. Only `proved` permits an optimized/lowered artifact into a
FULL gate. `unknown` is not success and selects no fallback.

### 10.1 Bit identity versus semantic equivalence

Bit identity is mandatory when both sides expose the same deterministic
representation: f64 bits, dd/qd words, scalar GUM bits, gate events, fuel, and
receipt canonical bytes under the EISA CPU profile.

Semantic equivalence is used for hidden representation choices such as SRET
layout, register allocation, instruction addresses, and object-file metadata.
Those choices must decode to bit-identical logical ENIR observations. GPU
profiles may use a separately named relation only when hardware behavior makes
CPU bit identity impossible; the relaxation, domain, and error bound are part
of the receipt and never inherited implicitly.

## 11. Stage Obligations

### Surface -> Semantic IR

- preserve literal spelling, source bits, units, effects, and epistemic intent;
- resolve `Knowledge`, GUM, interval, and EISA declarations without flattening;
- classify unsupported semantics before optimization.

### Semantic IR -> ENIR

- select an explicit semantic profile;
- materialize value/error/uncertainty/status/provenance axes;
- lower gates, branch policies, and fuel as operations;
- emit a deterministic ENIR artifact and R1-or-better receipt.

### ENIR -> MIR

- make CFG, calls, traps, and memory effects explicit;
- preserve one semantic fuel tick per ENIR instruction;
- keep logical products ABI-independent;
- validate every rewrite that changes operation order or precision.

### MIR -> ABI IR

- choose direct, split, by-reference, or SRET transport explicitly;
- specify field order, alignment, register/stack classes, and ownership;
- prove pack/unpack roundtrips for dd64, qd128, status, and provenance handles;
- forbid metadata constants from overwriting computed return values.

### ABI IR -> LIR/MachineIR -> ELF

- use strict target floating-point mode required by the profile;
- identify helper symbols and versions;
- retain op-to-PC mapping for counterexample localization;
- link runtime/helper hashes into the lowering receipt;
- produce no ELF on unresolved obligation or unclassified fallback.

## 12. The 39-Observation Oracle Matrix

The current mandatory corpus has 30 programs and 39 receipt observations.
The future manifest SHALL assign stable IDs to every row and observation; a
single aggregate `39/39` counter is insufficient.

| Program ID | Observations | Primary obligation |
|---|---:|---|
| `golden_mul` | 1 | Metron source, multiply, gate, store |
| `golden_add` | 1 | Metron source, add, gate, store |
| `golden_sqrt` | 1 | Metron source, sqrt, gate |
| `golden_poison` | 1 | divide-by-zero poison propagation |
| `e5_cancellation` | 2 | pre/post catastrophic cancellation |
| `v1_loop` | 1 | zero branch and halt |
| `v1_if_both` | 2 | negative branch and both paths |
| `v1_i6` | 1 | poisoned branch does not redirect unsafely |
| `v1_fuel` | 1 | dynamic fuel-stop receipt |
| `v1_highreg` | 1 | high-register state |
| `v1e_fixedpoint` | 1 | loop convergence and move |
| `v1e_frail` | 1 | dd64 frail boundary count |
| `v1e_emov_negzero` | 2 | move and negative-zero behavior |
| `v1_arith_high` | 1 | full arithmetic in high registers |
| `v1_fuel_high` | 1 | fuel stop with high last-write register |
| `v1_branch_high` | 2 | high-register zero/negative branches |
| `v2_const_gate` | 1 | minimal qd128 profile and receipt v3 |
| `v2_add` | 1 | qd128 add closure |
| `v2_sub` | 1 | qd128 subtract closure |
| `v2_mul` | 1 | qd128 multiply closure |
| `v2_div` | 1 | qd128 divide closure |
| `v2_sqrt` | 1 | qd128 sqrt closure |
| `v2_rump_qd` | 3 | qd128 Rump reconstruction success |
| `v1_rump_dd` | 3 | mandatory dd64 precision failure witness |
| `v2_fuel` | 1 | qd128-profile fuel-stop receipt |
| `v2_mem` | 1 | qd128 atomic store/load |
| `v2_emov` | 2 | deep-lane move and negative zero |
| `v2_loop` | 1 | qd128 zero branch |
| `v2_frail` | 1 | qd128 frail boundary count |
| `v2_mem_poison` | 1 | poison through store/load/move |
| **Total** | **39** | **all observations named and non-vacuous** |

The first ENIR gate runs each row through:

```text
Metron VM oracle
ENIR interpreter
ENIR -> MIR interpreter
Madaros x86-64 ELF
```

All four must agree under the declared observation relation. The existing
Metron VM/ELF equality remains necessary but no longer sufficient.

## 13. FULL Gate Requirements

The first S-step is `E0-ENIR-SPEC-FULL`; the first implementation step is
`E1-ENIR-CORPUS-FULL`.

`E1-ENIR-CORPUS-FULL` is complete only when all are true:

1. The 30 programs lower through real compiler-owned ENIR, not a table or
   fixture-specific emitter.
2. All 39 named observations match at every executable boundary.
3. Positive coverage includes every ENIR opcode and all v0/v1/v2 profiles used
   by the corpus.
4. Negative coverage rejects malformed types, unsupported profiles, invalid
   provenance, missing policies, unresolved helper symbols, and invalid ABI.
5. Edge coverage includes signed zero, min/max subnormal, min/max normal,
   overflow, underflow, Inf, qNaN, and sNaN. Unsupported rows must be explicit
   compile-time rejections until their algebra is implemented.
6. ABI coverage exercises direct values, dd64/qd128 SRET, imported calls,
   forwarding, memory roundtrip, and caller/callee disagreement negatives.
7. A source tamper changes exactly the data-dependent observations predicted by
   a dependency slice; "at least one line changed" is only the bootstrap rule.
8. Anti-vacuity proves dynamic receipt payloads and expected numeric digit runs
   are not embedded as constants in generated ELF.
9. Compact/legacy/fallback routes are absent on success and fail closed when
   explicitly selected but incapable.
10. Lowering and execution receipts reach R1 for every op; selected arithmetic
    and ABI rules reach R2 or document `unknown` and remain unselected.
11. The aggregate gate checks exact manifest equality, not only lower bounds:
    no missing, duplicate, reordered, or unexpected observation IDs.
12. The current oracle/runtime stays intact until two consecutive canon
    releases pass the full differential gate from source-fresh compilers.

## 14. Tamper And Anti-Theater Rules

A receipt proves behavior only if its claims are causally tied to execution.
The gate SHALL include:

- source-literal tamper with predicted dependency-slice changes;
- opcode tamper and policy-ID tamper;
- one receipt-byte tamper rejected by hash/checker;
- one lowering-artifact tamper rejected before execution;
- a dead-code tamper that changes artifact hashes but not observations;
- an ABI field-order tamper that produces a validator counterexample;
- an oracle-diversity check proving ENIR and Metron execution do not call the
  same semantic implementation;
- exact accepted/rejected/unknown counts and IDs.

Static strings, case matrices, or `implemented=true` fields never satisfy a
behavioral gate. CI consumes receipts only after independently recomputing the
relevant hashes and validation verdicts.

## 15. E-KAN Integration

E-KAN enters above ENIR as a proposal engine and below no semantic authority.
Its strongest defensible first role is **precision and validation scheduling**:

```text
features: ENIR op graph, dynamic trace summaries, cancellation indicators,
          provenance depth, gate margin, target, historical validator cost
outputs:  ranked precision profile, likely frail sites, validator budget,
          candidate lowering/e-graph rule
```

Each proposal includes model hash, training-corpus hash, feature schema,
calibration data, confidence/uncertainty, chosen action, exact fallback, and
validator result. The deterministic validator disposes:

```text
E-KAN proposes -> translation validation proves/refutes/returns unknown
              -> only proved candidates may be selected
```

### Falsifiable E-KAN experiment

Hypothesis: under a fixed validation budget, an E-KAN scheduler finds more
valid dd64 downgrades or qd128-required sites than random search, a linear
model, a tree model, and a hand heuristic, without increasing unsound accepts.

Primary metrics:

- validator-confirmed decisions per CPU-hour;
- false-accept count, required to remain zero;
- false-downgrade count on adversarial cancellation cases;
- calibration error for predicted validation success;
- held-out target and held-out kernel generalization;
- compile-time and runtime cost.

An E-KAN that does not beat simpler baselines is rejected. Interpretability is
evaluated by stable symbolic feature selection, not by architecture name.

## 16. Novelty Thesis And Boundaries

The publication hypothesis is the conjunction below. Its empirical increment,
before any stronger proof claim, is: **a compiler-owned product IR with
separate deterministic roundoff and epistemic lanes, causally linked static and
dynamic receipts, and validator-gated learned proposals**. The remaining items
state the system/evaluation required to test whether that increment is useful
and distinct, rather than six already-established contributions:

1. a compiler IR that carries a deterministic per-operation roundoff trail and
   a separate epistemic uncertainty model;
2. provenance- and policy-aware control decisions as compilable semantics;
3. proof/receipt-carrying lowering across typed IR, MIR, ABI, and executable;
4. a committed per-operation execution trace linked to stage receipts;
5. learned E-KAN proposals that cannot bypass deterministic translation
   validation;
6. self-host migration with the old VM/runtime retained as an independent
   differential oracle.

This is a research hypothesis, not a priority claim or a promise that receipts
alone make an unverified IR novel. Evaluation must show that ENIR catches
miscompilations or unsafe numerical transforms missed by type/ABI tests, that
its receipts localize them, and that the cost is acceptable. MLIR already establishes
multi-level extensible IR; Alive2 establishes practical bounded translation
validation; CompCert establishes a much stronger verified-compilation bar;
Herbie, Daisy, and FPTaylor cover important floating-point optimization/error
territory; egg establishes practical equality saturation; MLGO establishes
learned compiler heuristics; probabilistic languages formalize other kinds of
uncertainty. Madaros novelty must survive comparison with the conjunction of
those systems, not merely with conventional LLVM IR.

Explicit non-claims:

- ENIR is not formally verified end to end.
- Corpus equality is not universal semantic preservation.
- qd128 is not IEEE binary128 and is not exact real arithmetic.
- current scalar GUM propagation is not arbitrary covariance support.
- p-box and interval ENIR variants are reserved, not implemented.
- E-KAN is not trusted and is not known to outperform simpler models.
- current EISA has no complete NaN/Inf/subnormal algebra.
- receipts are not proofs unless a named checker validates their obligation.

## 17. Migration And Bootstrap Safety

### Phase A: freeze and specify

- Pin hashes of current Metron VM, `.eisax` format, corpus, and native gate.
- Version the semantic profiles and observation relation.
- Land this specification without compiler code.

### Phase B: shadow ENIR

- Emit ENIR sidecars after Semantic IR while continuing the current path.
- Parse/print/roundtrip ENIR deterministically.
- Compare static operation manifests; do not affect generated code.

### Phase C: independent ENIR interpreter

- Execute the full corpus in an interpreter that does not call EISA runtime
  arithmetic functions.
- Reach 39/39 four-way parity with tamper and anti-vacuity.

### Phase D: ENIR -> MIR -> ABI

- Lower one complete semantic profile, including memory, control, fuel, gates,
  status, and SRET.
- Run per-pass translation validation and preserve the current route as oracle.

### Phase E: guarded default

- Select ENIR by default only after full source-fresh gates pass on canon.
- Keep an explicit oracle comparison mode; no silent fallback.
- Roll back by selecting the previous receipt-pinned compiler artifact, not by
  deleting the new IR or weakening gates.

### Phase F: oracle retirement review

- Consider retiring duplicate production execution only after two consecutive
  canon releases, cross-target evidence, fuzzing, and an external semantic
  review.
- Preserve the reference interpreter and corpus permanently as test oracles.

Bootstrap generations record compiler hash, input source hash, every stage
artifact hash, validator hash, and final ELF hash. Bit-identical fixed point is
required only where deterministic serialization and target configuration make
it meaningful. Else the declared translation relation must pass at each edge.

## 18. Trust Model

The trusted computing base initially includes:

- ENIR semantic profile definitions;
- canonical serializers/hash implementation;
- receipt checker and translation validators;
- SMT solver result checking path, when used;
- target execution model and runtime helper specifications;
- assembler/linker and platform ABI assumptions;
- the oracle implementations used for differential evidence.

The compiler pass, E-KAN model, optimizer, and receipt producer are not trusted.
Long-term work should shrink the TCB by proof-checking certificates and using a
small independent receipt checker. A self-reported compiler receipt checked by
the same compiler instance is insufficient on its own.

## 19. Merge Gates And Governance

The protected `canon/madaros-v2-sota` ruleset requires PRs and the following
checks before update:

- `Full Test Suite`
- `Lean Proofs`
- `Contracts`
- `Native Self-Host (Linux x86_64)`
- `Native Self-Host (macOS arm64)`
- `Source-Bootstrap Self-Host (Linux x86_64)`

Future ENIR implementation PRs additionally require:

- ENIR parse/print/roundtrip gate;
- exact 30-program/39-observation manifest gate;
- per-stage translation-validation gate;
- exceptional-value negative matrix;
- ABI/SRET conformance matrix;
- tamper, anti-vacuity, and no-fallback gates;
- math review for numeric/GUM semantics;
- external artifact review before any paper or novelty claim.

## 20. Open Questions

1. Is the normative value lane always source-profile IEEE, or may a future
   profile promote it while retaining a separately observed hardware lane?
2. Should covariance sensitivity vectors be explicit SSA values or immutable
   references to a side table?
3. Which exceptional-value rows should EISA general profile define first?
4. What is the smallest independently implementable ENIR interpreter that does
   not share qd/dd code with the oracle while still satisfying the mandatory
   independence criterion?
5. Can an SMT model scale from per-op floating-point refinement to bounded CFGs
   with fuel, poison, and frail branch events?
6. Which provenance transforms commute, and which must preserve exact order?
7. What GPU observation relation is strong enough to be meaningful without
   falsely demanding CPU bit identity?
8. Which receipt certificates should be proof-checked rather than solver-trusted?
9. Does the 39-observation corpus contain enough independent numerical regimes,
   or must a generated adversarial suite be mandatory before E1 completion?
10. Does E-KAN outperform simpler calibrated models on the precision scheduler?

## 21. Literature Anchors

These are design constraints and comparison points, not claims of equivalence.

- Chris Lattner et al., **MLIR: Scaling Compiler Infrastructure for Domain
  Specific Computation** (2021). Reusable idea: typed dialects, explicit
  lowering boundaries, verifiers, and roundtrippable IR. Limitation for
  Madaros: MLIR does not supply EISA's numerical/epistemic semantics or
  receipts. <https://arxiv.org/abs/2002.11054>
- Nuno P. Lopes et al., **Alive2: Bounded Translation Validation for LLVM**
  (PLDI 2021). Reusable idea: automatic refinement checking with concrete
  counterexamples and explicit boundedness. Limitation: LLVM-focused,
  intraprocedural, and not an epistemic execution semantics.
  <https://doi.org/10.1145/3453483.3454030>
- Xavier Leroy et al., **CompCert** (2009 onward). Reusable idea: semantic
  preservation per pass and small verified validators. Limitation: full proof
  cost is incompatible with the first Madaros v2 migration; ENIR begins with
  pragmatic translation validation. <https://compcert.org/>
- Max Willsey et al., **egg: Fast and Extensible Equality Saturation** (POPL
  2021). Reusable idea: e-class analyses and non-destructive rewrite search.
  Limitation: equality saturation needs ENIR-specific floating-point,
  uncertainty, effect, and provenance legality. <https://arxiv.org/abs/2004.03082>
- Eva Darulova et al., **Daisy: Framework for Analysis and Optimization of
  Numerical Programs** (TACAS 2018), and Darulova, Horn, Sharma, **Sound
  Mixed-Precision Optimization with Rewriting** (2018). Reusable idea: sound
  error bounds and precision tuning. Limitation: static bounds are not EISA's
  deterministic per-execution correction trail.
  <https://doi.org/10.1007/978-3-319-89960-2_15>
- Alexey Solovyev et al., **Rigorous Estimation of Floating-Point Round-off
  Errors with Symbolic Taylor Expansions** (TOPLAS 2018). Reusable idea:
  rigorous a priori bounds. Limitation: straight-line/domain analysis does not
  replace dynamic provenance-linked observations.
  <https://doi.org/10.1145/3230733>
- Pavel Panchekha et al., **Automatically Improving Accuracy for Floating Point
  Expressions** (PLDI 2015). Reusable idea: counterexample-rich numerical
  rewrite search. Limitation: sampled improvement is not a proof and cannot
  authorize an ENIR rewrite alone. <https://doi.org/10.1145/2737924.2737959>
- Mircea Trofin et al., **MLGO: a Machine Learning Guided Compiler
  Optimizations Framework** (2021). Reusable idea: learned models replacing
  compiler heuristics with corpus/evaluation discipline. Limitation: a learned
  policy is not semantic authority. <https://arxiv.org/abs/2101.04808>
- Ziming Liu et al., **KAN 2.0: Kolmogorov-Arnold Networks Meet Science**
  (2024). Reusable idea: interpretable feature, module, and symbolic-structure
  discovery. Limitation: it does not establish KAN superiority as a compiler
  cost model or validator. <https://arxiv.org/abs/2408.10205>
- Kangfeng Ye, Jim Woodcock, Simon Foster, **Probabilistic Unifying Relations
  for Modelling Epistemic and Aleatoric Uncertainty** (TCS 2024). Reusable idea:
  distinguish uncertainty kinds and give them formal semantics. Limitation:
  probabilistic program semantics is not the same object as EISA's GUM and
  deterministic roundoff lanes. <https://doi.org/10.1016/j.tcs.2024.114876>

## 22. Immediate Next Action

After this specification passes external math/semantic review, implement only
the canonical ENIR data model, parser/printer, verifier, and deterministic hash
in a new lane. Do not yet switch code generation. The acceptance gate for that
next action is byte-identical ENIR roundtrip plus an exact static manifest for
the 30-program/39-observation corpus.
