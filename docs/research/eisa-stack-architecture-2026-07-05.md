<!-- docs:meta
topic_id: repo.docs.research.eisa-stack-architecture-2026-07-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.eisa-stack-architecture-2026-07-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# EISA stack architecture — the epistemic executable format (2026-07-05)

Status: operator-approved direction (2026-07-05): EISA is a **self-contained
executable stack** — its own binary format and virtual machine — independent
of ELF/x86. The classical native path becomes a *conformance bridge*, not
the primary target.

Supersedes: `docs/research/eisa-v0-spec-2026-07-05.md` (v0 spec). Carried
over from v0 unchanged: the three-lane register semantics (reference
implementation `stdlib/eisa/core.sio`), lane invariants I1–I5, the 10x/100x
gate policy and its GUM rationale, the textual `.eisa` grammar (§3 of the
v0 spec, still versioned and non-frozen). Replaced: the phasing (v0.1–v3)
and the receipt schema (extended with program provenance, below).

## 1. Why an own executable format

The v0 plan treated EISA as a layer over the existing pipeline: expand,
then emit ELF. The operator's thesis is stronger — novel medical/complex
modelling needs a novel stack, not annotations on the old one. Concretely,
an own format buys three things ELF cannot give:

1. **Provenance is part of the executable.** A `.eisax` binary carries the
   hash of its own code and constants; every receipt cites it. The v0
   design review (xai, 2026-07-05) flagged receipts without program
   identity as audit-trail theatre — in ELF there is nowhere honest to put
   it; in our own format it is a mandatory section.
2. **The machine model is epistemic by construction.** The EVM has no way
   to execute an arithmetic instruction without propagating all three
   lanes (value, measured roundoff, GUM uncertainty) and the poison rules.
   There is no "fast path that forgets the error" to fall back into.
3. **Conformance becomes symmetric.** Reference semantics (`eisa::core`),
   EVM, and the x86 bridge are three executors of the same format, checked
   bit-exactly against each other. x86 is demoted from "the meaning of the
   program" to "one more executor that must agree".

## 2. Stack overview

```
.sio epistemic regions ──(E3 backend)──┐
                                       ├──> .eisax ──> EVM (primary executor, receipts)
.eisa low-level text ──(E1 assembler)──┘        │
                                                └──(E4 bridge)──> x86-64 (conformance lane)
```

Components, with repository homes:

| Component | Home | Phase |
|---|---|---|
| `.eisax` container encode/decode | `stdlib/eisa/format.sio` | E1 |
| `.eisa` assembler (text → container) | `stdlib/eisa/asm.sio` | E1 |
| EVM loader + executor | `stdlib/eisa/evm.sio` | E2 |
| Reference semantics (unchanged) | `stdlib/eisa/core.sio` | done |
| Sounio → `.eisax` backend | new files under `self-hosted/` | E3 |
| `.eisax` → x86-64 bridge | new files under `self-hosted/` | E4 |

## 3. The `.eisax` container, version 0

Design constraints, learned the hard way today: the checker hangs on
arrays of nested structs and struct-array parameters, so the container is
defined over **flat parallel arrays**; and pure Sounio has no `f64`
bit-cast intrinsic, so the container keeps integers and floats in
separate planes rather than punning them.

A `.eisax` v0 image is two planes:

- **W-plane** `[i64]` — header, code, provenance (all integers)
- **C-plane** `[f64]` — constant pool (all floats)

This dual-plane layout is a deliberate format property, not an
implementation dodge: it means the container never needs to reinterpret
bits between integer and float, so a pure-Sounio reader is exact by
construction. (A single-byte-stream profile can be added later for
interchange; it requires an f64 bit-cast primitive first.)

### 3.1 W-plane layout

```
word 0   magic       0x45495341  ("EISA" in ASCII, as an i64)
word 1   version     0
word 2   n_code      number of instructions
word 3   n_const     number of C-plane entries
word 4   n_prov      number of provenance words (>= 2)
word 5   prog_hash   EISA-hash v0 over words 0..4 + code + provenance + C-plane (see 3.4)
word 6.. code        4 words per instruction: op, dst, a, b
...      provenance  n_prov words: [0] source id, [1] flags, rest reserved
```

Offsets are derived from the counts (code starts at word 6, provenance at
6 + 4·n_code); v0 deliberately has exactly three sections in fixed order.
Review finding (xai, MED): count-derived layout does not extend to new
sections without a version bump. Accepted as a v0 property — the version
word exists precisely so v1 can move to offset-table headers if debug
info or signatures are added; v0 refuses any trailing words beyond the
declared counts (validation below), so there is no gray zone.

Instruction operands follow the v0 spec opcodes (0=econst … 8=estore) with
one container-level change: **`econst` takes a C-plane index in field
`a`**, not an inline immediate — the code section is integer-only.
Opcodes 9..15 are **reserved**: the validator rejects them in v0, which
is the extension mechanism — a future version can assign them without
any v0 container becoming ambiguous.

### 3.1a Machine initialisation and entry (review finding, HIGH)

Two conforming loaders must be indistinguishable, so initial state is
part of the format, not the implementation:

- entry point is instruction 0; execution is straight-line to the last
  instruction (no control flow in v0 — unchanged from the v0 spec)
- all 16 registers start `val = 0.0, err = (0,0), u = 0.0, poison = 0`
- all 8 memory slots start the same way; a program that wants measured
  inputs materialises them through `econst`/arith or a host pre-load
  that is itself part of the witness setup, never ambient state
- termination status: 0 if all instructions executed, 1 if the machine
  halted on a malformed step (unreachable for a validated container,
  kept for the differential harness that drives `eisa_step` raw)

### 3.1b Observable-output model (review finding, HIGH)

The EVM's only externally observable effects, in order, are: (1) the
receipt lines emitted by `egate`, on stdout, in execution order; (2) the
termination status; (3) the final machine state as exposed by accessor
functions to the embedding program. There is no other I/O in v0 — no
loads from ambient files, no clock, no randomness. "Bit-exact agreement"
between executors is defined over exactly these three observables plus
the lane values of every register and memory slot at termination. Real
programs needing input feed it through the C-plane (constants) or
host-prepared memory as part of the harness; interactive I/O is a
format-version decision deferred past E5, not an implementation detail.

**Zero-sign caveat (E3 review finding, xai 2026-07-05):** the sign of a
floating-point zero is *not* part of the v0 observable surface. The
receipt decomposition maps both `+0.0` and `-0.0` to `s0e0m0`, and the
E3 backend's move synthesis (`eadd(x, Z)` with an exact-zero register —
v0 has no move opcode) canonicalises `-0.0` to `+0.0` per IEEE addition.
A program that routes a computed `-0.0` through a move and then divides
by it can observe the flip (`1/x` changes infinity sign); such programs
are outside the v0 conformance corpus, and a dedicated `emov` opcode
(reserved space 9..15) is the v1 fix if zero-sign ever becomes
scientifically load-bearing. Recorded rather than silently ignored.

### 3.2 Validation rules (loader MUST enforce, malformed = refuse to run)

- magic and version match; counts non-negative and consistent with the
  actual plane lengths; **no trailing words** beyond the declared
  sections (exact-length rule — nothing executable or hashable can hide
  past the counts)
- every opcode known and not reserved; every register index in [0,16);
  every memory slot in [0,8); every C-plane index in [0, n_const);
  every non-operand field exactly 0 (e.g. field `b` of `esqrt`) — no
  don't-care bits, so containers are canonical and hash-comparable
- every C-plane constant finite (I-invariant heritage: constants are
  exact finites; NaN/Inf enter execution only as *computed* poison)
- prog_hash recomputes exactly over header words 0..4, code, provenance,
  and the C-plane (review finding, MED: provenance inside the hash, so
  program identity covers the claimed source id and flags too; the hash
  word itself is the only word excluded)

Refusal is loud (loader returns malformed, no partial execution) —
mirror of I1's reject-never-clamp.

### 3.3 Receipts, version 1

As built in E2 (`stdlib/eisa/evm.sio`, normative since 2026-07-05):

```
eisa-receipt: v=1 prog=<hash-dec> gate=<counter> reg=e<r>
  val=s<s>e<e>m<m> roundoff=s<s>e<e>m<m> u=s<s>e<e>m<m> poisoned=<0|1>
```

(one line; wrapped here). Changes from v0: `v=1`, and `prog=` — the
program hash from the container, closing the provenance finding. Lane
values stay as sign/exponent/mantissa integer decompositions (bit-exact,
no decimal floats; NaN canonicalises to `s0e2047m1`, see the lean_single
NaN-semantics audit). `gate=` is the dynamic gate counter (the v1a
`site=#n` role); the gated register and its `val` lane are printed so a
receipt alone pins the observable value. The v0 draft's `gate=<code>`,
`policy=` and `cov=` fields are deferred: gate outcome is recoverable
from `poisoned=` plus the lanes under the fixed 10x/100x policy, and a
per-receipt policy field returns when policies become per-site (post-E5).
The E4 bridge must reproduce this line byte-identically.

### 3.4 EISA-hash v0

Adler-style over the code words and C-plane, chosen because it is exact
in i64 arithmetic with no overflow and no bit-cast:

```
M = 2147483647            (2^31 - 1)
a = 1; b = 0
input order:  header words 0..4, then code words in layout order,
              then provenance words, then C-plane constants in order
for each i64 word w:          mix(w)   (words are validated non-negative)
for each constant c:          decompose to (sign, exp, mant) and mix
                              sign, exp, mant_lo, mant_hi IN THIS ORDER,
                              mant_lo = mant mod 2^26,
                              mant_hi = mant / 2^26  (integer division)
mix(x): a = (a + (x mod M)) mod M;  b = (b + a) mod M
hash = b * 2147483648 + a     (fits in 62 bits, non-negative i64)
```

The mantissa split order and widths are normative (review finding, LOW:
an implicit order would let two conforming hashers disagree). There is
no endianness anywhere in the definition — the format is specified over
i64/f64 *values*, not bytes; byte order becomes a question only for the
future single-stream interchange profile, which will carry its own hash
profile (review finding, LOW, accepted: that profile is a new
conformance surface and is deferred deliberately).

Not cryptographic — it is an integrity/identity mark, stated as such.
Constants hash through their exact bit decomposition (`f64_decompose`
from E2/v1a work), so two containers hash equal iff their planes are
bit-identical. Negative code words (there are none in v0 — all operands
are non-negative indices) are excluded by validation, keeping `mod`
semantics unambiguous.

## 4. The EVM

The EVM is the primary executor: load (validate) then step. Machine
state, semantics, poison rules I1–I5, and gate policy are exactly the
v1a interpreter's (`EMachine`, 16 registers × {val, err.hi, err.lo, u,
poison}, 8 memory slots), with two additions:

1. receipts are v=1 and cite `prog_hash` from the loaded container;
2. the machine refuses unvalidated programs (no raw-instruction entry
   point in the public surface; `eisa_step` remains for the differential
   harness only).

Arithmetic delegates to `eisa::core` reference functions — in the EVM,
observational equality with the reference is by construction; the bridge
(E4) is where it becomes a theorem checked by witness.

## 5. Conformance model

Three executors, one format, bit-exact agreement on the witness corpus:

| Executor | What it is | Agreement checked on |
|---|---|---|
| Reference | direct `eisa::core` call chains | lanes (val, err.hi, err.lo, u) |
| EVM | `.eisax` loaded and stepped | lanes + receipts + gate codes |
| x86 bridge | AOT-translated `.eisax` | lanes + receipts + gate codes |

Corpus: the dd64/eisa witness families (EFT bit-exactness, cancellation,
quadrature, poison propagation, round trips) plus the E5 scientific
kernel. A bridge that disagrees with the EVM by one bit on one lane on
one corpus program is non-conforming — same discipline as the gen2/gen3
bootstrap fixed point.

Honesty note (review finding, MED): corpus conformance is exactly that —
agreement on the corpus, not a proof of agreement on all programs. The
claim is falsifiable (any counterexample program extends the corpus and
indicts an executor) and the v0 machine is straight-line with a closed
observable set (§3.1b), which keeps the program space small enough that
corpus families genuinely cover the semantic surface: every opcode, both
poison paths, both gate branches, and the EFT identities appear in at
least one corpus program. Stronger-than-corpus claims are not made.

## 6. Phasing (replaces v0 spec §7)

| Phase | Deliverable | Depends on |
|---|---|---|
| E0 | this document, offload-reviewed | — |
| E1 | `format.sio` (encode/decode/validate/hash) + `asm.sio` (text → container) + fixed-point witnesses (asm → bin → disasm → asm) | E0 format freeze |
| E2 | `evm.sio` loader/executor + v=1 receipts + differential harness vs reference | v1a interpreter (seed), E1 |
| E3 | Sounio→`.eisax` backend for epistemic regions | explorer reports on IR/passes and language surface |
| E4 | `.eisax`→x86-64 AOT bridge + bit-exact differential witness vs EVM | E1–E3 |
| E5 | scientific kernel (PK dose step or Rump) end-to-end: same receipts from EVM and bridge | all |

Naming (2026-07-05, operator-approved): the surface language compiled by
the E3 backend is **Metron** (μέτρον, "measure") — *computation as
measurement*. In external-facing text the executor is **Metron VM
(MVM)**, avoiding the saturated "EVM" acronym. Internal identifiers
(`eisa::evm`, `eisa::backend`, `.eisax`) are unchanged in v1. Rationale
and rejected candidates: `eisa-v1-plan-2026-07-05.md` §1.

As built (2026-07-05): the E5 kernel is the reduced-quartic catastrophic
cancellation ((x-1)^4 at x = 1+1e-6 in monomial form) rather than full
Rump 1988, which does not fit the E3 v0 budget (16 registers / 32
instructions with move synthesis) honestly. The kernel lives in
`examples/eisa_cancellation_kernel.sio` (EVM half, witnesses K1–K5 in
`tests/stdlib/eisa/test_eisa_e5_kernel.sio`) and is program 5 of the
bridge conformance corpus (`scripts/ci/eisa_bridge_conformance_gate.sh`):
the AOT-translated ELF reproduces both receipts byte-identically — gate 1
shows the O(1) operand about to be annihilated, gate 2 shows val
collapsed to exactly +0.0 with the true ~2^-80 value in the roundoff
lane. The gate also carries a tamper lane and an anti-vacuity lane
(receipt value digits must be absent from the ELF bytes), so E5
conformance is witnessed against both a tampered image and a
baked-receipt translator.

## 7. Known risks, stated up front

- **Compiler fragility is the schedule risk**, not the design: today the
  checker hung on nested struct arrays (v1a, worked around with parallel
  arrays) and the default lane still has the imported-lane wrong-code
  residue. All witnesses stay on `lean_single` until the forensic
  dispatches land; every new hang/wrong-code shape gets recorded, not
  worked around silently.
- **Review orthogonality is degraded**: xai is currently the only live
  offload provider (DeepSeek balance, OpenRouter credits, Groq key all
  down). Every E-phase review is single-provider until a key is
  restored; re-review flagged in the offload log.
- **The hash is not tamper-proof** and the receipts say so (`v=1` is
  execution evidence with identity, not a signature). Cryptographic
  sealing is out of scope for v0 of the format.
- **Performance is explicitly not a goal** of the EVM; the bridge exists
  so that conformant fast execution remains possible.
