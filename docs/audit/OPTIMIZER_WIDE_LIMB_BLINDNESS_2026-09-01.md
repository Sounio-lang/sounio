<!-- docs:meta
topic_id: repo.docs.audit.optimizer-wide-limb-blindness-2026-09-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.optimizer-wide-limb-blindness-2026-09-01
-->

# Every cleanup peel is blind to wide-integer limbs

**Status:** FIXED. Root cause below; the fix and its measurement are at the end.
**Scope:** the 26 `-O` divergences left after the four peel fixes of 2026-08-31.

## The fact

A wide integer is software-emulated over CONSECUTIVE INTEGER LIMBS. `ir.sio:1889`
states it:

> wide floats are software-emulated over consecutive INTEGER limbs (the 11
> IrWide* opcodes carry limb_count)

An `IrWide*` instruction names only the BASE register in `dst`/`src1`/`src2` and
carries the limb count in `imm_i64`. The value actually occupies
`r .. r + limb_count - 1`.

    grep -c limb self-hosted/ir/opt_cleanup.sio
    0

Not one peel knows this. Every one of the thirteen tracks state per register and
sees only the base, so limbs 1..n-1 of every wide value are invisible: to
liveness they are never used, to invalidation they are never written.

## Measured

`tests/run-pass/lorenz_i256_step1_taylor2_radius_artifact_tiny.sio`, one
compiler, built twice, differing only in `-O`. Without `-O` it returns 0; with
`-O` it returns 1.

**Three peels break it independently**, which is why disabling any single one
never fixed it and the file survived four rounds of peel fixes:

    only COPY_PROP enabled    rc=1
    only DCE_LOOP  enabled    rc=1
    only DCE_FINAL enabled    rc=1     (same peel, ocp_mfi_dce_once, second call site)
    all thirteen disabled     rc=0

`DCE_LOOP` and `DCE_FINAL` are the same peel at two call sites, so this is two
defects, not three.

**The defect tracks integer width exactly.** Same program, only the type
changed:

    i256   diverges     (4 limbs)
    i128   diverges     (2 limbs)
    i64    agrees       (1 limb -- no limb to be blind to)

That is the cleanest evidence available that the multi-limb representation is
the cause, and not, say, register pressure or program size.

## What it needs to reproduce

Each of the four mask functions in that file returns the SAME value with and
without `-O` when compiled alone. The divergence appears only when all four are
called in one expression:

    let ok_mask = conversion_mask() + derivative_mask() + next_radius_mask() + cap_mask()

Replacing the four with functions that return constants does not reproduce it --
the wide-arithmetic bodies are required. A shorter reproducer than the four
real bodies was not found; three attempts at synthesising one from the failing
function's shape all agreed under both builds.

## Where a fix would go

Ten of the eleven wide opcodes store the limb count in `imm_i64`:

    ir_wide_add sub mul cmp div mod shr shr_limb div_full mod_full   imm_i64 = limb_count
    ir_wide_reject                                                   imm_i64 = base

`ir_wide_reject` is the exception and must not be read the same way.

A peel needs to treat a wide operand as the RANGE `r .. r+n-1`:

  * `ocp_mfi_dce_once` must mark every limb of `src1`/`src2` used, and must not
    NOP a definition unless every limb of its `dst` is unused.
  * `ocp_mfi_copy_prop` must invalidate every limb of a written range, both as
    key and as value.
  * The other eleven peels have not been checked and are blind by the same
    grep.

## Caveats

`lorenz_i256_fixed_step` is in this family but is a trap: it is annotated
`known-failure`, returns 4 WITHOUT `-O` and 0 WITH it. There the optimiser is
right and the default path is wrong, which inverts the assumption that the
unoptimised build is the reference.

`wide_i128_fn_abi_known_failure` is likewise annotated as a known failure.

The 26 have not been shown to be ONE defect. They share a family and a width
dependence; that is a common cause, not a proof of a single one.


## Fixed

Seven peels now read an operand as the range `r .. r+n-1`: `dce_once`,
`copy_prop`, `dse`, `dedup_imm`, `const_fold`, `sccp` and `cse`. The other six
keep no per-register state.

Full sweep over `tests/run-pass`, one compiler built twice, differing only in
`-O`, `SOUNIO_STDLIB_PATH` pinned -- the same denominator as the measurement
above, so the two are comparable:

    compared                  1725
    divergent                    0   (26 before)
    of which output-changing     0
    new divergences              0

`-O` no longer changes the behaviour of any program in `tests/run-pass`.

Two asymmetries turned up while reading the opcode list, neither in the original
hypothesis above:

  `IrWideCmp` reads two wide values and writes ONE boolean, so its destination
  span is computed separately from its operand span. Widening the destination
  would clobber the register after it.

  `src2` is not always a register: on `IrWideShr` and `IrWideShrLimb` it carries
  the shift amount. `copy_prop` rewrote `src2` unconditionally, and a small
  shift count collides with a low register number.

`dse` is deliberately conservative: an elimination requires BOTH sides to be
single-limb, because NOPing a wide instruction whose base was overwritten would
delete the definitions of its other limbs with it.

The claim above that the 26 were not shown to be ONE defect still stands as
written -- they all cleared together under one change, which is consistent with
a single cause but was not proved to be one before the fix.
