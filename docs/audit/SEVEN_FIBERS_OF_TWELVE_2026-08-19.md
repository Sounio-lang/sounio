<!-- docs:meta
topic_id: repo.docs.audit.seven-fibers-of-twelve-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.seven-fibers-of-twelve-2026-08-19
-->

# Seven fibers of twelve — do they exist, and what are they?

**Filed:** 2026-08-19 · **Lane:** grok-cli5 · **Verdict:** SEVEN DISJOINT FIBERS

**SHA:** `f79c93c772` (`origin/main`). Lean 4.33.0 on Slurm (`cpuops-t560-proxmox`).

This is a measurement of a *prediction*, not of a theorem name.
claude-1 derived “there are seven fibers of twelve, orthogonal to the
annihilator split” from the arithmetic `84 ÷ 12 = 7`. Arithmetic is
true. The structural claim is only half true.

Existing theorems in `SounioZeroDivisorBridge` / `SounioSunflower` /
`SounioSurgicalCalculus` are **not edited**. New statements live in
`formal/lean4/SounioSevenFibers.lean`. No `sorry`. No Mathlib.

## Verdict

**SEVEN DISJOINT FIBERS.** Labels `{9,10,11,12,13,14,15}`, each of
cardinality 12, pairwise disjoint, union 84. The equivalence relation
is `xorLabel v = v.lo XOR v.hi`, not an orbit and not a conjugacy
class. Read from `Sounio.ZeroDivisorBridge.xorLabel`.

The orthogonality half of the prediction is **false**. For every
primitive `u`, the 4-element annihilator kernel sits entirely inside
`u`'s own fiber. The kernel never splits across fibers. The two
partitions are **nested**, not orthogonal.

## 1. Do the fibers exist as a partition?

Independent Python port of the Lean definitions
(`scripts/dev/seven_fibers_measure.py`) and Lean `native_decide` agree:

| | Python | Lean 4.33.0 (Slurm) |
|---|---|---|
| `validPrims` | 84 | `prim_total` |
| active labels | 7: 9..15 | `zd_fiber_labels` / `each_fiber_card_12` |
| size of each | 12 | 12 |
| pairwise overlap | 0 | `fibers_pairwise_disjoint` |
| union | 84 | `seven_times_twelve_is_eighty_four` |
| uncovered | 0 | `every_prim_has_active_label` |

There are not “some other number of fibers of mixed sizes”. There are
exactly these seven, all size 12. Each fiber list is `Nodup`
(`each_fiber_nodup`), so the 12 is a set cardinality.

## 2. What defines a fiber?

`xorLabel v := v.lo ^^^ v.hi`. A fiber is the preimage of one value
under that function, restricted to `validPrims`.

`validPrims` itself already excludes xor-label 8 (the diagonal family
`e_i + e_{i+8}`) and `hi = 8`. That is why the live labels start at 9,
not at 1. The 7 labels `{9..15}` are `{1..7}` with the high bit set
(`L ^^^ 8`), which is the Fano-line correspondence already recorded
as `zd_labels_mirror_fano_indices`. The fiber is an xor-class, not a
group orbit.

`edit u` is defined as that preimage:

```
| .edit => validPrims.filter (fun v => xorLabel v == xorLabel u)
```

So `edit` *is* the fiber. The other five ops are unions of blocks of
the annihilator split `{kernel 4, self 1, complement 79}`.

## 3. Positive control — the 12 members of `fiber9Prims`

In the order `fiberPrims 9` produces them (filter of `allPrims`):

```
e2+e11  e2-e11
e3+e10  e3-e10
e4+e13  e4-e13
e5+e12  e5-e12
e6+e15  e6-e15
e7+e14  e7-e14
```

Lean: `fiber9_is_these_twelve : fiber9Prims = fiber9Explicit`.
Python printed the same 12 strings.

`primA = e3+e10` is the third entry. Its kernel (independently
recomputed from `cdSigma` / `isZeroPair`) is

```
e4+e13  e5-e12  e6-e15  e7+e14
```

all of xor-label 9. That matches `annihilatorsOfA` in the Bridge file.

## 4. Negative control — a primitive not in `fiber9Prims`

`outsider = e1+e10` (`PrimSed.mk 1 10 false`).

- `isPrimValid outsider = true`
- `xorLabel outsider = 11`
- `fiber9Prims.contains outsider = false`

If the “fiber” had been the whole set of 84, this line would fail.

## 5. Are the two partitions orthogonal?

No. Census over all 84 primitives (Python, then Lean):

| | count |
|---|---:|
| kernel size 4 | 84 / 84 |
| distinct xor-labels in a kernel | **1** (84 / 84) |
| kernel entirely in `u`'s fiber | **84 / 84** |
| kernel splits across fibers | **0** |
| `u` itself in its kernel | 0 |

So the annihilator 4-set of `u` is not “one per fiber” and not
patternless. It is **all in the same fiber as `u`**.

Concretely, for any `u`:

```
fiber(u)  =  {u}  ∪  kernel(u)  ∪  7 other fiber-mates     (12)
unlearn u =  kernel(u)                                      (4)
edit u    =  fiber(u)                                       (12)
gate u    =  84 \ ({u} ∪ kernel(u))                         (79)
```

`{u}` and `kernel(u)` both sit inside one block of the fiber
partition. The other six fibers (72 primitives) lie entirely in
`gate u`. That is nesting, not orthogonality.

## 6. Does Editable (G5) reduce to ExactlyPrivate (G3)?

No, and they are not two independent primitives either.

- G3 / `unlearn` is the 4-kernel.
- G5 / `edit` is the 12-fiber, which **properly contains** the kernel
  (`kernel_subset_of_edit`, `edit_twelve_unlearn_four`,
  `self_in_edit_not_in_unlearn`).
- The eight-type family therefore does **not** collapse to one
  primitive (12 ≠ 4).
- The independence prediction — “reduces to two primitives, not one”
  — is **not supported**. The evidence is a *nested pair*: fiber, and
  kernel-inside-fiber. The other ops are Boolean combinations of the
  annihilator split.

## Lean on Slurm

Host `cpuops-t560-proxmox`, partition `cpu-ops`. Lean 4.33.0
(`d8b18978322de05a`, the pin in `formal/lean4/lean-toolchain`).
Slim toolchain shipped by tarball (workspace is invisible on the
node). `--threads=1 --tstack=8192` is required: without it Lean
aborts with `failed to create thread` inside the Slurm cgroup.

```
OK SounioCayleyDickson.lean
OK SounioZeroDivisorBridge.lean
OK SounioSurgicalInterventions.lean
OK SounioSurgicalCalculus.lean
OK SounioSevenFibers.lean
SUMMARY ok=5 fail=0
```

No `declaration uses sorry`. Existing theorem *statements* were not
changed.

## What this does not decide

- Whether xor-label is the *right* name for the G5 locality bound in
  the compiler. It is the name the Lean already uses.
- Whether the eight surface types should be rewritten. Measurement
  only.

## Reproduce

```bash
python3 scripts/dev/seven_fibers_measure.py --json-out /tmp/seven_fibers.json
# Lean, Slurm, not the pod; --threads=1 --tstack=8192
# lake build SounioSevenFibers   # or lean --o on the five-file chain
```

Instrument: `scripts/dev/seven_fibers_measure.py`.
Lean: `formal/lean4/SounioSevenFibers.lean`.
Table: `docs/audit/SEVEN_FIBERS_OF_TWELVE_2026-08-19.tsv`.
