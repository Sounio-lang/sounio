<!-- docs:meta
topic_id: repo.docs.audit.r2-6-pcg64-core-unify.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-6-pcg64-core-unify.dispatch
-->

# DISPATCH R.2.6 — Unify the three PCG64 inlinings into a shared core

**Opened.** 2026-05-17
**Predecessors.** R.2.4 (distributions.sio canonical, RESOLVED), R.2.5 (rng.sio + sampling.sio canonical, RESOLVED).
**Class.** Pure stdlib refactor. No compiler involvement. No algorithmic change.
**Priority.** P4 — purely tech-debt. None of {PBPK28, dissertation MC, downstream samplers} depend on this. Defer freely.
**Branch.** Continue on `sounio-pure/r2-1-park-miller` (oracles in scope).
**Time budget.** 1–2h single session.

---

## §0 — Sounio-Pure constraint

May read/write `stdlib/random/distributions.sio`, `stdlib/random/rng.sio`,
`stdlib/random/sampling.sio`, the new `stdlib/random/_pcg64_core.sio`,
`stdlib/random/lib.sio` (re-export hints if needed). Probes under
`docs/audit/r2_6_pcg64_core_unify/reference/`. No Python / R / external
tools. The R.2.4 and R.2.5 oracles are the canonical regression target —
no new pcg-cpp cross-check needed.

**HALT and report** if any algorithmic edit becomes necessary. R.2.6 is a
pure code-motion refactor.

---

## §1 — Scope

After R.2.4 + R.2.5, the stdlib ships **three independent copies** of the
same six-helper canonical PCG64 core:

| Module | Helper prefix | Step fn | Wrapper struct (U128 alias) |
|---|---|---|---|
| `distributions.sio` | `dst_pcg_` | `dst_pcg64_*` | `DstU128` |
| `rng.sio`           | `rng_pcg_` | `pcg64_*`     | `RngU128` |
| `sampling.sio`      | `smp_pcg_` | `smp_pcg64_*` | `SmpU128` |

Helpers duplicated (≈70 LOC × 3 = 210 LOC, bit-identical bodies modulo
prefix):

- `u_lt(a, b) -> bool`
- `lshr(x, n) -> i64`
- `umul64_high(a, b) -> i64`
- `u128_add(a_hi, a_lo, b_hi, b_lo) -> U128`
- `u128_mul(a_hi, a_lo, b_hi, b_lo) -> U128`
- `rotr64(x, rot) -> i64`
- (rng.sio + sampling.sio also: `umod(x, n) -> i64` for rejection-modulo)

The **step** function bodies are also bit-identical (after seeding), but
they're tied to module-specific state types (`DstPcg64` / `Pcg64` /
`SmpPcg64`). The refactor scope decides whether to unify these too.

---

## §2 — Design decisions

### §2.1 — Core module name

Recommend **`stdlib/random/_pcg64_core.sio`**. Leading underscore signals
"stdlib-internal, not part of the user-facing module surface." Alternative
without the underscore (`stdlib/random/pcg64_core.sio`) is also fine but
mixes with the user-facing module list in `lib.sio`. Underscore is the
existing convention for "library private" in this codebase if such
convention exists — verify before committing.

### §2.2 — Unify struct U128 too?

Three options:
- **(a) One `PcgU128` struct in the core module.** Cleanest. All three
  callers `use stdlib::random::_pcg64_core::{PcgU128, ...}`. Recommended.
- (b) Keep three structs, just share helpers via duck-typing. Doesn't work
  — Sounio struct types are nominal, so the helpers can only return one
  named type.
- (c) Skip core-level structs; have helpers take/return `(i64, i64)` tuples.
  Loses the named-field readability that R.2.4/5 introduced.

**Recommended: (a).**

### §2.3 — Unify the step function too?

The step body (`u128_mul state mult → u128_add → XSL-RR`) is identical
across all three callers. But each caller has its own state struct
(`DstPcg64`, `Pcg64`, `SmpPcg64`) so a single `pcg64_step` cannot return
all three.

Three options:
- **(a) Helpers only; step stays in each module.** Each step shrinks to
  ~10 LOC (mul + add + XSL-RR) and just calls into the shared helpers.
  Recommended. Smallest blast radius.
- (b) Add a fourth core type `PcgState` with hi/lo/inc_hi/inc_lo; have
  the three callers convert to/from it. Adds adapter shims; LOC delta
  negative but conceptually heavier.
- (c) Replace all three caller-side structs with the core's `PcgState`
  and rename users. Touches every downstream caller. Out of scope per §4.

**Recommended: (a).**

### §2.4 — Visibility

Core helpers and `PcgU128` are exposed as `pub` from `_pcg64_core.sio`.
The three caller modules `use` them. No re-export through `lib.sio` —
callers wanting the helpers explicitly import from `_pcg64_core`.

### §2.5 — Naming

Helpers in the core module drop the prefix:

| was (3×) | becomes (1×) |
|---|---|
| `{dst,rng,smp}_pcg_u_lt`        | `pcg_u_lt` |
| `{dst,rng,smp}_pcg_lshr`        | `pcg_lshr` |
| `{dst,rng,smp}_pcg_umul64_high` | `pcg_umul64_high` |
| `{dst,rng,smp}_pcg_u128_add`    | `pcg_u128_add` |
| `{dst,rng,smp}_pcg_u128_mul`    | `pcg_u128_mul` |
| `{dst,rng,smp}_pcg_rotr64`      | `pcg_rotr64` |
| `{rng,smp}_pcg_umod`            | `pcg_umod` |

Type: `{Dst,Rng,Smp}U128` → `PcgU128`.

---

## §3 — Attack plan

### Phase A — Author `_pcg64_core.sio` (20 min)

Write the new module with `pub` helpers and `PcgU128` struct. Body
bit-identical to the existing `dst_pcg_*` helpers (which are the R.2.4-
certified copy). Add the `pcg_umod` helper from R.2.5.

### Phase B — Wire callers (30 min)

For each of distributions.sio, rng.sio, sampling.sio:
1. Add `use stdlib::random::_pcg64_core::{PcgU128, pcg_u_lt, pcg_lshr, pcg_umul64_high, pcg_u128_add, pcg_u128_mul, pcg_rotr64}` (+ `pcg_umod` for rng/sampling).
2. Delete the six (seven) prefixed helpers.
3. Delete the local U128 struct.
4. Rewrite the step body to call the unshifted names and use `PcgU128`.
5. **Do not change** the seed function (preserves bit-exactness).

### Phase C — Regression validation (10 min)

Re-run all three R.2.4/5 probes:
1. `r2_4/reference/pcg64_fingerprint_probe.sio` — must still emit
   32 bit-exact samples vs pcg-cpp HEAD (already captured at
   `/tmp/pcg_verify/pcgcpp_out.txt`, but we just diff against the
   committed oracle files which are the canonical witness).
2. `r2_4/reference/stdlib_validation_probe.sio` — must still emit
   1024/1024 vs `r2_4/.../oracle_seed_*.txt`.
3. `r2_5/reference/sampling_validation_probe.sio` — must still emit
   256/256 vs R.2.4 oracle.
4. `r2_5/reference/rng_oracle_gen.sio` — must still emit
   1024/1024 vs committed `r2_5/.../rng_oracle_seed_*.txt`.
5. `r2_4/reference/phase_c_stat_sanity.sio` — 6/6 stat bands (this is
   the safety net against accidental algorithmic drift; oracle match
   already implies algorithm match).

Total: **2304/2304 bit-exact** + 6/6 stat.

### Phase D — Doc cleanup + commit (10 min)

1. `lib.sio`: add `_pcg64_core` to the submodule list with a one-line
   description ("stdlib-internal — shared u128 helpers; not user-facing").
2. `SYNTHESIS.md` in `docs/audit/r2_6_pcg64_core_unify/` against §7
   acceptance. Net LOC delta (expected: ≈ −150 across stdlib/random/).
3. Single commit; HALT for operator review before push.

---

## §4 — Out of scope

- **Algorithmic edits.** R.2.6 is pure code motion. Any algorithmic
  change → HALT.
- **Renaming user-facing API.** `dst_pcg64_*` / `pcg64_*` / `smp_pcg64_*`
  keep their names and ABIs. Downstream callers are not touched.
- **Renaming caller-side state structs.** `DstPcg64` / `Pcg64` /
  `SmpPcg64` remain.
- **Unifying step into one function.** Per §2.3 option (a).
- **Park-Miller, xoshiro, splitmix64, mt19937.** Untouched.

---

## §5 — Halt conditions

- **Any of the four oracle replays fails one sample.** Refactor introduced
  an algorithmic bug. Surface the diff against the committed oracle.
- **Stat sanity drifts outside R.2.4 Phase C bands.** Same as above.
- **`use` import edges produce typecheck errors that didn't exist
  pre-refactor.** Likely a visibility / `pub` issue; report and surface.
- **LOC delta is positive.** Means the refactor didn't actually
  consolidate; reconsider scope.
- **Any temptation to also unify the step function or rename callers.**
  HALT — that's a follow-up refactor, not R.2.6.

---

## §6 — Deliverables on close

1. `stdlib/random/_pcg64_core.sio` — the new core module (~80 LOC).
2. `stdlib/random/distributions.sio` — helpers + DstU128 removed; `use` added; step rewritten.
3. `stdlib/random/rng.sio` — same.
4. `stdlib/random/sampling.sio` — same.
5. `stdlib/random/lib.sio` — submodule list updated.
6. `docs/audit/r2_6_pcg64_core_unify/SYNTHESIS.md` — closing writeup with LOC delta and oracle-match table.

---

## §7 — Acceptance

R.2.6 is **VALIDATED** if and only if:

1. ✓ R.2.4 oracle replay: `stdlib_validation_probe` → 1024/1024 bit-exact.
2. ✓ R.2.5 sampling oracle replay: 256/256 bit-exact vs R.2.4 oracle.
3. ✓ R.2.5 rng self-oracle replay: 1024/1024 bit-exact vs committed `rng_oracle_seed_*.txt`.
4. ✓ R.2.4 statistical sanity (Uniform/Normal/Exp mean+var) 6/6 PASS.
5. ✓ Net LOC delta in `stdlib/random/` is ≤ −100 (consolidation actually happened).
6. ✓ No user-facing API renamed (`dst_pcg64_*` / `pcg64_*` / `smp_pcg64_*` ABIs preserved).
7. ✓ No algorithmic change (verified by §7.1-§7.3 bit-exactness).

If 1, 2, or 3 fails: FAIL. Refactor introduced a bug. Back to Phase A.
If 4 fails but 1-3 pass: HALT — likely typecheck regression somewhere;
investigate before declaring success.
If 5 fails: PARTIAL. The refactor didn't actually consolidate; reconsider.

---

## §8 — Notes

- The three callers were authored as separate inlinings because stdlib
  cross-imports were less ergonomic at the time. R.2.4/5 stabilised them
  algorithmically without consolidating. R.2.6 is the natural follow-up
  now that the algorithm is stable.
- Leading-underscore module names are not enforced anywhere in souc's
  module resolver — they're a docs convention only. If they break import
  paths in any caller, drop the underscore.
- This dispatch is genuinely optional. The duplication is harmless
  (no correctness or performance cost; the helpers are tiny pure
  functions). The win is purely maintainability: a future algorithmic
  fix would otherwise need to be re-applied in three places, with the
  R.2.4/5 sequence as evidence that "fix in three places independently"
  is error-prone.

**END OF DISPATCH.**
