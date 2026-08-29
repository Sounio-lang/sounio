<!-- docs:meta
topic_id: repo.docs.handoff.compiler-struct-array-mul-loop-fix-prompt
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.compiler-struct-array-mul-loop-fix-prompt
-->

# Prompt — fable5: fix the `[struct;N]` multiply-accumulate loop miscompile (unblocks exact CD over unbounded ℚ)

**For:** fable5 (compiler-internals agent)
**Authored by:** Claude (exact-algebra-core lane), 2026-07-06
**Tracks:** issue **#651**. Sequel to the generic-`<F>` unblock (**#650**, `2adb8f061`) — that one is DONE; this is the *one remaining* codegen wall in front of the exact-algebra lane.
**Type:** self-hosted compiler codegen (aggregate element store + struct-return temporary), **both engines** (lean_single + Madaros). Serialized surface `self-hosted/compiler/lean_single.sio` + `bin/souc-linux-x86_64` — coordinate with Lane 4. Fixed-point + **output-verified** gates mandatory.

---

## 0. Goal (one acceptance target)

Make a **read-modify-write accumulation loop over an array-of-struct** produce correct values. Concretely, `cd_mul_exact<Rational>` in `stdlib/algebra/cayley_dickson_exact.sio` must run correctly, so the exact Cayley–Dickson product works over `F = Rational` (and thence `F = BigInt` / `BigRational` — the exact product over **unbounded ℚ for all 16 components**, the lane's last residual). Today this multiply **silently corrupts** at small N and **SIGSEGVs** at large N.

## 1. The bug — exact, and it is NOT generics

The generic engine is fine for scalar `F` (`F=i64` is byte-identical to the hand-written engine). The defect is the `[Rational;N]` (array-of-**2-field-struct**) multiply-accumulate **loop**, and it **reproduces with no generics at all**.

**Symptom scaling** (stage2 souc from a post-#650 `main` run):
- `N=16`   → **deterministic garbage** (`c[0] ≈ 4.2e6/1`, varies run-to-run = corrupted memory; correct is `1/1`).
- `N=2048` → **SIGSEGV** (exit 139) — the miscomputed access walks out of bounds.

**Minimal repro** (committed, non-generic): `docs/handoff/repros/d8_generic_struct_F_mul_segv.sio`
```sounio
struct SmallR { c: [Rational; 16] }                       // array-of-struct
fn sr_mul(a: SmallR, b: SmallR) -> SmallR with Mut, Panic, Div {
    var r = SmallR { c: [rat_zero(); 16] }
    var i: i32 = 0
    while i < 16 { var j: i32 = 0
        while j < 16 {
            let idx = i ^ j
            r.c[idx as usize] = rat_add(r.c[idx as usize], rat_mul(a.c[i as usize], b.c[j as usize]))
            j = j + 1 }
        i = i + 1 }
    r
}
// a.c[1]=1, b.c[1]=1 → e1*e1: only idx=0 accumulates 1*1=1. Prints c0≈4.2e6 instead of 1.
```

## 2. Isolation — the single most useful thing here

I ran the discriminators. **Every sub-operation is correct; only the full nested loop over array-of-struct temporaries corrupts.** This is the fingerprint — use it.

| Case | Result |
|---|---|
| Generic RMW loop, `F=i64`, `[i64;16]` | ✅ correct → **generics are not the cause** |
| Non-generic concrete `[Rational;16]`, full loop (above) | ❌ garbage |
| Single **constant-index** RMW on `[Rational;2048]` (`r.c[0]=rat_add(r.c[0], rat_mul(a.c[1],b.c[1]))`) | ✅ correct |
| **Variable-index READ** of a `[Rational;16]` element | ✅ correct |
| Variable-indexed struct element as a **by-value arg** (`rat_mul(a.c[i], b.c[j])`) | ✅ correct |
| **One** variable-index RMW iteration | ✅ correct |
| The **full 16×16** accumulate loop | ❌ garbage@16 / SIGSEGV@2048 |

So the corruption needs the **combination**: a runtime-variable-indexed **store** back into `[struct;N]`, whose RHS is a **struct-returning call** (`rat_add`), **iterated** many times. `[i64;N]` (scalar) is unaffected — the concrete-i64 engine runs this exact loop over `[i64;2048]` correctly.

## 3. Relation to known defects (reconcile — don't refile a dup)

- **DISTINCT from #637** — #637 is a *cross-module arity-mismatch delegation* **compiler** SIGSEGV during lowering. This is **single-module**, a **runtime** wrong-value that only segfaults at large N. Don't conflate.
- **Same family as #643 / D6** — struct value-copy aliasing/corruption (`var r = a; r.field = x` aliased the caller). The likely mechanism here is the same class one level up: the RHS struct-return temporary (rat_add's returned `Rational`) and the array-of-struct destination slot are not kept disjoint across iterations.

## 4. Working hypothesis (verify, do not assume)

The `Rational` return of `rat_add`/`rat_mul` is a small aggregate returned via a hidden pointer / stack temporary. When its value is stored into `r.c[idx]` — an array element whose element **stride is `sizeof(Rational)` (2 slots), not 1** — one of these is likely wrong **only in the loop body**:
1. the element **address = base + idx * stride** miscomputes `stride` (uses 1-slot / i64 stride for a 2-field struct → writes land half-overlapping adjacent slots → neighbor corruption at N=16, OOB at N=2048); and/or
2. the **struct-return temporary is allocated once and reused**, so read (`r.c[idx]` as the add's LHS arg) and write-back alias across iterations; and/or
3. the copy of the returned aggregate into the slot copies the wrong **number of slots** (off-by-`nslots` — cf. the #640 struct-literal aggregate field-copy off-by-(nslots−1) already fixed for a related shape).

Start by dumping the lowered address arithmetic for `r.c[idx as usize] = <struct-returning call>` and compare the struct-element **stride/nslots** against the working `[i64;N]` store. #640's fix (`fix(native): a64 struct-literal aggregate field copy off-by-(nslots-1)`) and the #642 two-level chained field-assignment fix are the nearest prior art — grep those commits for the aggregate store/stride code and check the **indexed-array-of-struct** path shares (or should share) it.

## 5. Acceptance criteria (ALL; output-verified; both engines)

1. `docs/handoff/repros/d8_generic_struct_F_mul_segv.sio` → prints `c0=1/1` (promote a corrected copy to `tests/run-pass/struct_array_mul_loop.sio` asserting `1/1`).
2. The `[Rational;2048]` variant of the same loop **does not SIGSEGV** and gives the correct component.
3. `cd_mul_exact<Rational>` on the canonical sedenion pair `a=e₃+e₁₀, b=e₆−e₁₅` → **annihilates** (all 16 comps `0/1`), AND matches the `F=i64` engine component-for-component. Add `tests/run-pass/cd_exact_rational.sio` (the ZD pair + one genuinely fractional case, e.g. coeffs `1/2, 1/3`).
4. **Cross-verify against an independent oracle** — extend `scripts/research/cd16_oracle.py` (Python `fractions`) to emit the expected 16 rationals; the test's values must match element-wise. (souc miscompiles SILENTLY — a bare `PASS` is not evidence; see §6.)
5. Self-host fixed point preserved (canonical gate; gen2==gen3). No regressions: whole exact-algebra suite (`sedenion_zd_census_168`, `cd_exact_generic_*`, `sedenion_cd_full16_q`, `bignat_selftest*`, …), `generic_struct_*`, `closure_generic_hof` still pass; full run-pass fail-count not worse than baseline.
6. Madaros: mirror-fix + **output-verified** (assert printed values, not rc — the compact stub backend false-greens on exit code).

## 6. Hard-won gotchas (these cost me hours)

- **souc miscompiles SILENTLY.** rc is a false-green (exit 0 on wrong output); this very bug prints a clean, plausible-looking wrong number. NEVER trust a bare `PASS`. Verify every numeric result against a non-souc oracle (Python `fractions`). This bug was found ONLY by cross-checking a `Rational` result against the known `i64` annihilation.
- **CI uses a FRESH stage2 souc built from source** (`SOUNIO_TEST_SOUC_BIN=/tmp/souc-stage2`), NOT the committed `bin/souc` (older). Local pass ≠ CI pass. To repro a CI-only failure, download the exact artifact: `gh run download <runid> -n native-compiler-linux-x86_64` → `souc-stage2` (`mini_native <src> <out>`; `chmod +x` the ELF). I reproduced #651 with a post-#650 stage2, not `bin/souc`.
- **The garbage value varies run-to-run** (uninitialized/corrupted memory) — gate on `== 1/1`, never on the specific wrong number.
- Build on **current main** (has #630/#632/#633/#640/#642/#644 + the #650 generic-`<F>` work).

## 7. Protocol

Fresh worktree off `origin/main`; **CLAIM** in `artifacts/omega/agent_handoff.log.md` before editing; serialize `lean_single.sio` + the `bin/souc` token with Lane 4. Rebuild to fixed point; **output-verify** every witness. On **RELEASE**: post `commit=<sha>` + checks and ping `coord/exact-algebra-core` via the log so the lane switches `cd_mul_exact<Rational>`/`<BigInt>` on and closes the unbounded-ℚ residual.

## 8. Out of scope

The math (all verified in Lean + Python oracles) and the generic monomorphizer (#650, working). Smallest codegen change that makes an **indexed array-of-struct store whose RHS is a struct-returning call, iterated in a loop**, produce correct, in-bounds writes.
