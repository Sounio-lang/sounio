<!-- docs:meta
topic_id: repo.docs.audit.r2-8-alias-deep-resolve.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-8-alias-deep-resolve.dispatch
-->

# DISPATCH R.2.8 — Deep alias resolution beyond function return signatures

**Opened.** 2026-05-17 (post R.2.7 Path A second-attempt diagnosis).
**Predecessor.** R.2.7 (compiler patch `ce9810ee9` fixed Pass 1c.5 return-type tuple re-encoding; Phase B caller wiring exposed that it's insufficient).
**Class.** Compiler edit on `self-hosted/compiler/lean_single.sio`. Self-hosted; requires `lean_single_fixed_point_gate.sh` + `native_v2_cpu_compiler_umbrella_gate.sh` PASS.
**Priority.** P4 — purely unblocks R.2.7 caller wiring cosmetic refactor. No correctness issue in any current stdlib or PBPK code.
**Branch.** `sounio-pure/r2-1-park-miller`.
**Time budget.** 2–4h single session.

---

## §0 — Sounio-Pure constraint

May read/write `self-hosted/compiler/lean_single.sio`, rebuild `bin/souc-linux-x86_64`, run scripts/ci gates, write probes under `docs/audit/r2_8_alias_deep_resolve/reference/`. No Python / R / external tools.

If a compiler-side fix requires touching the stage1 bootstrap chain (boot3/boot4) — **HALT and report**. R.2.8 is scoped to lean_single.sio + bin/souc rebuild only.

---

## §1 — What R.2.7's compiler patch did

Commit `ce9810ee9` ("selfhost: Pass 1c.5 re-encodes tuple return hashes for all aliases") extended `Pass 1c.5` in lean_single.sio to:

1. Re-scan every function's recorded `FN_RET_TY_TOK` after Pass 1c registers `type X = Y` aliases.
2. For tuple return types, redo the slot-count encoding via the Pass-1 block so `first_nslots`/`total_nslots` are computed using the now-resolved base type.
3. Write back `FN_RET_TY[i]` / `FN_RET_HASH[i]`.
4. Re-register the `tup_cache` so `.0` / `.1` field access on the return value uses the resolved layout.

Validation against R.2.7 Phase A smoke: `type DstPcg64 = PcgState` then
`fn dst_pcg64_next_i64(rng: DstPcg64) -> (DstPcg64, i64) { return pcg_step(rng) }`
**compiles bit-exact** to canonical PCG64 (R.2.4 oracle reproduces seed=31415 first sample).

Umbrella gate: 12/12 sub-gates PASS, no regression.

## §2 — What R.2.7 Phase B revealed it doesn't do

When Phase B wired the three caller modules to use `type X = PcgState` aliases, **consumer probes** (which import the wired API and use it like `var rng = pcg64_new(seed); ...; rng = r.0`) fail with:

```
error: assignment type mismatch at line N
```

The error fires from the assignment typecheck path (10 sites in lean_single.sio: lines 17015, 17194, 17333, 18002, 18051, 18111, 18260, 18309, 29160, 29202). Each compares a variable's slot type (recorded at `var` binding time) against the RHS expression's type, via `ty_eq`.

The hypothesis: `var rng = pcg64_new(seed)` records `rng`'s slot type from the **inferred return type of `pcg64_new`**. If that inference still references the alias hash (k=0, h=h_Pcg64) — not the resolved base — the later `rng = r.0` assignment (where `r.0` has the resolved PcgState type from Pass 1c.5) hits the mismatch path.

`ty_eq` at line 3058 short-circuits with `if k1 == 0 || k2 == 0 { return true }`, so unresolved-alias-to-resolved-base **should** pass. But empirically it doesn't — which suggests either:

- The assignment-check sites don't call `ty_eq` (they may compare `ST_FTY`/`ST_FHASH` directly).
- The unresolved-alias-side has k=6 (already resolved to type-6 with an *alias* hash, not type-0), bypassing the k==0 short-circuit.
- Some intermediate caching captured the pre-1c.5 tuple hash for `r.0` before Pass 1c.5 fixed it up.

R.2.8 Phase A must diagnose which.

---

## §3 — Attack plan

### Phase A — Diagnose which site fires (45 min)

1. Re-apply R.2.7 Phase B edits locally (don't commit). Compile `rng_oracle_gen.sio` to capture the exact failing error sites with line numbers.
2. For each "assignment type mismatch at line N" in the imported sources, map the line to the lean_single.sio error site (one of the 10 listed in §2).
3. Add temporary `print_type_label` debug output at the failing site to dump the `k1, h1, k2, h2` quadruple before `ty_eq` is called (or the inline comparator).
4. Rebuild souc, recompile, capture the type mismatch values.

Expected discovery: one of —
- (Da) RHS still encodes alias hash because expression-side tuple hash construction uses pre-1c.5 ty_slot_count.
- (Db) LHS slot type was recorded at `var x = expr` time before Pass 1c.5 ran (but Pass 2 runs after Pass 1c.5, so this is unlikely).
- (Dc) `r.0` field-access lookup hits a stale `tup_cache` entry from the function-call return site.

### Phase B — Patch (30–60 min)

Fix the diagnosed site. Options:

- **(Ba)** Extend Pass 1c.5 to also re-scan function-call return-value tuple components after the call site is resolved. Likely a Pass-2-side fix inside `compile_call` rather than Pass 1c.5.
- **(Bb)** Generalize the unresolved-alias-to-resolved-base shortcut in `ty_eq` to handle k=6 with alias hash (look up `REF_TYPE_DATA` for the alias name during ty_eq).
- **(Bc)** Eager-resolve all `var X` slot types after Pass 1c.5 by re-walking each function body.

Recommended starts with (Bb) since it's the smallest surgery: ty_eq becomes alias-aware once. The cost: a REF_TYPE_HASH lookup per nominal-mismatch comparison. Should be cheap (REF_TYPE_COUNT capped at 32).

### Phase C — Validation (30 min)

1. lean_single fixed-point gate: PASS required.
2. native_v2_cpu_compiler_umbrella_gate: all 12 sub-gates PASS required.
3. R.2.7 caller-wiring smoke: `rng_oracle_gen.sio` compiles AND its output bit-matches R.2.5 `rng_oracle_seed_*.txt` (1024/1024).
4. R.2.4 distribution oracle: 1024/1024 bit-exact (regression check on dst_pcg64_*).
5. R.2.5 sampling oracle: 256/256 bit-exact.

Total acceptance: 2304/2304 bit-exact oracle samples + 12/12 umbrella + 1 fixed-point.

### Phase D — Apply R.2.7 caller wiring (15 min)

With the deeper compiler patch in, R.2.7 Phase B's three caller-module edits should now compile and pass all oracles. Apply the same edits R.2.7 had drafted:

- `pcg64_core.sio`: add `pub PcgState` + `pub fn pcg_step` / `pcg_next_f64` / `pcg_next_f64_nonzero` / `pcg_bounded`.
- `distributions.sio`: `type DstPcg64 = PcgState`; wrappers as one-liners.
- `rng.sio`: `type Pcg64 = PcgState`; splitmix64 seed preserved; wrappers one-liners.
- `sampling.sio`: `type SmpPcg64 = PcgState`; canonical seed preserved; wrappers one-liners.

### Phase E — Commit (10 min)

Three commits:
1. `self-hosted/compiler/lean_single.sio` + `bin/souc-linux-x86_64` — the compiler patch.
2. `stdlib/random/*.sio` — R.2.7 Path A caller wiring.
3. `docs/audit/r2_8_alias_deep_resolve/` — DISPATCH + SYNTHESIS + reference probes.
4. `docs/audit/r2_7_pcg_state_unify/SYNTHESIS.md` — flip from HALTED to RESOLVED, point at R.2.8 commit.

HALT for operator review before push.

---

## §4 — Out of scope

- **Bootstrap chain edits** (boot3/boot4/native-v2). R.2.8 stays in lean_single.sio.
- **Algorithmic changes to PCG64.** R.2.7 Path A is pure code motion.
- **Renaming user-facing API.** `dst_pcg64_*`, `pcg64_*`, `smp_pcg64_*` keep their names.
- **Refinement-type predicates.** Pass 1c.5's existing predicate metadata propagation is preserved.

---

## §5 — Halt conditions

- **Diagnosis Phase A inconclusive after 90 min.** Suggests the bug is in stage0 bootstrap or beyond lean_single. Surface and stop.
- **Patch breaks lean_single fixed-point.** Revert; reassess.
- **Patch makes umbrella regress on any non-R.2.7 sub-gate.** Revert; the alias-handling change has unintended scope.
- **Oracle replay fails one sample after Phase D.** R.2.7 Path A caller wiring still produces bit-different stream — fix algorithm, not aliases.
- **Temptation to refactor `ty_eq` more broadly.** HALT; the targeted alias-aware path is what's authorized.

---

## §6 — Deliverables on close

1. `self-hosted/compiler/lean_single.sio` — deeper alias resolution (one of Phase B options).
2. `bin/souc-linux-x86_64` — rebuilt; new md5; size delta documented in synthesis.
3. `stdlib/random/pcg64_core.sio` — extended with PcgState + step/sampler functions.
4. `stdlib/random/distributions.sio`, `rng.sio`, `sampling.sio` — wired via type alias.
5. `docs/audit/r2_8_alias_deep_resolve/DISPATCH.md` — this file.
6. `docs/audit/r2_8_alias_deep_resolve/SYNTHESIS.md` — closing writeup with diagnosis location + patch diff.
7. `docs/audit/r2_8_alias_deep_resolve/reference/alias_deep_probe.sio` — minimal Phase A reproducer.
8. `docs/audit/r2_7_pcg_state_unify/SYNTHESIS.md` — flip to RESOLVED.

---

## §7 — Acceptance

R.2.8 is **VALIDATED** iff:

1. ✓ `lean_single_fixed_point_gate.sh`: PASS (stage1 == stage2 == stage3).
2. ✓ `native_v2_cpu_compiler_umbrella_gate.sh`: 12/12 sub-gates PASS.
3. ✓ R.2.4 distributions oracle replay: 1024/1024 bit-exact.
4. ✓ R.2.5 sampling oracle vs R.2.4: 256/256 bit-exact.
5. ✓ R.2.5 rng self-oracle: 1024/1024 bit-exact.
6. ✓ R.2.4 stat sanity: 6/6 PASS.
7. ✓ R.2.7 Phase B caller wiring lands; R.2.7 SYNTHESIS.md flipped from HALTED → RESOLVED.
8. ✓ Net LOC delta in `stdlib/random/`: ≤ −40 (the R.2.7 §7.5 floor).

If 1 or 2 fails: FAIL. Revert.
If 3-5 fails: FAIL. Wiring drift; debug before resuming.
If 7 fails: PARTIAL — R.2.8 may still ship the compiler patch alone if Phase D hits unexpected friction; R.2.7 stays HALTED.

---

## §8 — Notes

- The R.2.7 compiler patch (`ce9810ee9`) is **load-bearing** independent of R.2.8. It fixes a real Sounio typechecker bug at the function-return site for any alias of a multi-slot struct. Even if R.2.8 fails, R.2.7's patch stays valuable.
- R.2.8 is the **completion** of R.2.7, not a replacement. The two together deliver the full R.2.7 Path A: zero-downstream-signature-change caller wiring via type aliases.
- After R.2.8, the stdlib RNG family lands at its natural end state: one source of truth in `pcg64_core` (helpers + state + step + samplers); each caller has its seed function, distribution samplers, and one-line `type X = PcgState` alias. Net session contribution to `stdlib/random/`: roughly −189 LOC vs pre-R.2.4 (R.2.6's −149 + R.2.8 Phase D's additional −40).

**END OF DISPATCH.**
