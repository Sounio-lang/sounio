<!-- docs:meta
topic_id: repo.docs.audit.r2-9-var-tuple-dot-zero-typecheck.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-9-var-tuple-dot-zero-typecheck.dispatch
-->

# DISPATCH R.2.9 — `var x = fn(seed); x = tuple.0` typecheck fragility

**Opened.** 2026-05-18 (R.2.8 §6 follow-up).
**Predecessor.** R.2.8 (RESOLVED-PARTIAL, `ff46b8ce5`). R.2.8's rng.sio
revert was forced by this bug surfacing under Pass 1c.5.
**Class.** Compiler edit on `self-hosted/compiler/lean_single.sio`.
Self-hosted; requires `lean_single_fixed_point_gate.sh` +
`native_v2_cpu_compiler_umbrella_gate.sh` PASS.
**Priority.** P4 — unblocks R.2.7's third caller (rng.sio) wiring and
makes the LOC floor (§7.8 of R.2.8 DISPATCH, ≤−40) reachable. No
correctness regression in current stdlib/PBPK; pure typechecker noise.
**Branch.** `sounio-pure/r2-1-park-miller`.
**Time budget.** 3–5h single session.

---

## §0 — Sounio-Pure constraint

May read/write `self-hosted/compiler/lean_single.sio`, rebuild
`bin/souc-linux-x86_64`, run scripts/ci gates, write probes under
`docs/audit/r2_9_var_tuple_dot_zero_typecheck/reference/`. No Python /
R / external tools.

Bootstrap chain (boot3/boot4/native-v2) is **out of scope**. If the
patch would require touching it — HALT and report.

---

## §1 — Observed failure surface

The pattern

```sounio
var sm = splitmix64_new(seed)        // sm : SplitMix64 (1-field struct)
let r1 = splitmix64_next(sm)         // r1 : (SplitMix64, i64)
sm = r1.0                            // ← "assignment type mismatch"
```

fails typecheck at the assignment line when:

1. `var sm` is initialised from a function call (not a struct literal).
2. The function returns a 1-field-struct + i64 tuple.
3. The assignment RHS is the tuple's `.0` field.
4. A `type X = Y` alias exists *anywhere* in the module being checked.

Conditions 1–3 alone fail standalone (`reference/standalone_repro.sio`),
but the failure is *masked* in certain compositions of an imported
module — R.2.8's pre-Phase-D rng.sio passed an oracle-gen importer
despite containing the pattern; post-Phase-D (which added
`type Pcg64 = PcgState`) it does not.

Three reproducers at `reference/`:

| File | Form | Result |
|---|---|---|
| `standalone_repro.sio` | single file, no alias, no import | FAIL (proves pattern is inherently fragile) |
| `imported_repro_{main,lib}.sio` | importer + lib, alias in importer | FAIL |
| (R.2.8 evidence) pre-Phase-D rng.sio | importer (oracle_gen) of rng.sio with no alias added | PASS — masking case |

---

## §2 — Hypothesis

When a type alias is registered, Pass 1c.5 (added by R.2.7
`ce9810ee9`) re-encodes tuple-return hashes for **every** function in
the module, not just those whose return type references an alias.
The re-encode produces a hash that *matches* the original scan_type
output for the alias case (idempotent) but **subtly diverges** for
non-alias 1-field-struct + i64 tuples — possibly via:

- (Ha) `tcount_enc2 = re_ret_hash / 1000000` losing precision when
  the original scan_type hash uses 100M format with non-zero
  `first_f64`/`last_f64` bits.
- (Hb) `tup_cache_register` is idempotent on the hash *key* but
  Pass 1c.5 also calls `scan_type(rp_tok + 1)` and `scan_type(second_q2)`
  which may register a new TUP_CACHE entry with a *different* hash than
  the one Pass 1 originally registered, causing later `.0` lookups to
  hit the wrong entry.
- (Hc) `FN_RET_HASH[rp_i] = re_ret_hash` overwrites the original
  hash with a slightly different value, and downstream assignment
  comparison (one of the 10 sites at lines 17015, 17194, 17333, 18002,
  18051, 18111, 18260, 18309, 29160, 29202) reads `FN_RET_HASH` directly
  instead of via tup_cache lookup.

The masking case (pre-Phase-D rng.sio with no alias) supports
hypothesis space H*: the bug only fires when Pass 1c.5 runs (gated by
`REF_TYPE_COUNT > 0`).

---

## §3 — Attack plan

### Phase A — Diagnose (90 min)

1. Use `standalone_repro.sio` as primary reproducer (no imports, no
   alias, no Pass 1c.5 — pure code path).
   - **Wait.** Standalone fails *without* alias. That contradicts the
     "Pass 1c.5 trigger" theory. Phase A first task: characterise
     the standalone failure independently. Compile it with debug
     prints in the assignment typecheck path (one of the 10 sites).
2. If standalone failure is *also* gated on some Pass 1c.5-adjacent
   condition (e.g. a different alias mechanism, or unrelated init-tracker
   regression), document. Otherwise the "Pass 1c.5 over-reach" framing
   is wrong and the bug is a pre-existing assignment-typecheck issue
   that R.2.7 incidentally surfaced.
3. Add `print_type_label` at the firing assignment site to dump
   `k_lhs, h_lhs, k_rhs, h_rhs` immediately before `ty_eq` (or the
   inline comparator).
4. Compare:
   - LHS slot type/hash recorded at `var sm = splitmix64_new(seed)`.
   - RHS expression type/hash for `r1.0`.
   - The corresponding `tup_cache` entry for `splitmix64_next`'s
     return hash.

Expected discovery: one of —
- (Da) LHS records hash from scan_type (100M format) while RHS
  field-access lookup uses the Pass 1 1M-format hash stored in
  `FN_RET_HASH`. Resolution: pick one format universally.
- (Db) `var x = fn_call(...)` records LHS slot count from a
  ty_slot_count call that diverges from how field-access derives the
  same number for `.0`.
- (Dc) The 1-field-struct case hits a degenerate path where
  `first_nslots == total_nslots/2` collides with the i64 second slot,
  causing the slot offset of `.0` to be miscalculated.

### Phase B — Patch (60–90 min)

Direction depends on Phase A outcome. Constraints from R.2.8:

- **DO NOT** generalise `ty_eq` more permissively. R.2.8's Bb proposal
  was wrong — re-encoded hashes are genuinely different, not a
  ty_eq permissiveness issue.
- **DO NOT** widen Pass 1c.5's scope. R.2.7's existing scope is
  load-bearing for distributions.sio + sampling.sio wiring.
- **DO** consider: idempotency check at the end of Pass 1c.5's
  re-encode — if `re_ret_hash == original FN_RET_HASH`, no-op.
- **DO** consider: gating Pass 1c.5 on "this fn's return token
  stream actually references an alias name registered in
  REF_TYPE_DATA".

### Phase C — Validation (30 min)

1. `lean_single_fixed_point_gate.sh`: PASS required.
2. `native_v2_cpu_compiler_umbrella_gate.sh`: 12/12 PASS required.
3. `standalone_repro.sio`: typechecks (was FAIL).
4. `imported_repro_main.sio`: typechecks.
5. R.2.4 distributions oracle: 1024/1024 bit-exact (regression check).
6. R.2.5 rng self-oracle: 1024/1024 bit-exact via R.2.8 reverted rng.sio.

### Phase D — Re-attempt R.2.8 Phase D for rng.sio (15 min)

With the typecheck fixed, retry `type Pcg64 = PcgState` in rng.sio.
Goal: ~−45 LOC additional reduction, pushing R.2.8 LOC delta past
the original −40 floor and flipping R.2.7 / R.2.8 to **RESOLVED**.

### Phase E — Commit + push (10 min)

Three commits:
1. `self-hosted/compiler/lean_single.sio` + `bin/souc-linux-x86_64` —
   the compiler patch.
2. `stdlib/random/rng.sio` — R.2.8's deferred rng.sio Phase D wiring.
3. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/` — DISPATCH +
   SYNTHESIS + reference probes.
4. `docs/audit/r2_8_alias_deep_resolve/SYNTHESIS.md` — flip PARTIAL → FULL.
5. `docs/audit/r2_7_pcg_state_unify/SYNTHESIS.md` — flip
   RESOLVED-PARTIAL → RESOLVED.

HALT for operator review before push.

---

## §4 — Out of scope

- Bootstrap chain edits (boot3/boot4/native-v2).
- Algorithmic changes to PCG64 / SplitMix64 / Xoshiro.
- Refinement-type predicate machinery in Pass 1c.5.
- Any rename in stdlib API surface.

---

## §5 — Halt conditions

- **Phase A diagnosis inconclusive after 90 min.** Surface findings;
  close R.2.9 as DEFERRED. The bug isn't blocking anything operational.
- **Patch breaks lean_single fixed-point.** Revert; reassess.
- **Patch causes any non-R.2.8 sub-gate regression in the umbrella.**
  Revert; the surgery has unintended scope.
- **Oracle replay fails one sample after Phase D.** rng.sio wiring
  caused stream drift — fix wiring, not the compiler.
- **Diagnosis reveals the fix needs to touch >50 lines of lean_single.sio
  outside Pass 1c.5 / assignment-typecheck.** HALT; this is a deeper
  refactor than R.2.9's budget. Defer to a multi-session R.3.x.

---

## §6 — Deliverables on close

1. `self-hosted/compiler/lean_single.sio` — the targeted patch.
2. `bin/souc-linux-x86_64` — rebuilt; new md5; size delta documented.
3. `stdlib/random/rng.sio` — `type Pcg64 = PcgState` + one-liner wrappers.
4. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/DISPATCH.md` — this file.
5. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/SYNTHESIS.md` —
   closing writeup with diagnosis location + patch diff.
6. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/reference/standalone_repro.sio` — pre-existing.
7. `docs/audit/r2_9_var_tuple_dot_zero_typecheck/reference/imported_repro_*.sio` — pre-existing.
8. `docs/audit/r2_8_alias_deep_resolve/SYNTHESIS.md` — PARTIAL → FULL.
9. `docs/audit/r2_7_pcg_state_unify/SYNTHESIS.md` — RESOLVED-PARTIAL → RESOLVED.

---

## §7 — Acceptance

R.2.9 is **VALIDATED** iff:

1. ✓ `lean_single_fixed_point_gate.sh`: PASS (stage1 == stage2 == stage3).
2. ✓ `native_v2_cpu_compiler_umbrella_gate.sh`: 12/12 PASS.
3. ✓ `standalone_repro.sio` typechecks (was FAIL).
4. ✓ `imported_repro_main.sio` typechecks (was FAIL).
5. ✓ R.2.4 distributions oracle: 1024/1024 bit-exact.
6. ✓ R.2.5 rng self-oracle: 1024/1024 bit-exact.
7. ✓ rng.sio Phase D wiring lands; R.2.7 + R.2.8 SYNTHESIS flipped to RESOLVED.
8. ✓ Net additional LOC delta in `stdlib/random/rng.sio`: ≤ −40.

If 1 or 2 fails: FAIL. Revert.
If 3 or 4 fails: FAIL. Diagnosis was wrong; rethink Phase B.
If 5 or 6 fails: FAIL. Stream drift; debug wiring before resuming.
If 7 or 8 PARTIAL: R.2.9 may still ship the compiler patch alone;
rng.sio stays inlined for another R.2.x cycle.

---

## §8 — Notes

- The failure surface in §1 is *narrow* (1-field-struct + i64 tuple +
  var-init-from-fn-return + .0-assignment). It is plausible that the
  fix is a 10–30 line change in one specific code path. The risk is
  that diagnosis takes longer than the fix.
- The standalone-failure observation (no alias, no Pass 1c.5) is the
  load-bearing puzzle. If standalone fails *without* Pass 1c.5, the
  bug is pre-existing and R.2.7's patch merely exposed it in
  composition. If standalone fails *because* of some adjacent pass
  that always runs, R.2.9's fix is more central.
- Either way, this is the last known blocker on R.2.7 Path A's full
  realisation (rng.sio caller wiring), and it cleanly retires the
  RESOLVED-PARTIAL tail on R.2.7 + R.2.8.

**END OF DISPATCH.**
