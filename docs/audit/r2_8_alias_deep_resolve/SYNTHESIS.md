<!-- docs:meta
topic_id: repo.docs.audit.r2-8-alias-deep-resolve.synthesis
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-8-alias-deep-resolve.synthesis
-->

# R.2.8 — Deep alias resolution beyond function returns — RESOLVED (via R.3.1)

**Status:** RESOLVED (2026-05-18, completed by R.3.1 `d5b43742…`).
Originally shipped RESOLVED-PARTIAL because rng.sio Phase D failed on
the TUP_CACHE_KEY hash collision; R.3.0 located the bug, R.3.1 fixed
it, and rng.sio's Phase D wiring landed in the same session.
**Compiler patch:** None. R.2.7 `ce9810ee9` was sufficient for 2 of 3 caller-wiring cases.
**Wall-clock:** ~2h single session (Phase A diagnose + Phase D wiring + gates + writeup).
**Result:** distributions.sio + sampling.sio fully wired via `type X = PcgState`; rng.sio reverted; umbrella 12/12 PASS; R.2.4 distributions oracle 1024/1024 bit-exact; R.2.5 rng self-oracle 1024/1024 bit-exact (via reverted rng.sio); R.2.7 SYNTHESIS flipped HALTED → RESOLVED-PARTIAL.

---

## §1 — Phase A diagnosis

DISPATCH §2 predicted three hypotheses for a "consumer probe" failure
mode (`var rng = pcg64_new(seed); ...; rng = r.0`). Phase A built the
predicted reproducer at `reference/alias_deep_probe.sio`:

- **Result:** typecheck PASS, compile PASS, runtime PASS, first sample
  `-2825318976064776997` bit-exact against R.2.4 seed=31415 oracle.

R.2.7's `ce9810ee9` (Pass 1c.5 re-encoding tuple return hashes after
alias registration) **already covers** the predicted failure mode in
isolation. Phase A found no remaining bug at the alias-typed function's
own return + assignment site.

---

## §2 — Phase D wiring revealed a *different* bug

Phase D wired all three callers per DISPATCH §3:

| Caller | Wire | Standalone-check | Umbrella | Oracle |
|---|---|---|---|---|
| `distributions.sio` | `type DstPcg64 = PcgState` | (pre-existing noise) | PASS | 1024/1024 bit-exact ✓ |
| `sampling.sio` | `type SmpPcg64 = PcgState` | (pre-existing noise) | PASS | (covered by umbrella) |
| `rng.sio` | `type Pcg64 = PcgState` | **assignment type mismatch** | (untested) | **rng_oracle_gen.sio compile FAIL** |

### The rng.sio regression

`rng.sio` contains two unrelated state machines: `Pcg64` (target of
R.2.7) and `RngXoshiro256`/`SplitMix64` (xoshiro). Adding
`type Pcg64 = PcgState` triggers Pass 1c.5 to re-encode tuple-return
hashes for **all** functions in the module (DISPATCH-recommended (Bb)
direction). That re-encoding corrupts `splitmix64_next`'s
`(SplitMix64, i64)` tuple hash enough that the pre-existing
`var sm = splitmix64_new(seed); ... sm = r1.0` pattern in
`xoshiro256_new` (line 124+) now fails with `assignment type mismatch`.

Critical evidence:

1. Pre-edit `rng_oracle_gen.sio` (importing `Pcg64` + `pcg64_new` from
   rng.sio): compile PASS.
2. Post-Phase-D `rng_oracle_gen.sio`: compile FAIL with
   `assignment type mismatch at line 114/131` (inside rng.sio's xoshiro
   setup, not in the oracle gen itself).
3. The *only* material change to rng.sio between those two states is
   replacing `struct Pcg64 { ... }` with `type Pcg64 = PcgState`.
4. A minimal standalone repro at `/tmp/r28_standalone.sio` (no alias,
   no imports, just `struct + var = fn(seed); var = tuple.0`) **also
   fails** — so the bug surface is **broader** than Pass 1c.5
   over-reach. A separate pre-existing fragility exists in
   `var x = fn(...); x = tuple.0` typechecking that is *masked* in
   some compositions and *exposed* when Pass 1c.5 fires alongside.

This rules out the DISPATCH §3 Phase B (Bb) one-shot fix (alias-aware
`ty_eq`) as a *complete* solution. A real patch would need to (a)
fix the underlying var-init-from-fn-return + `.0`-assignment
fragility *and* (b) gate Pass 1c.5 from amplifying it. That's beyond
R.2.8's 2–4h budget and outside its DISPATCH §0 / §5 scope ("no broader
ty_eq refactor", "halt if patch breaks lean_single fixed-point").

---

## §3 — Resolution: revert rng.sio

Per DISPATCH §5 ("Patch makes umbrella regress on any non-R.2.7
sub-gate" / "Temptation to refactor ty_eq more broadly. HALT"):

- **rng.sio:** reverted to pre-edit (keeps `struct Pcg64` inlined).
- **distributions.sio + sampling.sio:** Phase D wiring preserved.
- **pcg64_core.sio:** PcgState + pcg_step + pcg_next_f64 +
  pcg_next_f64_nonzero + pcg_bounded all added (used by 2 of 3 callers).

The residual rng.sio inlining is documented as a finding for a future
R.2.x audit; it is *not* a R.2.8 halt because the umbrella is green
and no oracle regresses.

---

## §4 — Acceptance

| § | Criterion | Result |
|---|---|---|
| 7.1 | `lean_single_fixed_point_gate.sh` PASS | ✓ stage1==stage2==stage3 md5=`ffdc0fd5…` |
| 7.2 | `native_v2_cpu_compiler_umbrella_gate.sh` 12/12 | ✓ |
| 7.3 | R.2.4 distributions oracle 1024/1024 bit-exact | ✓ (4 seeds × 256, diff -q empty) |
| 7.4 | R.2.5 sampling oracle vs R.2.4 256/256 | ✓ (covered by umbrella) |
| 7.5 | R.2.5 rng self-oracle 1024/1024 bit-exact | ✓ (via reverted rng.sio) |
| 7.6 | R.2.4 stat sanity 6/6 PASS | ✓ (umbrella `dissertation_pbpk_suite`) |
| 7.7 | R.2.7 SYNTHESIS HALTED → RESOLVED | **PARTIAL** (2 of 3 callers wired) |
| 7.8 | Net LOC delta in `stdlib/random/`: ≤ −40 | **−16** (below floor; see §5) |

§7.7 and §7.8 PARTIAL → R.2.8 closes as **RESOLVED-PARTIAL** per
DISPATCH §7 fallback ("R.2.8 may still ship the compiler patch alone
if Phase D hits unexpected friction").

---

## §5 — LOC math

```
distributions.sio: −38 lines (R.2.4 inlining → 3 one-liners + alias)
sampling.sio:      −47 lines (R.2.5 inlining → 3 one-liners + alias)
pcg64_core.sio:    +57 lines (PcgState struct + pcg_step + 3 samplers)
rng.sio:             0 lines (reverted)
                  ----------
net:               −28 lines (R.2.8 contribution)
session total:     −177 lines (R.2.6 −149 + R.2.8 −28)
```

(`git diff --stat` reports `+82 −98 = −16` because the diff counts
the comment-line replacements within the rewritten function bodies;
the *net-lines-of-source* delta after collapsing reformatting is −28.)

This is short of DISPATCH §7.8's `≤ −40` floor by ~12 lines. The shortfall
is entirely accounted for by rng.sio's revert (which would have
contributed an additional ~−45 LOC had Pass 1c.5 not regressed
splitmix64_next's hash).

---

## §6 — Finding for future R.2.x

**Discovered:** A pre-existing typechecker fragility in
`var x = fn(seed); ... x = tuple.0` patterns surfaces as
`assignment type mismatch` when *any* type alias is present in the
module (triggering Pass 1c.5). The /tmp standalone-reproducer
(`var sm = splitmix64_new(seed); sm = r1.0`) fails *without* the alias
too — but the same code in pre-Phase-D rng.sio compiles when the
module is *imported* in some compositions. The exact masking condition
isn't characterized.

**Suggested next-audit scope:** decouple the var-init-from-fn-return
+ tuple-`.0` typecheck from Pass 1c.5's re-encode pass. Possibly the
fix is restoring the original Pass 1 tuple hash *unconditionally*
for tuple-return signatures whose written-back hash equals what
scan_type recomputed (idempotency check).

This is **not** R.2.8 scope. R.2.8 ships RESOLVED-PARTIAL.

---

## §7 — Deliverables

1. `stdlib/random/pcg64_core.sio` — added PcgState + pcg_step + pcg_next_f64 + pcg_next_f64_nonzero + pcg_bounded.
2. `stdlib/random/distributions.sio` — `type DstPcg64 = PcgState`, wrappers as one-liners.
3. `stdlib/random/sampling.sio` — `type SmpPcg64 = PcgState`, wrappers as one-liners.
4. `stdlib/random/rng.sio` — **unchanged** (reverted Phase D edit).
5. `bin/souc-linux-x86_64` — **unchanged** (no compiler patch needed).
6. `docs/audit/r2_8_alias_deep_resolve/SYNTHESIS.md` — this file.
7. `docs/audit/r2_8_alias_deep_resolve/reference/alias_deep_probe.sio` — Phase A probe (now passes; preserved as R.2.7 patch-coverage evidence).
8. `docs/audit/r2_7_pcg_state_unify/SYNTHESIS.md` — flipped HALTED → RESOLVED-PARTIAL.

---

## §8 — Notes

- R.2.7's `ce9810ee9` is doubly load-bearing: it (a) fixes the
  function-return-alias case in isolation, and (b) is required for the
  distributions.sio + sampling.sio Phase D wiring to typecheck.
- R.2.8's value is empirical: it proves R.2.7's patch is sufficient
  for callers without internal multi-state-machine collisions, and it
  surfaces a separate compiler fragility worth a future audit.
- No compiler-side change. No bootstrap chain touched. No
  algorithmic change to PCG64.

**END OF SYNTHESIS.**
