# Alignment: nested-store codegen fix ↔ G1 lane (g1/e008-bridge-fix) — 2026-06-03

Reconciles this branch (`codegen/deref-nested-store`, the *mut/native-codegen lane)
with the G1 frontend lane (`g1/e008-bridge-fix`, base of this branch). Comparison is
of committed state (G1 still at 4bab1996a = this branch's base; my fix is the delta).

## 1. Root cause — FULLY AGREED

Both lanes independently reached the identical root cause:
**two-level nested `*mut` field writes don't persist** — the in-place collect's
`(*c).fn_sigs.entries[i] = sig` / `(*c).fn_sigs.count = i+1` store into a discarded
copy, so `fn_sigs` stays empty → `find()=-1` → `current_return_type` unset →
spurious E008 (×122) + E170 (×27) + a silent body-type hole.
(G1: docs/audit/g1_wip/E008_ROOTCAUSE_NESTED_MUT_WRITE_2026-06-02.md. Mine:
NESTED_DEREF_STORE_FIX_2026-06-02.md.)

## 2. This branch DELIVERS the fix the G1 doc explicitly requested

The G1 root-cause doc says verbatim: *"The proper fix is a **codegen fix for nested
`*mut` field-write persistence** (so the cheap manual writes work) — this is the
`*mut`/native lane's domain."* That is exactly this branch: commits 6d8326d37 +
0f3628957 in lean_single.sio make `(*p).f1.f2[=/[]` and `p.f1.f2[=/[]` persist, so
check.sio's existing nested writes work **with no source change**.

→ **The G1 source patches (`fn_sigs_e008_*.patch`) are SUPERSEDED** and should NOT
land — G1 itself flagged them "do NOT land on the gate." This codegen fix replaces
them as the canonical E008 resolution.

## 3. Equivalence PROVEN — identical corpus census

G1's source patch and this codegen fix produce the **same** end state on the 504
run-pass corpus (both under 1 GB stack):

| | PASS | FAIL | CRASH | E008 |
|---|---:|---:|---:|---:|
| baseline | 125 | 376 | 3 | 122 |
| G1 source patch | **112** | 222 | **170** | 28 |
| this codegen fix | **112** | 222 | **170** | ~28–67* |

(*E008 prog-count differs only by counting method — any-E008 vs first-error.)
Identical PASS/FAIL/CRASH (112/222/170) confirms the two fixes are semantically
equivalent; the codegen one is the correct mechanism (no source restructuring, no
net-negative *source* diff to carry).

## 4. Crasher hypothesis — REFINED by this lane's measurement

G1's doc offered a "useful reframe": *"one codegen fix (nested-write persistence +
large-aggregate `*mut` copy) likely unblocks BOTH the E008 lever and most of the 170
crashes."* This lane's A/B measurement **partially refutes** the single-fix framing:
- The **nested-write half alone** (this fix) clears E008 but leaves the **170
  crashers untouched** — 0 of the 3 (big-stack) baseline crashers move; the fixed
  compiler produces the identical 170. So nested-write persistence is NOT what
  unblocks the crashers.
- The crashers are the **other half** G1 hypothesised (large-aggregate / by-value
  copy). gdb on the dominant crasher (131/170 at one instruction 0x4c2805b) pins it:
  a **by-value Checker method** (672 KB frame, arg0 = 164 KB Checker copy) crashes
  copying its **4th, 16-byte struct argument from address −1** (a find()-miss
  sentinel used as a by-value-arg source). This matches G1's own commits
  `9e72844b1` ("gdb the arg-checker crash — large-by-value-TypeEntry-arg codegen
  miscompile") and `7f8c4dac8` ("sig.params lead REFUTED") — **both lanes converged**
  on the same separate bug.

→ Net: **two distinct codegen bugs**, not one. This branch fixes #1 (nested write).
#2 (large by-value struct-argument from a −1 sentinel) is the shared next lane.

## 5. G1's NOTE resolves a contradiction this lane hit

While root-causing the crasher I found check.sio:15304 ("ExprCall never sets
e.right") contradicting the by-value-bridge path I'd inferred. G1's doc has the
answer: *"source read ≠ execution path here … do not trust the by-value `check_*`
source as 'what runs' without instrumenting."* So the dispatch source is NOT
authoritative for what executes — which is why my source-level path inference and my
isolated bootstrap repros didn't line up. The crasher needs instrumented-build /
symbol-level isolation, not source reading.

## Recommended joint plan

1. **Adopt this branch's codegen fix as the E008 resolution**; drop the G1 source
   patches (`fn_sigs_e008_*.patch`).
2. **Sequencing caveat:** rebuilding bin/souc with this fix makes check.sio's nested
   writes persist → the corpus goes net-negative (112/222/170) *until* bug #2 is
   fixed, exactly as the source patch did. The fix is still correct codegen; land it
   for correctness, but do not expect a green-corpus jump until the crasher lane
   lands too. (Don't gate-block on the temporary net-negative.)
3. **Open shared lane for bug #2**: large by-value struct-argument-from-(−1) in the
   in-place check pass. Both lanes' RC agree; fingerprint in
   STRUCT_RETURN_FIX_ATTEMPT_2026-06-02.md. Needs symbol/instrumented build.
