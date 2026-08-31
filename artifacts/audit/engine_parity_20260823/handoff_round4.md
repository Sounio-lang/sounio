# Handoff to claude-1, round 4 — from qwen, 2026-08-23

Founder direction: Madaros becomes the only user-facing compiler; it must run
everything lean_single runs. This round builds the cutover instrument and
measures the gap properly. Read-only on the worktree; deliverables in /tmp.

## 1. The parity measurement (valid run)

First run was garbage: the pod's default 8MB stack makes the raw Madaros ELF
SEGV on trivial programs (rc=139 on `fn main() -> i32 { 0 }`). Hazard posted:
`madaros-needs-big-stack`. With `ulimit -s 524288` throughout:

    corpus=1771  AGREE=731  DIVERGE=88  MADAROS-ONLY=108
    LEAN-ONLY=242  NEITHER=602  (jobs=8, timeout=120s)

Vs the 2026-08-10 baseline (LEAN-ONLY 268 / MADAROS-ONLY 358 / NEITHER 311):
- The cutover gap is LEAN-ONLY=242. Families: lorenz 13, closure 5, viz 4,
  madaros 3, then long tail. The big four (graphics/viz/lorenz/solver) are
  mostly gone — lorenz/solver moved to NEITHER under load (see caveat).
- NEITHER=602 is INFLATED: 274 of the 291 new NEITHERs are lorenz/solver, the
  family the gate's own header says flips verdict under parallel load. The
  corpus gate pins JOBS=4 for exactly this reason. Authoritative number needs
  JOBS=4 on Slurm. Raw logs: /tmp/qwen_parity_run3.log.

## 2. Wiring, ready to land

`/tmp/qwen_engine_parity_workflow.yml` — drop-in workflow, same shape as
madaros-corpus.yml: builds Madaros from PR source, runs
`scripts/ci/engine_parity_gate.sh` against the baseline. Both steps carry
`ulimit -s 524288` (the stack hazard). No separate ratchet needed — the
baseline IS the shrink-only ratchet: any new non-AGREE entry fails, any
improvement is reported by name. Wiring this removes engine_parity_gate.sh
from the unnamed-gate set (it is in there today, confirmed).

## 3. build_modular_madaros.sh is fail-open (real defect)

My from-source build printed TWO errors and still produced an ELF, rc=0:

    error: match must be exhaustive at self-hosted/resolve/imports.sio:384
    error: match must be exhaustive at self-hosted/check/check.sio:19937

The script only checks `[[ ! -s "$OUT" ]]` — never the compiler rc. A seed
that refuses the source still ships a binary. The errors are real and have
exact fixes (below); the script fix is one line: capture the rc from the
second souc-build-lock call and fail on nonzero.

## 4. The two missing ItemClaim arms (exact patches)

`ItemKind::ItemClaim` landed 2026-07-26 (65639f3b8a, on main). Two matches
never got the arm:

a) `self-hosted/resolve/imports.sio:383` — collect_module_exports_item.
   Claims are consumed by the parse driver (parser/mod.sio:40 drops them from
   the module item list), so the arm is a no-op, matching ItemImpl/ItemUse:

       ItemKind::ItemUnit => {}
   +   ItemKind::ItemClaim => {}

b) `self-hosted/check/check.sio:19936` — check_item. Same reasoning, matching
   the surrounding "already handled in collect pass" style:

       ItemKind::ItemOntology => self  // Already handled in collect pass
   +   ItemKind::ItemClaim => self  // Consumed by the parse driver (parser/mod.sio)

Both are on main too (verified via git show origin/main). Until fixed, every
from-source Madaros build prints two errors and pub claims inside modules are
silently absent from exports.

## 5. Cutover sequence (from the measured state)

1. Land the two ItemClaim arms + the build-script rc check (unblocks clean
   from-source builds — the corpus job on 8/21 failed for want of this class
   of fix).
2. Wire /tmp/qwen_engine_parity_workflow.yml. Baseline freezes LEAN-ONLY.
3. Re-measure on Slurm with JOBS=4 for the authoritative NEITHER split.
4. Burn LEAN-ONLY by family (closure 5 and viz 4 first — small, contained).
5. Triage DIVERGE=88 (wrong-answer class; needs per-program judgement).
6. When LEAN-ONLY=0 and DIVERGE triaged: flip full-test-suite's
   SOUNIO_TEST_SOUC_BIN from souc-stage2 (lean_single) to a from-source
   Madaros. lean_single stays only as the bootstrap seed.
