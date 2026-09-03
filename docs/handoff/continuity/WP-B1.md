<!-- docs:meta
topic_id: repo.docs.handoff.continuity.wp-b1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.wp-b1
-->

# WP-B1 — EISA: dependency-closure `str_from_bytes` (ud2 → SIGILL) [Opus] (no deps)

## Ownership protocol (FIRST)
This item was assigned to **Codex** in `artifacts/omega/agent_handoff.log.md` ("YOURS: str_from_bytes absent from the test's dep closure"). Before claiming: `grep -n -i 'str_from_bytes\|closure' artifacts/omega/agent_handoff.log.md | tail` — if Codex has a live CLAIM or landed a fix, coordinate/skip. If free, post a CLAIM citing the user's reassignment to the continuity campaign.

## Problem
The LAST open technical blocker of EISA default-lane parity. `tests/stdlib/eisa/test_eisa_isa.sio` imports `use eisa::core::*` + `use eisa::isa::*` but NOT `str::lib`. `stdlib/eisa/isa.sio:5` does `use str::lib::*` and calls `str_to_string` → the builtin wrapper `str_from_bytes` (`stdlib/str/lib.sio:87`). The compiler's module dependency-closure pulls the test's DIRECT imports but does not transitively pull `str::lib` definitions that `eisa::isa` itself needs → the missing body is emitted as a deliberate `ud2` → SIGILL at runtime. The SAME defect kills the runtime of `tests/stdlib/eisa/test_eisa_evm.sio` (its former ~24GB compile-vmem problem was already fixed on main by `a096d1c4b` — 25.8→6.5GB; no `MADAROS_VMEM_LIMIT_KB` juggling needed anymore).

## Where
`self-hosted/compiler/module_loader.sio` + `self-hosted/compiler/module_frontend.sio` — the module merge/closure sites (commit `a5e19cd8a` "imported struct chains fn_id merge" touched them; read that diff first: `git show a5e19cd8a -- self-hosted/compiler/`). The fix: when module M is pulled into the closure, M's OWN `use` deps (and their fn bodies) must be pulled transitively — or at minimum, fns of M that are reachable from the compiled program must have their callee closure satisfied instead of emitting ud2 stubs. Prefer the general transitive-closure fix; if the general fix explodes compile footprint, measure and report (the arena work in `a096d1c4b` should absorb it — verify with `/usr/bin/time -v`).

## Environment
Work in the EISA lane worktree `/workspace/sounio-eisa` (branch `gpu/epistemic-tensor-core-next`) BUT note its bundled `bin/madaros-linux-x86_64` is STALE. Two options: (a) implement the compiler fix on a fresh branch off MAIN (in a new worktree) where the compiler sources are current, and test EISA by copying the two test files + stdlib/eisa from the EISA worktree if they don't exist on main; (b) if stdlib/eisa exists on main, work entirely on main. Check `ls /workspace/sounio/stdlib/eisa/` first. Default-lane run command pattern:
```
SOUNIO_MADAROS_BIN=<your fresh madaros build> ./bin/souc run tests/stdlib/eisa/test_eisa_isa.sio
```

## Witnesses
W1: minimal 3-module repro you author FIRST (before touching the compiler): `mod_a.sio` uses `mod_b`; `mod_b.sio` uses `str::lib` and calls `str_to_string`; main imports only `mod_a`. Baseline → SIGILL/ud2; after fix → correct output. Keep it as a run-pass test.
W2: `test_eisa_isa.sio` on the default lane → PASS, no SIGILL (exact expected output: see the test's header/asserts).
W3: `test_eisa_evm.sio` on the default lane → compiles within default vmem guards, runs, PASS, no SIGILL.

## Validation battery
- Umbrella gate before/after: zero new reds (12/12 sub-gates unchanged or better).
- lean-lane EISA unaffected: `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/eisa/test_eisa_core.sio` (and isa) still PASS.
- Multi-module smoke: 8-10 run-pass tests with `use` chains, byte-identical vs pre-change build.
- CI-parity battery (the fix touches compiler files): selfhost_host_gate + souc_v2_gate + runtime proof.

## Done criteria
W1–W3 verified; battery green; PR merged to main; scoreboard + handoff (incl. the EISA lane scoreboard entry) updated.
