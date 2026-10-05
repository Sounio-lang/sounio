# Handoff to claude-1, round 5 — from qwen, 2026-08-23

Cutover gap decomposed to root causes. Read-only on the worktree; fresh
from-source Madaros built at /tmp/qwen_madaros_fresh.elf (with the stack
wrapper /tmp/qwen_madaros_bigstack), gate-faithful measurement
(ulimit -v 8000000 like the gate's worker).

## 1. The parity numbers are stable, and the NEITHER inflation is NOT parallelism

JOBS=4 and JOBS=8 produced identical verdicts (agree=731 diverge=88
madaros-only=108 lean-only=242 neither=602). The 291 new NEITHERs are the
E035 Mod migration: 11 of 12 sampled fail with "effect not declared
(missing: Mod)" — exactly your e035-migration-in-flight hazard. The baseline
was cut 2026-08-10, eleven days BEFORE Mod became id 29 (#2059, 08-21), so
every Mod-needing program moved category. CONSEQUENCE FOR WIRING: do not
wire /tmp/qwen_engine_parity_workflow.yml until the refinereturn migration
lands and the baseline is refreshed with --update-baseline, or the gate
fails every PR on migration churn, not regressions.

## 2. LEAN-ONLY=39 decomposed (the actual cutover debt)

Gate-faithful (compile with ulimit -v, then run):

    compile_fail = 28    run_crash = 11    run_ok = 0

The 28 compile-fail split:
- 2 x parse_fail: the `on` reserved word (root cause below)
- 9 x lowering SEGV: lorenz_i256_*_imported family dies at
  "lower_array: seed_begin" — imported-module array lowering crash
- 17 x clean refusal: mostly your #1876 print-dispatch refusals
  (budget64_dof_sentinel_print, plot3d_test, rapamycin_iso_budget,
  rapamycin_rk4_budget — the exact five from your 08-22 bisect) plus the
  lorenz trajectory2 family and 2 viz receipts (one E035 missing Div/Panic)

The 11 run-crash: **5 closure codegen SEGVs** (closure_fn_ref,
closure_higher_order, closure_sort_by, closure_lambda_lift,
closure_effect_checked — all compile rc=0, SEGV at runtime; lean runs them
rc=0). Sprint-228 function-references feature is broken in Madaros codegen.
Plus 2 gpu, 2 test, 1 sobol, 1 madaros.

## 3. Root cause: `on` is a Madaros-only reserved word

The Madaros lexer hardcodes `on` as its own lexical class
(self-hosted/parser/parser.sio:649) for Contest syntax
`[m1, m2] on subject` (exprs.sio:421). lean_single has no such class. So
`let on = ...`, `fn f(on: i32)`, `if on {` parse-fail in Madaros, compile in
lean. Measured on BOTH the committed Aug-17 ELF and the fresh build; minimal
repro:

    fn f(on: i32) -> i32 { on }   -> madaros parse_failed=true, lean rc=0

Affected stdlib (on-as-identifier): graphics/surface.sio (2 sites),
graphics/text.sio (4), viz/ir.sio (4), theorem/cdcl.sio (6). surface.sio
cascades: canvas_ext imports surface -> renderer3d imports canvas_ext ->
viz/ir imports renderer3d -> every viz_* test blocked by one parse failure.

FIX (verified): rename `on` -> `is_on` in those 4 files (~16 sites).
Renamed surface.sio alone parses clean (rc=0) and compiles to ELF. This is
a stdlib rename, not a compiler change — but the language question deserves
a founder call: making `on` contextual (only after `]` in Contest position)
would keep the identifier legal and match lean_single.

## 4. Suggested order

1. Rename on->is_on in the 4 stdlib files (unblocks the graphics/viz/render
   chain — 2 of the 28 compile-fail plus the viz receipt programs).
2. Wait for refinereturn Mod migration to land, then --update-baseline and
   wire /tmp/qwen_engine_parity_workflow.yml.
3. Closure codegen SEGVs (5 programs) — the single biggest run-crash family;
   Sprint-228 fn-ref lowering is the suspect.
4. lorenz imported-module lowering SEGV (9 programs) — lower_array seed path.
5. #1876 print-dispatch refusals (your bisect already located the three
   missing ExprFieldAccess/global/if branches in lower.sio:10719).

Raw data: /tmp/qwen_parity_run3.log (jobs=8), /tmp/qwen_parity_run4.log
(jobs=4, identical), /tmp/lean_only_242.txt (the 39 new LEAN-ONLY paths).
