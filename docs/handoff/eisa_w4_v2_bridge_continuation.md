# W4 status — v2 (qd128) AOT bridge

Worktree `/workspace/sounio-eisa`, branch `gpu/epistemic-tensor-core-next`. All
commits **local, not pushed** (branch 8 ahead / 8 behind origin; hold for
operator direction).

## Done and gated — the full v2 arithmetic surface + flagship

The v2 (qd128) err lane now runs through the x86-64 AOT bridge **byte-identical
to the Metron VM**, including the Rump 1988 flagship. Full differential
conformance gate green (`scripts/ci/eisa_bridge_conformance_gate.sh`): 16 v0/v1
lanes unchanged, all v2 lanes below, plus tamper + anti-vacuity (v=3).

Commits (oldest → newest):
- `7467d716` micro-step 1 — v2 dispatch + econst/gate + receipt v3 (+ the
  `v2_strings_off` overlap fix).
- `af6d7dde` micro-step 2 — v2 `eadd` via emitted SSE2 qd subroutines.
- `20d3422a` micro-step 3 — `esub`.
- `5ed85f8b` micro-step 4 — `emul` (13-term box).
- `f5d03125` micro-step 5 — `ediv`.
- `02f0d652` micro-step 6 — `esqrt` (completes the 5 arithmetic ops).
- `d7332451` flagship — `v2-rump-qd` (version 2, qd128 reconstructs
  −54767/66192 across roundoff0..3 while the val lane carries the wrong-sign
  cancellation garbage) + `v1-rump-dd` (version-1 dd64 failure lane, decision
  #15). Enlarged the emission buffer 65536 → 131072 for the ~72KB Rump ELF, and
  fixed a `put32` byte-2 regression the resize introduced (the 2^16 divisor must
  stay 65536 — do NOT let a blind buffer-size replace touch it; it truncates
  p_filesz to total&0xFFFF for ELFs over 64KB → SIGSEGV at runtime).

Conformance lanes: `v2-const-gate v2-add v2-sub v2-mul v2-div v2-sqrt
v2-rump-qd v1-rump-dd`. Unit tests also green: `test_eisa_bridge` (X1–X5),
`test_eisa_bridge_v1` (Y1–Y9), `test_eisa_evm_v2` (W-A…W-H).

### How it is built (bridge_x86.sio)
- Emitted-once qd subroutines (~30KB, recorded as local offsets in
  `bridge_translate_v2`): `two_sum`/`quick_two_sum`/`two_prod` (register-only
  leaf EFTs), `qd_renorm5` (Priest Alg.9, branchy k-accumulator, b[k] via a
  rcx-scaled SIB store `st_sib`), `qd_double_accumulate`, `qd_components_sorted8`
  (unrolled insertion sort), `qd_add`, `qd_add_f64`, `dd_add`, `dd_add_f64`,
  `qd_three_sum`, `qd_six_three_sum`, `qd_nine_two_sum`, `qd_nine_one_sum`,
  `qd_mul`, `qd_mul_f64`, `qd_sub`, `qd_div`, `qd_sqrt`.
- Static disjoint `v2q` slot ABI (no recursion → sound). Slot map documented in
  the "v2 qd128 subroutines" comment block; leaf EFTs are register-only.
- Per-op bodies `em_v2_binop_addsub` / `em_v2_binop_mul` / `em_v2_binop_div` /
  `em_v2_sqrt` reproduce the core_v2 closure contract (val = fl64 op; err =
  qd_sub(qd_op(true(a),true(b)), qd(val)); u = verbatim from em_arith) + the I3
  poison re-check (`em_v2_finish_result`). Divisor-zero + sqrt(≤0) guards match
  the VM.

## Remaining (optional parity + landing)

1. **v2 control-flow lanes** — `ebrz`/`ebrn` (frail band = max(u,|e0|)), the
   v2 fuel-stop receipt, `eload`/`estore`/`emov`. Currently these opcodes on the
   v2 path return a loud `status 42` (never a silent miscompile). The Rump
   flagship does not use them (straight-line), so they are not on the critical
   path — add `v2-loop`/`v2-fuel`/`v2-poison`/`v2-frail` lanes reusing the v1
   machinery when full v2 parity is wanted. `em_v1_normalize_u_lane` /
   `em_v2_set_last_written` / the v1 branch+fuel patch lists are the pieces to
   generalise to the 48-byte lane.
2. **Full Slurm battery** — `bash slurm-jobs/eisa/submit-eisa-battery.sh` (then
   `<run-id>`), the pinned-node OrangeFS-safe validation, before landing.
3. **W5** — Rump receipt showcase in the positioning doc (now unblocked: the
   real v2-rump-qd bridge receipt exists), the `strings → grep -a` gate
   portability patch (W4 no longer collides), branch reconciliation + push.

## Verify
```
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"; export SOUNIO_SOUC_ENGINE=lean_single
export TMPDIR=/workspace/sounio-eisa/artifacts/tmp
bash scripts/ci/eisa_bridge_conformance_gate.sh          # all lanes byte-identical
./bin/souc run tests/stdlib/eisa/test_eisa_evm_v2.sio     # W-A..W-H
```

## Concurrency warning
This worktree was `git reset --hard`'d at 2026-07-06 09:37 by an external
workspace step (wiped uncommitted scaffolding once). Commit every green step.
Do not run a second agent on `/workspace/sounio-eisa`.
