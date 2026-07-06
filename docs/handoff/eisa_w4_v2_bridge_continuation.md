# W4 continuation — v2 (qd128) AOT bridge

Status as of this handoff. Worktree `/workspace/sounio-eisa`, branch
`gpu/epistemic-tensor-core-next`. All commits are **local, not pushed** (the
branch is 8 ahead / 8 behind origin; hold for operator direction).

## Done and gated (commit `7467d7160`)

Micro-step 1 (`v2-const-gate`) is **green and byte-identical to the Metron VM**,
with all 16 v0/v1 conformance lanes unchanged. This proves the whole high-risk
plumbing surface with no qd arithmetic yet:

- `stdlib/eisa/bridge_x86.sio`:
  - `struct StrsV2` + v2 offset helpers (`v2_lanes`, `v2_rp2`, `v2_mlanes`,
    `v2_mp2`, `v2q`, `v2_reg_*_off`, `v2_mem_*_off`). **6 contiguous f64 per
    register (val,e0,e1,e2,e3,u = 48 B)** at `v2_lanes(r)=2048+r*48`; qd scratch
    `v2q(i)=7424+i*8` (224 slots → state top 9216). `v2_strings_off()=9472`
    (128+9216+128 slack) — the string blob can no longer overlap the state
    region (the earlier scaffolding put it at 9216, **inside** state; that was
    the layout-migration SEGV class the arch doc §5/§8.2 warns about).
  - `em_v2_set_poison`, `em_v2_normalize` (I3 over all four err components),
    `em_v2_set_last_written` (6-word snapshot into sc(36),(55),(56..61)),
    `em_v2_econst`, `em_v2_emit_receipt`, `em_v2_emit_fuel_stop_receipt`,
    `build_strings_v2`.
  - `bridge_translate_v2` + `img.w[1]==eisax_version_v2()` dispatch. **Arithmetic
    and control-flow opcodes on the v2 path return `status 42`** (loud, never a
    half-written template) — this is the growth point.
- `tools/eisa/eisa_evm_run.sio` + `tools/eisa/eisa_bridge_emit.sio`:
  `v2_const_gate_img()` (`econst e3=7.25; gate e3; ehalt`), wired into both mains
  at the tail (order must stay in lockstep — the gate diffs concatenated stdout).
- `scripts/ci/eisa_bridge_conformance_gate.sh`: `v2-const-gate` in `programs=`,
  anti-vacuity prefix `v2-*) expected_prefix="v=3 prog="`.

VM oracle line to reproduce for const-gate:
`eisa-receipt: v=3 prog=86490366849337713 gate=1 reg=e3 val=s0e1025m3659174697238528 roundoff0=s0e0m0 roundoff1=s0e0m0 roundoff2=s0e0m0 roundoff3=s0e0m0 u=s0e0m0 poisoned=0 frail=0`

## Remaining: the qd arithmetic (micro-steps v2-add … v2-rump)

### The op semantics to reproduce (core_v2.sio, byte-oracle)

`true(x) = qd_add(qd_from_f64(x.val), x.err)` (NOT `qd_add_f64` — match the
oracle's exact call). Per op:
```
val_z = fl64(x.val op y.val)              # one SSE op — copy from em_arith
t     = qd_op(true(x), true(y))           # full qd128
err_z = qd_sub(t, qd_from_f64(val_z))     # 4-word result -> e0..e3
u_z   = <verbatim the u-lane code from em_arith (bridge lines ~700-1100)>
```
The **val and u lanes are byte-identical to the existing v0/v1 `em_arith`** —
reuse that machine code; only the err lane changes to qd calls. eadd u=
`sqrt(xu²+yu²)`, emul u=`sqrt((yv·xu)²+(xv·yu)²)`, ediv u=`(1/|yv|)·sqrt(xu²+(q·yu)²)`,
esqrt u=`xu/(2z)` (see core_v2.sio lines 74-131, all match em_arith verbatim).

### qd primitives to emit once (record offsets in a `SrOffs`), in dep order

Port line-by-line from `stdlib/math/dd64.sio` / `stdlib/math/qd128.sio` (RN, no
FMA — the SSE encoders already satisfy this; NEVER introduce `mulsd`+`addsd`
fusion). Reference algorithm locations:

| sub | source | notes |
|---|---|---|
| `two_sum` | dd64.sio:61 | 6-flop; **register-only** (a=xmm0,b=xmm1 → hi=xmm0,lo=xmm1, scratch xmm2..5) is cleanest — it's a leaf |
| `quick_two_sum` | dd64.sio:70 | 3-flop; register-only, leaf |
| `dd_split` | dd64.sio:78 | splitter 134217729.0 (c_split()); leaf |
| `two_prod` | dd64.sio:86 | calls dd_split ×2; 17-flop |
| `dd_add` | dd64.sio:100 | for nine_two_sum |
| `dd_add_f64` | dd64.sio:110 | |
| `qd_renorm5` | qd128.sio:79 | **hardest**: branchy k-accumulator; the `if r.lo!=0` cascade + the `k<4` residual-discard rule (lines 145-154). Keep live f64s in private slots, k in rcx (register-only two_sum/qs don't clobber rcx). |
| `qd_add_f64` | qd128.sio:163 | 4×two_sum + renorm5 |
| `qd_double_accumulate` | qd128.sio:171 | for qd_add |
| `qd_components_sorted8` | qd128.sio:189 | **2nd hardest**: deterministic insertion sort by descending magnitude over an 8-slot array; `qd_mag` = clear sign bit (andpd mask or the em_v1_abs_copy trick, bridge line ~1204). Magnitude compare can use unsigned integer bit-pattern order after masking sign. |
| `qd_add` | qd128.sio:223 | sorted8 + double_accumulate loop (k<4 && i<8) + renorm4(=renorm5 with a4=0) |
| `qd_neg` | qd128.sio:265 | flip sign bit on 4 words (pxor sign mask) |
| `qd_sub` | qd128.sio:269 | `qd_add(a, qd_neg(b))` |
| `qd_mul_f64` | qd128.sio:273 | 3×two_prod + a3*b + 7×qd_add_f64 |
| `qd_three_sum` | qd128.sio:290 | |
| `qd_six_three_sum` | qd128.sio:297 | |
| `qd_nine_two_sum` | qd128.sio:310 | uses dd_add / dd_add_f64 |
| `qd_nine_one_sum` | qd128.sio:320 | plain 8-add chain |
| `qd_mul` | qd128.sio:326 | 13-term box |
| `qd_div` | qd128.sio:364 | 5 quotient terms; each `q_i=r.x0/b.x0` then `r=qd_sub(r, qd_mul_f64(b,q_i))`; final renorm5. Divisor-zero (b.x0==0 → the pre-op `x.val op y.val` already poisons via em_v2_binop guard). |
| `qd_sqrt` | qd128.sio:396 | seed `1.0/qd128_sqrt_f64(a.x0)` (reuse em_newton `ns`); 3 Newton iters; a.x0<=0 → zero |

### Subroutine-call ABI (avoids the aliasing-SEGV class)

- `rbx` = state base for the whole program (unchanged inside subroutines, so
  `v2q(i)` rbx-relative addressing works in callees). `call`/`ret` via the
  machine stack; no recursion ⇒ a **static** slot allocation is sound.
- Leaf EFTs (`two_sum`, `quick_two_sum`, `dd_split`) are cheapest **register-only**
  (args/results in xmm0/xmm1, scratch xmm2..7) — they touch no slots, so callers
  keep their live set in slots and only marshal two operands through xmm0/1.
- Composite subroutines pass qd operands/results through **fixed, disjoint** slot
  groups (e.g. AQ=v2q(0..3), BQ=v2q(4..7), RQ=v2q(8..11), FB=v2q(12)); give each
  composite its OWN private scratch window (renorm5: v2q(20..45); qd_add:
  v2q(50..75); etc. — 224 slots are ample). Document the slot map in a comment
  block like the v0 `sc()` map (bridge lines ~34, ~160). Keep the u-lane scratch
  in low `sc(<32)`; counters `sc(32..36)` and the 6-word snapshot `sc(55..61)`
  are RESERVED — op bodies must never touch them.

### Per-op templates + dispatch

- `em_v2_binop(op,dst,a,b)`: `em_v2_normalize(a)`, `em_v2_normalize(b)`, combine
  bad → poison; ediv divisor-zero guard (`ucomisd` on b.val==0 → poison); build
  `true(a)`,`true(b)` (2×`qd_add`), `call qd_op`, `qd_sub(t, qd_from_f64(val_z))`;
  val block = one SSE op copied from em_arith; u block = verbatim from em_arith;
  then a `finish`-style 6-word write with I3 poison. Mirror the v1 `em_binop`
  (bridge ~924) structure.
- `em_v2_esqrt`, and `eload`/`estore`/`emov` (copy all 6 words incl. e2/e3, then
  `em_v1_normalize_u_lane` + `em_v2_set_last_written`). Branches/fuel/frail reuse
  the v1 machine unchanged (frail band = max(u,|e0|), same as v1).
- In `bridge_translate_v2`, replace the `status 42` fall-through branch by branch
  with the new templates, one opcode per micro-step.

### Micro-step order (add golden program, gate green, `sha256sum artifacts/eisa/*.eisax.elf`, then next)

1. `v2-add` (W-A shape: `econst;econst;eadd;gate`) — first qd math; debug
   `qd_add`/`renorm5`/`sorted8` here in isolation.
2. `v2-sub` → 3. `v2-mul` → 4. `v2-div` (+zero-divisor) → 5. `v2-sqrt` (+x<=0).
6. `v2-poison`/`v2-frail`/`v2-fuel` (reuse v1 machine; verify v3 fuel-stop + `lw`).
7. `v2-rump-qd` (flagship: full Rump 1988 at version 2 — val carries wrong-sign
   f64 garbage, err qd128 reconstructs −54767/66192) + `v2-rump-dd` (the mandated
   dd64 failure lane, decision #15 — a version-1 image that already runs through
   the v1 path; only wiring). Both from the W-H builder in
   `tests/stdlib/eisa/test_eisa_evm_v2.sio`.

### Verify (from repo root)
```
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"; export SOUNIO_SOUC_ENGINE=lean_single
export TMPDIR=/workspace/sounio-eisa/artifacts/tmp
./bin/souc run tools/eisa/eisa_evm_run.sio | sed -n '/v=3/p'   # oracle lines
bash scripts/ci/eisa_bridge_conformance_gate.sh                 # differential gate
bash slurm-jobs/eisa/submit-eisa-battery.sh                     # full battery before landing
```

### Concurrency warning
This worktree was `git reset --hard`'d at 2026-07-06 09:37 by an external
workspace step, wiping the previous (uncommitted) scaffolding. **Commit every
green micro-step locally** so a reset cannot destroy work again. Do not run a
second agent on `/workspace/sounio-eisa`.
