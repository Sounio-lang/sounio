<!-- docs:meta
topic_id: repo.docs.audit.madaros-f64-unary-neg-signed-zero-dispatch-2026-09-14
authority: repo_only
audience: users
last_validated: 2026-09-15
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-f64-unary-neg-signed-zero-dispatch-2026-09-14
-->

# Madaros lowers f64 unary minus as `0.0 - x` — dispatch

**Date:** 2026-09-14
**Base:** `origin/main` @ `4fa6999bcc` (worktree `/workspace/sounio-worktrees/madaros-fneg`, branch `fix/madaros-f64-unary-neg-signed-zero`)
**Engines:** default Madaros; cross-checked against `SOUNIO_SOUC_ENGINE=lean_single`
**Owner:** unassigned (self-hosted/native core_ir codegen + self-hosted/ir/opt_cleanup.sio)
**Status:** evidence recorded; source-built baseline **reproduces**; patched build **passes the witness** (2026-09-15). Full-suite A/B: see [Patched build](#patched-build-2026-09-15)

## Why this dispatch

Under Madaros, `-x` on an f64 with `x = +0.0` produces `+0.0`. IEEE 754 negation
flips the sign bit, so the answer is `-0.0`. lean_single already gets it right.
The two values compare equal under `==`, so ordinary value tests do not see the
difference. It shows up in the bit pattern, in `1.0 / -x`, in `atan2`, in branch
cuts, and in printing.

The repository already works around the symptom rather than recording it:
`docs/audit/ENGINE_PARITY_ADJUDICATION_2026-08-02.md` (`print_f64_negative`) and
several tests spell negative zero as `0.0 * (0.0 - 1.0)`. No audit doc records
that unary minus itself is wrong.

## Repro

```sounio
fn neg(x: f64) -> f64 { -x }
fn chk(tag: i64, got: i64, want: i64) -> i64 with IO { if got != want { print("BAD "); print(tag); print("\n"); return 1 } 0 }
fn main() -> i32 with IO, Mut, Div, Panic {
    let bad = chk(5, f64_to_bits(neg(0.0)), (0 - 9223372036854775807) - 1)
    if bad == 0 { print("OK\n") }
    0
}
```

The repro compares bits instead of printing them. Under Madaros,
`print(f64_to_bits(...))` fails to lower with "unresolved scalar kind", which is
a separate defect and not part of this dispatch.

| Compiler | Command | Output |
|---|---|---|
| committed `bin/madaros-linux-x86_64` (md5 `57c015c1…`) | `SOUNIO_SOUC_ENGINE=madaros ./bin/souc build repro.sio -o r.elf` | `BAD 5` |
| committed `bin/souc-lean-single-x86_64` (md5 `cfed784e…`) | `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile repro.sio -o r.elf` | `OK` |
| Madaros built from `4fa6999bcc` with `make build-madaros` (`artifacts/self-hosted/madaros`, md5 `85e199ed…`) | `artifacts/self-hosted/madaros build repro.sio r.elf` under `ulimit -s 524288` | `BAD 5` |
| Madaros built from `4fa6999bcc` **plus this fix** (md5 `882c55a4…`) | same | `OK` |

The first observation used the committed binary at tree `c6e1d7b09b`. The rows
above were re-measured at `4fa6999bcc`. The committed binary's md5 is the same
at both. The source-built row settles principle 15: the defect is in current
source, not only in a stale shipped ELF.

Running the source-built artifact **raw** with the default shell stack limit
segfaults (rc 139) inside `lower_array`, before any output. Raise it first
(`ulimit -s 524288`, as `scripts/dev/souc-build-remote.sh` does), or go through
`SOUNIO_SOUC_ENGINE=madaros ./bin/souc build … -o …`. Both routes gave identical
witness output.

## Witness

`tests/run-pass/f64_unary_neg_signed_zero_bits.sio` compares everything by bit
pattern. It builds its inputs without unary minus and checks them before use:
-0.0 is `0.0 * (0.0 - 1.0)`, and ±inf is `1.0 / 0.0` through a function. The
expected bits were computed independently with Python `struct.pack('<d', …)`.

| Tag | Expression | Expected | Committed Madaros | Source-built Madaros (base) | Source-built Madaros (patched) | lean_single |
|---|---|---|---|---|---|---|
| 1–4 | input preconditions (+0, -0, +inf, -inf) | as built | pass | pass | pass | pass |
| 10 | `neg(+0.0)` | -0.0 | **BAD** | **BAD** | pass | pass |
| 11 | `neg(-0.0)` | +0.0 | pass | pass | pass | pass |
| 12, 13 | `neg(+inf)`, `neg(-inf)` | ∓inf | pass | pass | pass | pass |
| 14, 15 | `neg(1.5)`, `neg(0.0 - 2.25)` | -1.5, 2.25 | pass | pass | pass | pass |
| 16 | `neg(neg(-0.0))` | -0.0 | **BAD** | **BAD** | pass | pass |
| 20 | `-c0`, `let c0 = 0.0` | -0.0 | **BAD** | **BAD** | pass | pass |
| 21–23 | `-c15`, `-c225`, `-(-c225)` | exact | pass | pass | pass | pass |
| 24, 25 | `-nz`, `-pinf` | +0.0, -inf | pass | pass | pass | pass |
| 26 | `-(0.0)` | -0.0 | **BAD** | **BAD** | pass | pass |
| 27 | `-(1.5)` | -1.5 | pass | pass | pass | pass |
| 28 | `-0.0` | -0.0 | **BAD** | **BAD** | pass | pass |
| 30–33 | `0.0 - x` at +0, -0, `c0 - c0`, `0.0 - c15` | +0, +0, +0, -1.5 | pass | pass | pass | pass |

Every failure is a negation whose operand is +0.0. Every other value negates
correctly. Tag 11 passes only by coincidence, because `0.0 - (-0.0)` happens to
equal `+0.0`. That pattern is exactly what `0.0 - x` produces, and nothing else
in the table explains it.

## Root cause (source)

**Lowering is correct.** `Lowerer::lower_unary_expr_ref`
(`self-hosted/ir/lower.sio`, the tail of the fn) emits
`IrUnaryOp(OpNeg, src)` and marks the destination as float when the operand is
float. Nothing at the IR level is a subtraction.

**Native core_ir codegen turns it into a subtraction.** In
`compile_ir_function_v2_core_ir_into`
(`self-hosted/native/codegen_x86_linux.sio`), the `IrUnaryOp` / `OpNeg` arm for
a float-marked or float-typed source emits:

```
load src -> rax ; movq xmm1, rax
mov rax, 0      ; movq xmm0, rax
subsd xmm0, xmm1          ; xmm0 = 0.0 - x
movq rax, xmm0  ; store
```

That is `0.0 - x`, which is +0.0 at x = +0.0 under round-to-nearest.

**Latent, same family: opt_cleanup treats float constants as integers.** Float
literals lower to `IrLoadImm(f64_to_bits(v))` plus a float marker
(`Lowerer::emit_f64_const`). opt_cleanup records every `IrLoadImm` as a known
constant whether or not it is marked float, so:

1. `ocp_const_fold_pass_a2` (Sprint 82 Block J) and `ocp_mfi_const_fold` fold
   `IrUnaryOp(OpNeg, const)` to `IrLoadImm(0 - bits)`. For a float that is
   two's-complement negation of the bit pattern, which is wrong for every value
   except those where it coincides. At bits = 0 it gives 0, which is +0.0. That
   accounts for tags 20, 26 and 28 whether they reach codegen or the fold. The
   witness cannot tell those two paths apart from outside.
2. `ocp_const_fold_pass_b` Block CD rewrites `IrBinOp(OpSub, 0, x)` into
   `IrUnaryOp(OpNeg, x)`. For floats, `0.0 - x` and `-x` are **different
   functions**: they disagree at x = +0.0. Today the rewrite is invisible only
   because codegen implements neg as that same subtraction. **Fixing codegen
   alone would turn Block CD into a new miscompile of `0.0 - x`.** Tags 30–33
   are there to catch exactly that.

## Proposed fix

1. **codegen_x86_linux.sio, core_ir `OpNeg` float arm:** replace the `subsd`
   sequence with a sign-bit flip on the loaded bit pattern, `btc rax, 63`
   (`48 0F BA F8 3F`, checked with GNU as/objdump). It uses no scratch register
   and no xmm, and handles ±0, ±inf and NaN payloads uniformly.
2. **opt_cleanup.sio, `ocp_const_fold_pass_a2` and `ocp_mfi_const_fold`:** do
   not fold `OpNeg` when the source register carries a float marker or float
   bit (`ocp_rebracket_has_float_marker_before`). Codegen then does the flip.
3. **opt_cleanup.sio, Block CD:** do not normalise `0 - x` to `neg(x)` when
   either operand is float.
4. Add the witness above as a run-pass test.

Acceptance: the witness prints `PASS` on a Madaros rebuilt from the patched
source and on lean_single. The unpatched source-built Madaros must fail it
(control).

## Patched build (2026-09-15)

The three changes above were applied to `4fa6999bcc` and Madaros was rebuilt
with `make build-madaros` under `ulimit -s 524288` (exit 0, 16m40s; md5
`882c55a4…`, control md5 `85e199ed…` from the same commit without the fix).

**Witness.** Patched Madaros prints `PASS`: all 33 tags, including 10, 16, 20,
26 and 28. Re-measured the same day, the control still fails exactly those five
and lean_single still prints `PASS`.

**Emitted code.** The binaries have no section headers, so objdump cannot
disassemble them; the counts below come from a raw byte search of the witness
ELFs.

| Encoding | Control ELF | Patched ELF |
|---|---|---|
| `btc rax, 63` (`48 0F BA F8 3F`) | 0 | 11 |
| `subsd xmm0, xmm1` (`F2 0F 5C C1`) | 18 | 7 |

Eleven negation sites moved from subtraction to a sign-bit flip. The seven
remaining `subsd` are real subtractions, which include the `0.0 - x` cases of
tags 30–33. Those tags pass, so Block CD no longer rewrites them into negation.

**Float/negation subset.** 27 run-pass tests were selected by grepping for
`0.0 - `, `-0.0`, `atan2`, `copysign` and signed-zero names. They were run with
`scripts/dev/run_sio_test_suite.sh --test-list … --jobs 2` and
`SOUNIO_TEST_SOUC_BIN` set to each compiler.

| | Pass | Fail | Stale known-failure (XPAS) | Skip |
|---|---|---|---|---|
| Control (`85e199ed`) | 16 | 2 | 1 | 8 |
| Patched (`882c55a4`) | 17 | 1 | 1 | 8 |

The only difference is `f64_unary_neg_signed_zero_bits.sio`: FAIL on the
control, PASS on the patched build. `math_atan_quadrant_reduction.sio` fails
preflight (type check) on **both**, before codegen, so it is independent of
this change. `cpc2026_scientific_float_parser.sio` is XPAS on both.

**Full suite.** `scripts/dev/run_sio_test_suite.sh` over all 3286 tests, with
`SOUNIO_TEST_SOUC_BIN` set to each compiler. Control: 15:45:40–16:22:52Z,
`--jobs 6`, load ≈ 78 on 96 cores at start. Patched: 15:57:56–16:37:19Z,
`--jobs 5`, load ≈ 108 at start. The two runs overlapped on a shared,
heavily loaded host.

| | Pass | Fail | Known failures | XPAS | Vacuous (tolerated) | Skip |
|---|---|---|---|---|---|---|
| Control (`85e199ed`) | 1300 | 525 | 26 | 9 | 20 | 1406 |
| Patched (`882c55a4`) | 1305 | 520 | 26 | 9 | 20 | 1406 |

Seven tests differ between the runs:

| Test | Control | Patched | Reason in harness log | Isolated rerun, control / patched |
|---|---|---|---|---|
| `f64_unary_neg_signed_zero_bits.sio` | FAIL | PASS | missing stdout `PASS` | FAIL / PASS |
| `test_eisa_evm_v2.sio` | PASS | FAIL | run timed out after 30s | PASS / PASS |
| `complex_arithmetic.sio` | FAIL | PASS | run timed out after 30s | PASS / PASS |
| `compress_crc32.sio` | FAIL | PASS | run timed out after 30s | PASS / PASS |
| `compress_deflate_stored.sio` | FAIL | PASS | run timed out after 30s | PASS / PASS |
| `test_eisa_bridge_v1.sio` | FAIL | PASS | run timed out after 30s | PASS / PASS |
| `test_epistemic_curvature_e2e.sio` | FAIL | PASS | run timed out after 30s | PASS / PASS |

The isolated reruns built each test with the raw compiler into
`/workspace/.tmp/fneg-rerun/` and ran it with no harness timeout. Every ELF
finished in ≤ 1 s on both compilers. The six 30 s timeouts are load noise in
one run or the other, not a behaviour difference. **The only test whose outcome
depends on the compiler is the witness.**

The ELFs were kept outside `/tmp` because the workspace watchdog
(`/workspace/.watchdog/souc-watchdog.sh`) SIGKILLs `/tmp/*.elf` processes older
than 600 s. During both runs `watchdog.log` has one KILL line
(`2026-09-15T16:28:17`, `/tmp/wdprobe.uam17s/probe.elf`), which is not a suite
test.

The 520 failures remaining on the patched build are all present in the control
run. By harness reason the control's 525 split into: 249 preflight
(parse/closure/type check), 156 check-step or other harness errors, 42 compile
or compile-fail expectation, 41 30-second run timeouts, 26 nonzero exit after
compile, 8 crashes (134/139), and 3 wrong stdout. This change does not address
them.

## Not claimed / out of scope

- `native_v2_emit_unary_code` (MIR path), `self-hosted/native/lower_ir.sio`
  `lower_unaryop`, and the aarch64 preview `MIR_OP_UNARY` all emit an
  integer `neg` with no float arm. Whether any float reaches them in the
  default x86-64 Madaros path was not measured. Not patched here.
- The other `is_neg` rewrites in `ocp_const_fold_pass_b`
  (e.g. `x + neg(x)`, `neg(a) * neg(b)`) were not audited for IEEE exactness
  on floats.
- `print(f64_to_bits(...))` "unresolved scalar kind" under Madaros.
