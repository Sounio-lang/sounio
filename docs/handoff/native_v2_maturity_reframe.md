<!-- docs:meta
topic_id: repo.docs.handoff.native-v2-maturity-reframe
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.native-v2-maturity-reframe
-->

# Reframe — native-v2 "backend maturity" track (empirical baseline, 2026-07-11)

**Authored by:** Claude (EISA lane, `gpu/epistemic-tensor-core-next` @ `b638354c0`), 2026-07-11
**Type:** compiler-internals characterization (modular `main.sio` / native-v2). No code changes — measurement + scoping only.
**Priority:** informational. Corrects the working premise before anyone invests in the track.

## TL;DR

The track was framed as *"the native-v2 backend can't compile non-trivial programs (e.g. cross-module struct-return)."* **Measured against a freshly-built modular `madaros` (from the just-merged tree), that premise does not hold for the `madaros run` path** — struct, struct-return, recursion, and struct+recursion all compile and run correctly. The real open work is three *different*, largely-independent items, none of which is "structs don't codegen":

1. **Stale prebuilt seed** — `bin/souc` / `bin/souc-linux-x86_64` lag the source, so all 41 `native_v2_*` gates are red at the *build* step. **The track is currently unmeasurable via its own harness.**
2. **Driver by-value-param blocker** — the fresh `madaros` cannot compile `self-hosted/compiler/native_compile_driver.sio` (the entry the gates actually use): `by-value parameter of fn parser_set_tokens — consider passing by &!/reference`. So `madaros run` and the driver path are genuinely different backends, and the driver path has its own blocker.
3. **Generic `<F>` monomorphizer residual** — turbofish and simple generic-struct-return now PASS, but a generic struct with a `[F; N]` array field (the exact-algebra spike) still fails E009 (`found CDExact__T` vs `expected CDExact__i64`).

**Honest boundary:** this shows *6 common shapes* work on *one* path. It does **not** show the backend is broadly mature — the 41-gate suite is unmeasured (item 1).

## Evidence (fresh madaros, built via `build_modular_madaros.sh` on `b638354c0`, 96.8 MB)

CLI: `madaros run <prog.sio>` compiles + executes; process exit code = program's return value.

| Program shape | `madaros run` exit | Expected | Result |
|---|---:|---:|---|
| `fn main()->i64 {42}` | 42 | 42 | ✓ |
| function call + arith | 42 | 42 | ✓ |
| recursion (`fib(10)`) | 55 | 55 | ✓ |
| struct construct + field access | 42 | 42 | ✓ |
| **struct return** (`fn mk()->P`) | 42 | 42 | ✓ |
| **struct + recursion** (`mk(fib(8),2)`) | 23 | 23 | ✓ |
| `tests/run-pass/turbofish.sio` | — | — | ✓ PASS (all 3 asserts) |
| `tests/run-pass/generic_struct_return.sio` | — | — | ✓ prints `6` / `spike PASS` |
| `docs/handoff/spike_generic_struct_return.sio` (`[F;4]` field) | — | — | ✗ E009 `found CDExact__T` |

The earlier-in-session "struct+fib fails `native_v2_compile` rc=1" was the **stale W3 binary**, not the current tree.

## Root causes & candidate lanes

### Lane 1 — refresh the stale prebuilt seed (harness unblock)
- **Symptom:** every `native_v2_*_gate.sh` fails at build/check — e.g. `native_v2_e2e_exit_code`: `could not build modular compiler`; `native_v2_struct_return`: `bin/souc check native_compile_driver.sio` fails under `set -euo pipefail`. `bin/souc` routes to a prebuilt Madaros (`bin/souc-linux-x86_64`, 2.3 MB, lags source).
- **Leverage:** highest — without a current seed the whole track is unmeasured.
- **Risk/uncertainty:** memory + the struct-return handoffs say seed refresh goes via the **Sounio Compiler Foundry, NOT workspace-local**. Feasibility of a workspace-local refresh is unconfirmed — probe first.

### Lane 2 — driver large-by-value-param blocker
- **Symptom:** `madaros run self-hosted/compiler/native_compile_driver.sio -- <prog> -o <out>` → `parse error: expected token at line 545` + `bytes) — by-value parameter of fn parser_set_tokens — consider passing by &!/reference or using heap/global storage`. The fresh madaros cannot compile the 8986-line driver itself.
- **Fix direction:** convert the offending large by-value struct/array parameter(s) to `&!`/reference or module-scope BSS (same pattern as `NC_BIG_ELF` / the W5 `KAXI_WRITE_STAGING`). Local rebuild + re-run the driver path to verify.
- **Risk:** low-moderate, local, verifiable; does **not** touch `lean_single.sio`. Most tractable of the three. Likely unblocks the driver path and several gates once Lane 1 is also addressed.

### Lane 3 — generic `<F>` with `[F; N]` field monomorphizer residual
- **Symptom:** `spike_generic_struct_return.sio` → `error[E009] expected CDExact__i64 / found CDExact__T`. The declared/return side substitutes `F→i64`; the body's *constructed-value* type keeps `F` symbolic and mangles `__T`. NB `substitute_type_param` (`check/check.sio:646`) already recurses into `ty.inner`/`tuple_elems` — the 2026-07-06 diagnosis's point #4 is done; this residual is deeper (which checker the modular compiler actually uses is itself unconfirmed — `main.sio` has no `use check::check`).
- **Consumer:** exact-algebra `<F>` Cayley-Dickson engine (`stdlib/algebra/cayley_dickson_exact.sio`). See [[compiler_generic_struct_return_diagnosis]] / [[compiler_generic_struct_return_fix_prompt]].
- **Risk:** high — the diagnosis mandates serialized surfaces, BOTH engines, fixed-point + output-verified gates, and a `lean_single.sio` Foundry rebuild (touches the proven gen2==gen3 compiler). Rabbit hole, not a quick win.

## Reproduction

```
# fresh compiler
bash scripts/ci/build_modular_madaros.sh /tmp/madaros   # typecheck OK, ~96.8 MB

# backend works on common shapes (madaros run path)
printf 'struct P{x:i64,y:i64}\nfn mk(a:i64,b:i64)->P{P{x:a,y:b}}\nfn main()->i64{let p=mk(40,2);p.x+p.y}\n' > /tmp/sr.sio
/tmp/madaros run /tmp/sr.sio; echo $?     # 42

# driver path blocked (Lane 2)
/tmp/madaros run self-hosted/compiler/native_compile_driver.sio -- /tmp/sr.sio -o /tmp/o.elf   # parser_set_tokens by-value

# generic-<F> residual (Lane 3)
/tmp/madaros run docs/handoff/spike_generic_struct_return.sio   # E009 found CDExact__T

# harness stale (Lane 1)
bash scripts/ci/native_v2_e2e_exit_code_gate.sh   # FAIL: could not build modular compiler
```

## Recommendation

If a lane is picked: **Lane 2** (driver by-value-param) is the most tractable and lowest-risk — local, verifiable, no proven-compiler exposure — and is a prerequisite for exercising the driver-path gates anyway. **Lane 1** is highest-leverage but gated on a Foundry-vs-local feasibility probe. **Lane 3** is real consumer value but a high-risk rabbit hole touching `lean_single`.
