<!-- docs:meta
topic_id: repo.docs.audit.madaros-native-v2-codegen-census-2026-06-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-native-v2-codegen-census-2026-06-19
-->

# Madaros native_v2 `build` codegen — capability census (2026-06-19)

Measured against prebuilt `bin/madaros-linux-x86_64` v0.80.0 (main @ `659492156`, md5
`0ce26917…`, unchanged during the run). Each probe was built with
`madaros build <f> -o <e>` and **verified to actually emit** (`native_v2_compile: emitted
path=<e>` marker — the same one `scripts/ci/madaros_full_gate.sh` asserts), then run, with
the result checked via **process exit code** to avoid the broken integer-print path.

> Harness lesson (cost me 3 false readings): `madaros build` can return **rc=0 yet emit no
> ELF**. A naive harness then runs a *stale* ELF from a previous probe and reports its result.
> Always `rm` the output first AND require the `emitted path=` marker before running.

## Scope of the gate vs reality
`madaros_full_gate.sh` is almost entirely `--check` assertions. Its only codegen coverage is
`fn main() -> i64 { 0 }` and the synthetic `--native-v2-emit-scalar 42`. So the holes below
are **uncovered immature paths, not regressions** — the gate is green because it never
exercises them.

## Results
| Probe | native_v2 `build` | Notes |
|---|---|---|
| int arithmetic `5+7` | ✅ OK (rc 12) | deterministic ×3 |
| array index | ✅ OK (rc 30) | |
| struct field | ✅ OK (rc 5) | |
| nested struct | ✅ OK (rc 7) | |
| recursion (factorial) | ✅ OK (rc 120) | |
| tuple `.0/.1` | ✅ OK (rc 3) | |
| `while` loop | ✅ OK (rc 5) | |
| string `println` | ✅ OK | prints, rc 0 with explicit `return 0` |
| **int `println`** | ❌ **SIGSEGV@run** | deterministic ×3; integer→output path. `println("42")` works, `println(42)` crashes |
| **enum `match`** | ❌ **silent no-emit** | `--check` OK; `build` rc=0 but no ELF |
| **`for` range loop** (`for i in 0..10`) | ❌ **silent no-emit** | `--check` OK; `build` rc=0 but no ELF. (`while` works) |
| **method call** (`c.get()`) | ❌ **no-emit** | `build` rc=0, no ELF |
| **`Box::new`** | ❌ **SIGSEGV@compile** | see `MADAROS_BOXNEW_SIGSEGV_2026-06-19.md` (Codex owns) |
| Option `match` | ⚠️ inconclusive | checker error E018/E137 in probe — needs a clean repro |

## Implication for "Madaros 100%"
The `--check` frontend (lex/parse/typecheck, incl. multi-module visibility) is solid. The
**native_v2 codegen→ELF path is early-stage**: it compiles straight-line int/array/struct/
recursion/tuple/while/string code, but cannot yet compile integer printing, method calls,
enum `match`, or `for` range loops — all of which the compiler's own source uses pervasively.
So the fixed point (Madaros self-compiling main.sio → gen2==gen3) is far off; `Box::new` is
just the first crash hit. Closing these codegen holes is the bulk of the remaining work.

### Suggested fix order (independent of the Box::new dispatch Codex is on)
1. integer `println` (SIGSEGV@run) — highest-frequency, blocks observability of everything else
2. `for` range-loop codegen (silent no-emit) — `while` works, so likely a localized lowering gap
3. enum `match` codegen (silent no-emit)
4. method-call codegen (no-emit)
5. `Box::new` (Codex — `MADAROS_BOXNEW_SIGSEGV_2026-06-19.md`)

Also worth fixing: **`build` returning rc=0 while emitting nothing** should be a hard error,
not a silent success — it actively hides codegen gaps (and broke my harness).
