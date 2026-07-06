# WP-B1 witness + root-cause (premise falsified)

**Assigned premise (WP-B1.md):** EISA default-lane failure is a codegen
dependency-closure bug — `eisa::isa` calls `str_to_string` → `str_from_bytes`,
whose body is "absent from the closure" and emitted as a `ud2` stub → SIGILL at
runtime. Fix expected in `module_loader.sio` / `module_frontend.sio`.

**Falsified.** Build-verified 2026-07-06 on two freshly-built Madaros binaries —
`origin/main` @ `a08a0a737` and base `integration/continuity-mtrack` — via the
Slurm build+run driver. Both produce the **identical** failure, and it is **not**
a runtime SIGILL and **not** a codegen/closure problem:

- The module file dep-closure works. `module_frontend_compile_imported_to_file`'s
  BFS (`module_frontend.sio`) transitively loads `str::lib` for both the minimal
  repro here (`loaded 4 modules`) and `test_eisa_isa` — `str::lib` bodies ARE
  merged. No `ud2` is emitted; no ELF is ever produced.
- The failure is **pre-codegen**, in the multi-module MERGED TYPE-CHECKER
  `check::mod::check_modules_verdict_boot4` (collect-all-modules → check bodies).

## Error inventory (fresh main a08a0a737, `<src> -o out`)

Minimal repro (`depclosure_str_main.sio`, deps = str::lib only):

| code | count | meaning | source |
|---|---|---|---|
| E004 | 48 | "these types cannot be combined" — `i64` vs `i32` | `stdlib/str/lib.sio` (`Str.len: i32` mixed with `i64` loop vars, e.g. `i < s.len`) |
| E007 | 3 | branches have incompatible types (i32/i64 if-arms) | str::lib |
| E008 | 3 | return value does not match declared return type | str::lib |
| E009 | 3 | argument type does not match parameter | str::lib |
| E012 | 10 | this type has no field named | str::lib (merged-checker field resolution) |

`test_eisa_isa` adds, on top of the same str::lib E004 family:

| code | count | meaning |
|---|---|---|
| E137 | 37 | use of undeclared variable — merged-checker does NOT resolve transitively-imported `math::dd64` / `eisa::core` symbols (e.g. `dd_to_f64`, `dd_abs`, `f64_decompose` at isa.sio byte ~6230) |
| E015 | 2 | unknown struct type |

The **shipped prebuilt** `bin/madaros-linux-x86_64` (older source) fails the same
program with **E175** "function is private in its defining module" instead —
because a recent commit set `enforce_visibility=false` in the merged check,
which SUPPRESSED E175 and EXPOSED the underlying E004 int-width errors. Both are
the same fundamental fact: **the default Madaros lane cannot type-check any
multi-module program that transitively imports `str::lib`.** EISA has therefore
always been pinned to `lean_single` (see the `validated_lane: lean_single`
header on `test_eisa_isa.sio`), whose token-pattern checker is lenient about
integer width and does not run this merged pass.

## Why no in-scope fix reaches the assigned witnesses

- Madaros's native lane is **multi-module only** (a `use`-free source returns
  "native compile disabled for single-module streaming lane"), so `str::lib` is
  *only ever* checked via `check_modules_verdict_boot4`. There is no
  single-module Madaros path that already accepts it — so there is no
  "make-multi-match-single" narrow fix.
- The true fix is in `self-hosted/check/` (`compat.sio` binary-op integer-width
  widening for `is_integer_type` operands under arith/cmp; plus merged-checker
  transitive symbol/field/visibility resolution for E137/E012). That is **out of
  the `module_loader.sio` / `module_frontend.sio` scope** the WP named, is a
  broad checker change with cross-suite regression risk, and shares the merged
  `check_modules_verdict_boot4` path with WP-A6. No scoped change in the WP's
  named files makes `test_eisa_isa`/`test_eisa_evm` type-check.

## Witness status

| # | what | lane | expected | actual |
|---|---|---|---|---|
| W1 | this 3-module repro | lean_single | rc 0 | **rc 0 PASS** |
| W1 | this 3-module repro | default (Madaros) | rc 0 | **BLOCKED — E004×48 at typecheck** |
| W2 | test_eisa_isa | default (Madaros) | PASS | **BLOCKED — E004 family + E137×37 + E015×2** |
| W3 | test_eisa_evm | default (Madaros) | PASS | **BLOCKED — same merged-checker family** |

Run the lean-lane witness:

```
SOUNIO_STDLIB_PATH=stdlib ./bin/souc-lean-single-x86_64 \
  docs/handoff/continuity/wp-b1-witness/depclosure_str_main.sio /tmp/w1.elf && /tmp/w1.elf; echo rc=$?
# => rc=0
```
