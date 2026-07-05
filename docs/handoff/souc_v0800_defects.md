# souc v0.80.0 compiler defects — minimal repros + exact stdout evidence

Compiler: `Madares v0.80.0 -- the Sounio self-hosted compiler` (self-hosted Madaros
engine, invoked as `souc`), worktree `/workspace/sounio-exact-algebra`
(branch `coord/exact-algebra-core`). All commands below were run as:

```
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc check <file>
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc build <file> -o <out>   # or `run`
```

**Reminder honored throughout**: `souc` exits `rc=0` even on failure. Every
verdict below is read from literal stdout content (`check: OK`, program
output, `error[Exxx]`, `Segmentation fault`), never from the shell exit code
alone — except where the exit code itself is cited as corroborating evidence
(e.g. `Segmentation fault` messages, or SIGSEGV rc=139).

All repro files live under `docs/handoff/repros/`.

---

## D1 — multi-module compile false-green to a silent stub

### Symptom
Reported as: a program importing 2+ modules compiles to a ~140-byte
do-nothing ELF that prints nothing at runtime, with `rc=0` looking
successful.

### Minimal repro
- `docs/handoff/repros/d1_modA.sio`, `docs/handoff/repros/d1_modB.sio` — two
  trivial two-param exported functions.
- `docs/handoff/repros/d1_main_one_import.sio` — control, imports ONE module.
- `docs/handoff/repros/d1_main_two_imports.sio` — imports BOTH modules.

```sio
// d1_modA.sio
pub fn a_value(a: i64, b: i64) -> i64 { a + b }

// d1_modB.sio
pub fn b_value(a: i64, b: i64) -> i64 { a * b }

// d1_main_one_import.sio
use d1_modA::{a_value}
fn main() -> i32 with IO {
    let r = a_value(3, 4)
    println("one-import result:")
    print_int(r)
    0
}

// d1_main_two_imports.sio
use d1_modA::{a_value}
use d1_modB::{b_value}
fn main() -> i32 with IO {
    let ra = a_value(3, 4)
    let rb = b_value(3, 4)
    println("two-import result A:")
    print_int(ra)
    println("two-import result B:")
    print_int(rb)
    0
}
```

### Exact stdout

**Two-import case (`check` then `build` then execute the produced ELF):**

```
$ souc check d1_main_two_imports.sio
...
run_check_mode: about to check 2
 modules
run_check_mode: verdict=0

check: OK

$ souc build d1_main_two_imports.sio -o /tmp/d1_two_final.elf
...
Native compilation failed: imported_simple_ir_emit_failed
module_native_driver: compact IR ELF write failed; rc=1
; falling back to full IR path
...
Merged IR: 8
 functions
Written to /tmp/d1_two_final.elf
Compilation successful!
   Output: /tmp/d1_two_final.elf

$ ls -la /tmp/d1_two_final.elf
-rwxr-xr-x 1 ... 12296 ... /tmp/d1_two_final.elf

$ /tmp/d1_two_final.elf; echo RC=$?
two-import result A:
7two-import result B:
12RC=0
```

**One-import control:**

```
$ souc check d1_main_one_import.sio  → check: OK
$ souc build d1_main_one_import.sio -o /tmp/d1_one_final.elf
... same "compact IR ELF write failed ... falling back to full IR path" ...
Written to /tmp/d1_one_final.elf
$ ls -la /tmp/d1_one_final.elf
-rwxr-xr-x 1 ... 12296 ... /tmp/d1_one_final.elf
$ /tmp/d1_one_final.elf; echo RC=$?
one-import result:
7RC=0
```

### Could NOT reproduce the literal "140-byte silent stub" in this build

This is an honest negative result, not a hedge. Both the one-import control
and the two-import case:
- produce identically-sized (12296-byte) ELF binaries,
- print real, correct computed output,
- go through the exact same code path in the compiler.

Root cause of the discrepancy with the reported symptom, traced in
`self-hosted/compiler/module_native_driver.sio`
(`compile_multimodule_native_maybe_streaming`, lines ~1160-1190): the modular
build always attempts a **"compact modular IR table" fast path** first
(`load_multimodule_imported_simple_ir_global` +
`native_driver_write_imported_simple_ir_elf`). That fast path is a hand-rolled
x86 emitter (`self-hosted/compiler/module_native_driver.sio:480-928`) that
only recognizes a small, closed set of hard-coded function *shapes*
(`fn_kind` 1..24) — several of which (kinds 7, 10, 11, 16, 17, 18, 23) require
a sibling function with one of a handful of **literal names** baked into the
compiler itself (`"double"`, `"select_unary"`, `"pass_through"`, `"entry"`,
`"closure_inc"`, `"closure_triple"`, `"closure_add"`, `"closure_sub"`,
`"make_adder"`, `"apply_captured"` — clearly compiler-bring-up test fixture
names, not a general mechanism). For any code that does not match one of
these templates (including both repros above), `native_driver_emit_imported_simple_fn`
returns `false`, the driver prints
`Native compilation failed: imported_simple_ir_emit_failed`, and — critically
— **correctly falls back** to `module_frontend_compile_imported_to_file`
(the full IR path), which produces a correct, non-stub binary. This fallback
is what both repros exercise.

The originally-reported 140-byte stub would require the compact path's
`.ok`/`simple_rc == 0` branch (line 1173-1177) to *succeed* rather than fail —
i.e. real source code coincidentally matching one of the fixture-name
templates above, or a shape whose classification is wrong in a way that
still passes `native_driver_emit_imported_simple_fn`. That could not be
triggered from generic 2-module code in this build. In this build, the
compact path acts as a fail-fast guard rather than a silent-wrong-success
path for ordinary multi-module programs. If the 140-byte stub is still live
elsewhere, it likely requires either (a) a `lean_single`-engine build (a
different engine — see below, but that failed with a different, earlier
symptom: `error: no main`, for both one- and two-import cases, so it is not
directly comparable either), or (b) a real name collision with one of the
hard-coded fixture identifiers.

```
$ SOUNIO_SOUC_ENGINE=lean_single souc build d1_main_two_imports.sio -o /tmp/x.elf
...
error: no main
$ SOUNIO_SOUC_ENGINE=lean_single souc build d1_main_one_import.sio -o /tmp/y.elf
...
error: no main
```
(lean_single fails identically regardless of import count in this build —
not the reported stub symptom either, and not investigated further; flagged
only as a data point for whoever revisits this.)

### Impact
D1 (as originally diagnosed, whether or not it reproduces in this exact
build) is why the exact-algebra work
(`docs/EXACT_CORE.md`) kept every executing test to a **single** import
(inlining `cd_sigma` rather than importing `cayley_dickson_exact_i64` +
`sedenion_verdict` together) and left `sedenion_verdict.sio` "import-ready
once multi-module is fixed" rather than actually imported anywhere. That
constraint should be re-tested against this build: the two repros above
*did* successfully exercise a genuine 2-module import+call+print with
correct output, so the single-import self-containment constraint may be
over-cautious for this specific build/shape — but D2's cross-module arity
bug (below) is a separate, confirmed, still-live reason to keep cross-module
aggregate-passing calls out of the shipped tests.

### Suspected area
`self-hosted/compiler/module_native_driver.sio` — specifically the "compact
modular IR table" fast path (`compile_multimodule_native_maybe_streaming`,
`native_driver_write_imported_simple_ir_elf`,
`native_driver_emit_imported_simple_fn`) vs. the full-IR fallback
(`module_frontend_compile_imported_to_file`); function-shape classification
lives in `self-hosted/compiler/module_frontend.sio`
(`imported_simple_ir_global_fn_kind`, `MODULE_FRONTEND_IMPORTED_SIMPLE_FN_KINDS`).

---

## D2 — large-aggregate struct cross-module arity-mismatch segfault

### Symptom
A struct with a fixed-size array field (e.g. `[i64; 2048]`), when passed
by value across module boundaries through functions of **mismatched
aggregate-parameter arity**, segfaults the compiled binary at runtime.

### Minimal repro (isolated)
- `docs/handoff/repros/d2_lib_arity_mismatch.sio` (library) +
  `docs/handoff/repros/d2_main_crossmod_arity_mismatch_segv.sio` (main) —
  **CRASHES**.
- `docs/handoff/repros/d2_boundary_singlemod_arity_mismatch_ok.sio` — same
  arity-mismatch shape, but single-module — **does NOT crash**.
- `docs/handoff/repros/d2_lib_arity_matched.sio` +
  `docs/handoff/repros/d2_boundary_crossmod_arity_matched_ok.sio` —
  cross-module, but matching arity — **does NOT crash**.

```sio
// d2_lib_arity_mismatch.sio
pub struct S { c: [i64; 16], bits: i32 }
pub fn make_zero(k: i32) -> S with Mut, Panic { S { c: [0; 16], bits: k } }
pub fn make_one(k: i32) -> S with Mut, Panic {
    var r = S { c: [0; 16], bits: k }
    r.c[0] = 1
    r
}
pub fn add2(a: S, b: S) -> S with Mut, Panic, Div {     // 2 aggregate params
    var r = make_zero(a.bits)                            // calls a 1-arg aggregate-return fn
    var i: i32 = 0
    while i < 16 { r.c[i as usize] = a.c[i as usize] + b.c[i as usize]; i = i + 1 }
    r
}
pub fn add3(a: S, b: S, c: S) -> S with Mut, Panic, Div { // 3 aggregate params
    let ab = add2(a, b)                                   // calls a 2-arg aggregate-return fn (mismatch!)
    add2(ab, c)
}

// d2_main_crossmod_arity_mismatch_segv.sio
use d2_lib_arity_mismatch::{make_one, add3}
fn main() -> i32 with IO, Mut, Panic, Div {
    let a = make_one(4); let b = make_one(4); let c = make_one(4)
    let r = add3(a, b, c)
    println("done")
    print_int(r.c[0])
    0
}
```

### Exact stdout

**Crashing case:**
```
$ souc check d2_main_crossmod_arity_mismatch_segv.sio
...
run_check_mode: about to check 2
 modules
run_check_mode: verdict=0

check: OK

$ souc run d2_main_crossmod_arity_mismatch_segv.sio
...
Merged IR: 10
 functions
Written to /workspace/.tmp/madaros-run.53wLWQ/main.elf
Compilation successful!
   Output: /workspace/.tmp/madaros-run.53wLWQ/main.elf
/workspace/sounio-exact-algebra/bin/madaros: line 342: 228863 Segmentation fault      "$out" "$@"
RC=139
```
Note: `check` reports `check: OK` (rc=0-looking) — the type checker sees
nothing wrong; the segfault only manifests when the produced native binary
actually **runs**, i.e. `check` alone is not sufficient evidence of
correctness for this shape, confirming the "judge by stdout, not rc" rule
matters here in a second way (rc=139 is a real crash signal, but a plain
`check` pass would false-green it).

**Boundary control A — single-module, identical arity mismatch — OK:**
```
$ souc run d2_boundary_singlemod_arity_mismatch_ok.sio
...
Merged IR: 8
 functions
Written to /workspace/.tmp/madaros-run.GqlsPI/main.elf
Compilation successful!
   Output: /workspace/.tmp/madaros-run.GqlsPI/main.elf
done
3RC=0
```

**Boundary control B — cross-module, matching arity (3-param calling 3-param) — OK:**
```
$ souc run d2_boundary_crossmod_arity_matched_ok.sio
...
Merged IR: 9
 functions
Written to /workspace/.tmp/madaros-run.SwSFfY/main.elf
Compilation successful!
   Output: /workspace/.tmp/madaros-run.SwSFfY/main.elf
done
3RC=0
```

### Boundary
Array size is **not** the discriminant: the crashing shape (`add3`/`add2`
arity mismatch, cross-module) was confirmed to segfault identically at
array sizes 16, 64, 256, 1024, **and** 2048 (all five sizes tested, same
crash). The actual discriminant is the **combination** of:
1. crossing a module boundary (`use`), **and**
2. an aggregate-by-value function of arity N calling another
   aggregate-by-value/return function of a *different* arity M ≠ N.

Either condition alone is safe: same arity-mismatch shape single-module (no
`use`) runs correctly; same cross-module shape with matching arity (3→3)
runs correctly. Only cross-module + arity-mismatch crashes, and it crashes
at every array size tested. This matches (and sharpens) the pre-existing
in-repo note in `stdlib/algebra/cayley_dickson_exact_i64.sio` (lines ~85-97)
documenting the same bug and its workaround (constructing the zero
accumulator via an inline struct literal instead of delegating to
`cd_zero_exact_i64`, to avoid the mismatched-arity nested call).

### Impact
This is the reason `stdlib/algebra/cayley_dickson_exact_i64.sio`'s
`cd_add_exact_i64`/`cd_sub_exact_i64`/`cd_mul_exact_i64` construct their
zero accumulator inline (`CDElementExactI64 { c: [0; 2048], bits: ... }`)
rather than calling `cd_zero_exact_i64` — sidestepping the segfault at the
cost of code duplication. It also motivated keeping the exact-algebra test
suite single-module/self-contained rather than composing helper functions
of varying arity across module boundaries.

### Suspected area
Cross-module aggregate-by-value ABI lowering — most likely
`self-hosted/ir/lower.sio` (aggregate/struct-by-value lowering, SRET/return
forwarding) and/or `self-hosted/ir/inline.sio` (cross-module inlining of
calls with differing aggregate-parameter counts), given the historical audit
trail (`docs/audit/MADAROS_MULTIMODULE_NATIVE_SEED_SEGFAULT_2026-06-22.md`,
`docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md`) pointing at
the same modular-native/multimodule-import lowering machinery as D1.

---

## D3 — `<<` yields i64, breaking i32 contexts (E004)

### Symptom
The `<<` (left-shift) operator's result type is always inferred as `i64`
regardless of operand types, so `let half = 1 << (bits - 1)` in an `i32`
function breaks any subsequent i32-typed use of `half`.

### Live evidence in the real file
```
$ SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc check stdlib/algebra/cayley_dickson.sio
Madares v0.80.0 -- the Sounio self-hosted compiler
the bare highland that does not negotiate with ill-formed code -- Sfakia, Crete
Horizon 3: self-hosted primary compiler.

error[E008
] at 8608
..8679
: return value does not match function's declared return type
   |
   = expected i32
   = found i64
   |
   = help: change the return value type to match the function's declared return type
   = note: the return type is declared in the function signature
error[E007
] at 8782
..8816
: branches have incompatible types
   |
   = expected i32
   = found i64
   |
   = help: ensure both branches return the same type, or use explicit conversion
   = note: the type of an if-expression is determined by its branches
```
The crash site is `cd_sigma` (line 31: `let half = 1 << (bits - 1)`, line 35:
`a & (half - 1)`, etc., in `stdlib/algebra/cayley_dickson.sio`). Note: the
**current** diagnostic codes surfaced by this build are `E008`/`E007`
(return-type mismatch / if-branch type mismatch), not the `E004` cited in
prior notes for this exact file — the codes appear to have been renumbered
or the cascade path changed since that note was written, but the root cause
(the `<<` operator's i64 inference) is unchanged and confirmed below.

### Minimal repro
`docs/handoff/repros/d3_shift_i32.sio`:

```sio
pub fn shift_half(bits: i32, a: i32) -> bool {
    let half = 1 << (bits - 1)
    a >= half
}
```

### Exact stdout
```
$ SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc check docs/handoff/repros/d3_shift_i32.sio
Madares v0.80.0 -- the Sounio self-hosted compiler
the bare highland that does not negotiate with ill-formed code -- Sfakia, Crete
Horizon 3: self-hosted primary compiler.

error[E004
] at 0
..192
: these types cannot be combined with this operator
   |
   = expected i32
   = found i64
   |
   = help: use explicit type conversion with `as` if the types are compatible
   = note: binary operators require operands of compatible types
error[E004
] at 0
..192
: these types cannot be combined with this operator
   |
   = expected i32
   = found i64
   |
   = help: use explicit type conversion with `as` if the types are compatible
   = note: binary operators require operands of compatible types
```
This minimal repro reproduces `error[E004]` verbatim, matching the originally
reported code exactly (the difference in error code on the full
`cayley_dickson.sio` file vs. this minimal repro is because the full file's
`cd_sigma` has an explicit `-> i32` return type and `if`/`return` expressions
that surface as E008/E007 further down the same root cause — both stem from
the same `<<` i64-inference defect).

### Impact
Blocks importing the f64 analytic/flow Cayley-Dickson layer
(`stdlib/algebra/cayley_dickson.sio`) anywhere — `cd_sigma`,
`cd_count_nonassociative`, and everything downstream fail to type-check.
This is why `stdlib/algebra/cayley_dickson_exact_i64.sio` **inlines** its own
copy of `cd_sigma` (with explicit `as i32` casts added around every shift)
rather than importing the shared one, per that file's own header comment
(lines 17-28).

### Suspected area
Binary-operator type inference/checking for shift operators — likely in the
typechecker (`self-hosted/check/check.sio`, given that's where other E00x
diagnostics for binary operators and shift/return-type mismatches are
raised in this compiler, per the E004 grep hits in that file).

---

## D4 — data-carrying enum `match` codegen flake

### Symptom
Two structurally-mirrored functions matching over the same 3-variant
data-carrying enum can disagree: one gives the fully correct truth table,
while its "mirror image" sibling returns the wrong boolean for the
`MeasuredF64 { eps }` variant.

### Minimal repro
`docs/handoff/repros/d4_enum_match.sio` (both functions in one module):

```sio
enum Verdict { Proved, MeasuredF64 { eps: f64 }, MeasuredF256 { eps256: f64 } }

fn make_measured_f64(e: f64) -> Verdict { Verdict::MeasuredF64 { eps: e } }

fn requires_proof(v: Verdict) -> bool {
    match v {
        Verdict::Proved => true,
        Verdict::MeasuredF64 { eps } => false,
        Verdict::MeasuredF256 { eps256 } => false,
    }
}

fn is_measurement(v: Verdict) -> bool {
    match v {
        Verdict::Proved => false,
        Verdict::MeasuredF64 { eps } => true,
        Verdict::MeasuredF256 { eps256 } => false,
    }
}

fn main() with IO {
    let v = make_measured_f64(1.0)
    let rp = requires_proof(v)
    let v2 = make_measured_f64(1.0)
    let im = is_measurement(v2)
    println("requires_proof(MeasuredF64) [expect false]:")
    if rp { println("true") } else { println("false") }
    println("is_measurement(MeasuredF64) [expect true]:")
    if im { println("true") } else { println("false") }
}
```

(Direct enum-struct literals inside an `if` condition, e.g.
`if f(Verdict::MeasuredF64 { eps: 1.0 })`, PARSE-FAIL on brace ambiguity —
confirmed while building this repro — so values are always built via a
constructor fn or a `let` binding first, never inlined directly into a
condition.)

### Exact stdout
```
$ souc check docs/handoff/repros/d4_enum_match.sio → check: OK
$ souc run docs/handoff/repros/d4_enum_match.sio
...
Merged IR: 6
 functions
Written to /workspace/.tmp/madaros-run.oOSvtR/main.elf
Compilation successful!
   Output: /workspace/.tmp/madaros-run.oOSvtR/main.elf
requires_proof(MeasuredF64) [expect false]:
false
is_measurement(MeasuredF64) [expect true]:
false
```
`requires_proof` is fully correct (`false` — a measurement is not a proof).
`is_measurement` is WRONG: it returns `false` for a `MeasuredF64` value,
when it should return `true`.

### Boundary / isolation
Tested whether this requires BOTH functions in the module (an interaction
bug) — it does **not**:

`docs/handoff/repros/d4_enum_match_isolated.sio` has **only**
`is_measurement` (no `requires_proof` anywhere in the module):
```
$ souc run docs/handoff/repros/d4_enum_match_isolated.sio
...
Merged IR: 5
 functions
Written to /workspace/.tmp/madaros-run.bXlFHS/main.elf
Compilation successful!
   Output: /workspace/.tmp/madaros-run.bXlFHS/main.elf
is_measurement(MeasuredF64) alone in module [expect true]:
false
```
Still wrong, in isolation. Also tested whether the construction path matters
— `docs/handoff/repros/d4_enum_match_inline_literal.sio` builds the value via
a direct `let v = Verdict::MeasuredF64 { eps: 1.0 }` literal (no constructor
fn, no function-return indirection):
```
$ souc run docs/handoff/repros/d4_enum_match_inline_literal.sio
...
is_measurement(MeasuredF64) via direct literal let-binding [expect true]:
false
```
Still wrong. So the flake is **not** an interaction between two match
functions and **not** dependent on how the enum value is constructed
(function-return vs. direct `let`-literal) — it appears to be a codegen bug
tied purely to `is_measurement`'s specific arm-body sequence
(`false, true, false` in Proved/MeasuredF64/MeasuredF256 order), independent
of everything else in the module.

### Impact
This is why `stdlib/algebra/sedenion_verdict.sio` (lines 51-68) ships only
`requires_proof` and derives "is a measurement" by negation
(`!requires_proof(v)`) instead of exposing a redundant `is_measurement`
helper — the helper was dropped because it was unreliable, per that file's
own in-code comment (lines 58-61).

### Suspected area
Data-carrying-enum `match` codegen — likely in
`self-hosted/ir/lower.sio` (match-arm lowering to branch code) given the
symptom depends on the specific true/false sequence of arm bodies rather
than on structural properties like arm count, variant shape, or module
composition.

---

## Summary table

| Defect | Repro confirmed | Signature |
|---|---|---|
| D1 (2+ module compact-stub false-green) | **NOT reproduced** (honest negative — both 1-import and 2-import cases produced correct, equally-sized, non-stub binaries in this build; fallback path in `module_native_driver.sio` engages correctly) | `Native compilation failed: imported_simple_ir_emit_failed` → `falling back to full IR path` → correct output |
| D2 (cross-module aggregate arity-mismatch segfault) | **Yes**, isolated to array-size-independent cross-module + arity-mismatch combination | `Segmentation fault` (rc=139) after `Compilation successful!`; `check` reports `check: OK` (false-green at the check level) |
| D3 (`<<` infers i64) | **Yes**, exact `error[E004]` match on minimal repro; real file shows same root cause surfacing as E008/E007 | `error[E004]: these types cannot be combined with this operator … expected i32 / found i64` |
| D4 (enum match codegen flake) | **Yes**, confirmed in combined module, in isolation (no sibling fn), and independent of construction method | `is_measurement(MeasuredF64)` prints `false`, should print `true`; `requires_proof` on the same enum is fully correct |

---

## Filed as GitHub issues (2026-07-05)

| Defect | Issue | Status |
|---|---|---|
| D2 — cross-module aggregate arity-mismatch SIGSEGV | Sounio-lang/sounio#637 | filed, repro verified |
| D3 — `<<` infers i64 in i32 context (E004) | Sounio-lang/sounio#638 | filed, repro verified |
| D4 — data-enum `match` returns wrong arm value | Sounio-lang/sounio#639 | filed, repro verified |
| D1 — multi-module compact-stub false-green | NOT filed | **negative result** — not reproducible in this build; earlier reports of a 140-byte stub on engine-importing tests were a manifestation of **D2** (aggregate lowering), not an independent multi-module defect |

---

## D5 — while-loop reassigning a loop-carried var via a global-reading/dividing guard fn clobbers it

### Symptom
A `while` loop that reassigns a loop-carried mutable var based on a guard **function that reads a
global and does integer division** mis-compiles — the call clobbers the carried var (collapses to 0).

### Minimal repro (CONFIRMED)
`docs/handoff/repros/d5_loop_clobber_reproduces.sio` — `mul_ovf(p,10)`-guarded `p = p*10`/`p = 0` in a
`while`. Output: `k=1 p=10  k=2 p=100  k=3 p=0  k=4 p=0` (expected `10,100,1000,10000`).

### Isolation
`docs/handoff/repros/d5_loop_trivial_guard_ok.sio` — same loop, trivial literal guard (no global, no
division) → correct `10,100,1000,10000`. Direct (non-loop) reassignment → correct. So the trigger is
the guard fn's global-read + division interacting with the loop-carried var, NOT "call in loop" alone.

### Impact
Forced the exact-measure sweep (`sedenion_measure_annihilation_general.sio`) to be **unrolled**;
`var_at_scale` is correct with literal args but wrong in a loop. Any overflow-checked numeric loop
(guard fn + accumulator) is at risk.

### Suspected area
Register allocation / caller-saved handling around a call whose callee issues `div` (clobbers
rdx:rax on x86) inside a loop with a live carried var — likely `self-hosted/native/codegen_x86_linux.sio`.

### Filed
Sounio-lang/sounio#641.

---

## D6 — copy-then-mutate a struct aliases the caller's value (broken value semantics) [MOST SEVERE]

### Symptom
`var r = a; r.field = x` where `a` is a struct **parameter** mutates the caller's ORIGINAL in place —
struct value/copy semantics are broken; `var r = a` aliases. **Silent data corruption, clean compile.**

### Minimal repro (CONFIRMED)
`docs/handoff/repros/d6_struct_copy_aliases.sio`: `flip(a:P)->P { var r=a; r.flag=!a.flag; r }` →
`p.flag=true q.flag=true` (expected `false/true` — the caller's `p` was mutated).
Control `d6_struct_fresh_literal_ok.sio`: fresh literal `P { x:a.x, flag:!a.flag }` → correct.

### Impact
Silently corrupted an exact bigint: `big_neg` used `var r = a; r.neg = !a.neg`, so `delta = -s`
flipped `s` itself — mathematically wrong yet cleanly compiling. Caught ONLY by the oracle diff.
Fixed in `stdlib/math/bignat.sio` (fresh literal). Any copy-then-mutate on a struct param is at risk.

### Suspected area
Local init from an aggregate value emits a pointer/alias instead of a value copy (missing memcpy) —
`self-hosted/ir/lower.sio` (aggregate local init) or codegen for `var x = <aggregate>`.

### Filed
Sounio-lang/sounio#643.
