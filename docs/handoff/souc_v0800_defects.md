<!-- docs:meta
topic_id: repo.docs.handoff.souc-v0800-defects
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.souc-v0800-defects
-->

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

---

## D7 — data-carrying enum variants rejected (E200 "undefined variable" on the field) [REGRESSION]

### Symptom
`enum V { M { eps: f64 } }` with construction `V::M { eps: e }` fails on the current self-hosted
souc (the stage2 binary the CI Full-Test-Suite builds): **`E200 \`eps\`` — "undefined variable"** at
the construction site → typecheck fails → no ELF → the runner reports `run exited 1`.

### Regression
The **committed prebuilt `bin/souc` (older) compiles and runs it fine**; the current source's stage2
does not. So a change between the prebuilt and current `self-hosted/compiler/lean_single.sio` dropped
data-carrying-enum-variant support. `stdlib/genomics/io/fasta.sio` (`FastaError { message: str }`)
and any data-carrying enum are affected.

### Minimal repro
`docs/handoff/repros/d7_data_carrying_enum.sio` — verified with the CI `souc-stage2` artifact:
`E200 \`eps\` at line 4`, no ELF. The payload-free form (`enum V { P, M }`) compiles + runs.

### Impact
`sedenion_verdict.sio` / `sedenion_verdict_boundary.sio` used `Verdict::MeasuredF64 { eps: f64 }`;
rewritten **payload-free** so they compile under the CI compiler. The boundary guarantee (proof vs
measurement, via `requires_proof`) is unchanged; the `eps` payload returns when this is fixed.

### Suspected area
Enum-variant struct-literal construction / field-name resolution in the checker
(`self-hosted/compiler/lean_single.sio`, the E200 emit at ~14486 — the name-resolution fallthrough).

---

## D8 — dereferencing a fixed-array reference (`*arr`) aliases the caller's array [DISTINCT FROM D6]

### Symptom
`var x = *arr` (or `var x: [T;N] = *arr`), where `arr` is a `&[T;N]` reference parameter, ALIASES the
caller's array rather than copying it — mutating `x` afterward silently corrupts the caller's original.
**Silent data corruption, clean compile.** This is distinct from D6 (struct copy-then-mutate,
Sounio-lang/sounio#643): D6 covers struct **parameters** copied by plain `var r = a` assignment; D8
covers **array reference dereference** specifically (`*ref`, not plain array-to-array assignment).
A reader of D6 alone would reasonably but incorrectly conclude arrays are safe from this bug class —
they are not, via the `*`-dereference path. Discovered 2026-08-25 during the Madaros AEAD ciphers plan
(`docs/superpowers/plans/2026-08-25-madaros-aead-ciphers-plan.md`), first as a real bug in
`gcm_increment_counter` (`stdlib/crypto/gcm.sio`): `var result = *block` followed by mutating
`result[12..15]` corrupted the caller's J0 counter block in place, producing a wrong tag against GCM
Test Case 2 while GHASH and AES were both already independently verified correct. Every crypto
primitive written under that plan (`gcm.sio`, `poly1305.sio`) now avoids `*ref` on fixed arrays,
building fresh arrays element-by-element from scalar reads instead.

### Minimal repro (CONFIRMED)
`docs/handoff/repros/d8_array_deref_aliases.sio`:
```sio
fn mutate_alias(arr: &[u8;4]) -> u8 {
    var x = *arr
    x[0] = 99
    return x[0]
}
fn main() -> i64 with IO {
    var original: [u8;4] = [1, 2, 3, 4]
    let r = mutate_alias(&original)
    print_int(r as i64)
    print_int(original[0] as i64)
    return 0
}
```
Expected (correct value semantics): prints `99` then `1`. Observed on Madaros v0.80.0 (2026-08-25):
prints `99` then `99` — `original[0]` was silently corrupted to match `x[0]`.

### Confirmed scope (2026-08-25)
- **Not type-specific**: reproduces identically for `[u8;4]` (above) and `[i64;4]` (verified with an
  equivalent repro swapping the element type) — the defect is in the array-reference-dereference
  mechanism, not tied to any particular element type.
- **A type annotation on the `var` does NOT suppress it**: `var x: [u8;16] = *ref` (or
  `var x: [i64;4] = *arr`) aliases exactly the same as the untyped `var x = *ref` form.
- **Plain array-to-array assignment is unaffected**: `var b = a` or `let c = a`, where `a` is already
  a local array *value* (not a dereference of a reference), DOES copy correctly on this compiler —
  verified: mutating `b` after `var b = a` leaves `a` untouched. The trigger is specifically the
  `*`-dereference of a `&[T;N]` reference, not array assignment in general.

### Impact
Any future call site written against `&[T;N]` parameters that copies via `var x = *ref` and then
mutates `x` while still needing the original is at risk of exactly this class of silent corruption.
The AEAD ciphers plan's own `gcm.sio` and `poly1305.sio` were audited and confirmed to avoid this
pattern throughout (see `gcm_increment_counter`'s own inline comment and `poly1305_mac`'s key-byte
copy loop); `ghash_multiply` (`stdlib/crypto/gcm.sio`) was hardened during the plan's final review to
copy element-by-element rather than via `var v: [u8;16] = *x`, even though its existing call sites
never read the original afterward (harmless today, but a latent trap for a future call site).

### Suspected area
Same general family as D6: local init from a dereferenced aggregate value emits a pointer/alias
instead of a value copy (missing memcpy) — likely `self-hosted/ir/lower.sio` (local init from a
`*`-dereference expression) or the corresponding codegen path, but for the reference-dereference
expression form specifically rather than plain identifier-to-identifier assignment.

### Filed
Not yet filed as a GitHub issue as of this entry (2026-08-25) — track alongside Sounio-lang/sounio#643
(D6) as a related but distinct value-semantics defect.

---

## D9 — cross-struct field-name collision corrupts a returned struct's fixed-array field at a tuple-return boundary

### Symptom
Discovered 2026-08-25 during the Madaros TLS 1.3 handshake plan
(`docs/superpowers/sdd/2026-08-25-madaros-tls13-handshake-plan/`), Task 6
(`stdlib/tls/handshake.sio` + `client_hello.sio` + `server_hello.sio`).
`ServerHelloInfo` and `ClientHelloParams` both declare fields named
`random: [u8;32]` and `x25519_public: [u8;32]` (mandated by the task brief —
the shared field names are intentional, not an accident). When both structs
were compiled into the same final linked program, `decode_server_hello`'s
`x25519_public` field came back corrupted at the caller's tuple-destructure
point immediately after the function returned — even though every field of
the struct read correctly right up to `decode_server_hello`'s own `return`
statement, confirmed via `print_int` probes at every stage inside the
function.

### Minimal repro
No fully principled minimal repro was found despite roughly 2 hours of
bisection work, including building several minimal standalone repro
modules with two structs sharing field names. Per this file's own
precedent (D1, D8), the absence of a clean isolated repro does not exempt
a real, measured defect from being recorded here — it is stated honestly
below rather than omitted.

### Boundary (honest, from direct investigation)
The corruption is **data- and module-composition-dependent, not simply
"two structs with the same field name"**:
- A third struct declared with the same field names sometimes made the
  corruption disappear again — the trigger is not purely "N structs share
  a field name."
- The bug was insensitive to **which file physically declared which
  struct** — moving `ServerHelloInfo` or `ClientHelloParams` to a
  different file did not by itself change the outcome.
- The one variable that did correlate with the corruption: **whether both
  structs ended up compiled into the same final linked program.**

### Exact stdout
Not captured as a standalone transcript (no isolated repro exists to run);
the corruption was observed and confirmed via `print_int` probes bracketing
`decode_server_hello`'s internal field reads (all correct) versus the
caller's tuple-destructured `x25519_public` immediately after the call
returned (wrong), inside the real `tls_handshake_codec_rfc8448.sio` test
during Task 6 development, before the workaround below was applied.

### Impact
Forced `stdlib/tls/handshake.sio` to split into three files instead of one
(`client_hello.sio`, `server_hello.sio`, `handshake.sio`) for Task 6. The
verified, stable workaround is **physical file separation**:
`ServerHelloInfo`/`decode_server_hello` and
`ClientHelloParams`/`encode_client_hello` are declared in different files
that are never both `use`d directly by the same caller — callers instead go
through `handshake.sio`'s `pub use tls::server_hello::*` /
`pub use tls::client_hello::*` re-exports. This specific configuration was
exercised across dozens of runs of the real RFC 8448 handshake-codec test
and held stable.

**Open question this workaround does not close, recorded here for whoever
picks up Task 7**: file separation only avoids putting both struct
*declarations* in one file's own compilation unit — it does not obviously
avoid "both structs compiled into the same final linked program," which is
the condition this investigation actually correlated the corruption with.
Task 7 (not yet built as of this entry) is a future orchestration function
that will necessarily `use` both `ServerHelloInfo` and `ClientHelloParams`
together in the same function, at which point both structs are compiled
into the same linked program regardless of which files declared them. That
is exactly the condition under which this defect was observed. Task 7 must
explicitly re-verify `x25519_public`/`random` integrity once that linkage
exists — the file-separation workaround verified here should not be assumed
to transfer to that shape without a fresh check.

### Suspected area
Field/offset resolution for struct-typed values crossing a function-return
(tuple-destructure) boundary when two or more struct types with identical
field names are live in the same compilation — plausibly struct layout or
field-access lowering in `self-hosted/ir/lower.sio`, or symbol/offset table
construction in the checker (`self-hosted/check/check.sio`), given the
symptom is keyed to whole-program composition rather than to any single
function's local IR. Not investigated further inside compiler internals as
part of Task 6 — this is a workaround-and-record entry, not a root-caused
one.

### Filed
NOT filed as a GitHub issue as of this entry (2026-08-25) — matching D1's
and D8's own honest-negative-result convention for defects without a clean
isolated repro.

---

## D10 — `rawbuf_set` write-clobber applies to out-of-order length-field backpatching, not just limb-to-byte conversion

### Symptom
Discovered 2026-08-25 during the same Madaros TLS 1.3 handshake plan, Task 6.
`encode_client_hello`'s first implementation attempt wrote the message body
first (positions 4 and up), then went back to patch the 4-byte handshake
header at position 0 and the extensions-length field mid-buffer — a classic
out-of-order write pattern with no subsequent write to correct the forward
clobber it causes. This silently corrupted `ClientHelloParams.random` in
the function's own returned buffer: confirmed via `print_int` probes that
`random` was correct immediately after being written, then wrong by the
time the later header-patch write executed.

### Root cause — same mechanism as the existing Finding 20
This is the same `rawbuf_set` write-side defect already documented in this
codebase at `stdlib/x509/cert.sio` (comments around lines 947 and
1218–1236, referred to there as "Finding 20"): each `rawbuf_set` call also
clobbers the 7 bytes *following* the byte it targets, so a buffer written
in a single ascending, uninterrupted pass is self-correcting (each write's
forward clobber is overwritten by the next write in sequence) except for
the buffer's own trailing bytes, which have nothing after them to correct
the clobber. Finding 20 documents this for **out-of-order limb-to-byte
conversion**; this entry (D10) is the same root mechanism surfacing in a
**different failure-mode shape: header/length backpatching** — writing a
message body before going back to patch an earlier length or header field,
rather than converting numeric limbs out of ascending order. This file has
no separate numbered `D`-entry for Finding 20 itself (it exists only as
inline comments in `stdlib/x509/cert.sio`); this entry cross-references it
directly rather than duplicating its explanation.

### Minimal repro
Not isolated as a standalone repro file; reproduced directly in
`encode_client_hello` (`stdlib/tls/client_hello.sio`) during Task 6
development, before the fix described below was applied.

### Exact stdout
Not captured as a standalone transcript. Observed via `print_int` probes
inside `encode_client_hello`: `ClientHelloParams.random` bytes printed
correctly immediately after the body-write loop that wrote them, and
printed incorrectly after the subsequent header-patch (`rawbuf_set` at
buffer position 0) and extensions-length-patch writes executed.

### Impact
Fixed by computing every length value `encode_client_hello` needs
(extensions-total-length, body-length) **analytically, before writing any
byte**, as a pure function of `server_name_len`/`cookie_len` — so the
entire message (header through the last extension) is written in one
single ascending, uninterrupted pass with no backpatching at all. The only
remaining uncorrected clobber lands in the function's own 8 bytes of
trailing allocation slack, which is this codebase's existing standard
mitigation for the Finding-20 write-clobber (the same pattern already used
elsewhere, e.g. `stdlib/x509/cert.sio` lines ~1553–1558). `encode_finished`
(`stdlib/tls/handshake.sio`) already wrote header-then-body in the correct
ascending order but lacked the 8-byte trailing slack; that slack was added
while fixing this defect.

An unrelated, self-inflicted arithmetic bug was found and fixed during the
same debugging pass: the analytical `body_len` formula initially used a
fixed offset of 45 bytes for the pre-extensions part of ClientHello; the
real byte count is 47 (an off-by-2 that produced a correct-looking but
wrong header length field, caught by the test's own length-consistency
assertion). This arithmetic bug is not itself a compiler defect and is
recorded here only for completeness of the investigation trail.

### Suspected area
Same as Finding 20: `rawbuf_set`'s write implementation — see
`stdlib/x509/cert.sio` lines 947 and 1218–1236 for the existing
documentation of the underlying 7-byte-forward-clobber mechanism.

### Filed
NOT filed as a GitHub issue as of this entry (2026-08-25) — the underlying
mechanism (Finding 20) is an existing, already-documented, already-worked-
around behavior; this entry records a new failure-mode shape it produces,
consistent with D1's/D8's honest-negative-result convention for entries
without a dedicated fresh GitHub issue.

---

## D11 — a tuple-destructured local loses its struct type, so a field read resolves by NAME across every struct in the linked program

> **UPDATE 2026-08-26 — RESOLVED.** Fixed in `self-hosted/ir/lower.sio`,
> commit `3ec2d971d` (dispatch `f83b20ce3`). Regression guard:
> `tests/run-pass/tuple_destructure_field_name_collision_regression.sio`.
> The 36-module TLS program now returns `tbs_start=4, tbs_len=523` with an
> unmodified stdlib; the fixed compiler type-checks the whole 120-module
> compiler closure; 886 suite tests across prefixes a-i show byte-identical
> pass/fail to the pre-fix baseline. The end-to-end handshake against a live
> `openssl s_server` has NOT been re-run (needs the server up and the embedded
> test certificate regenerated — it expires 2026-08-27). Rebuild the compiler
> ELF (gitignored) with
> `bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros`.
>
> **Root cause and full trail:
> [`docs/audit/D11_ARENA_SCRATCH_RESET_CROSS_MODULE_CORRUPTION_DISPATCH_2026-08-26.md`](../audit/D11_ARENA_SCRATCH_RESET_CROSS_MODULE_CORRUPTION_DISPATCH_2026-08-26.md).**
> The entry below is preserved as the original investigation trail; two of its
> conclusions are now known to be wrong.
>
> - **It is not a scale/threshold defect and the arena is not involved.** The
>   `arena_reset_skipped (call-arg scratch overflow)` path is *fail-safe* (it
>   declines to reset, so nothing can dangle); the corruption reproduces with
>   `arena_reset_totals ok=35 skip=0`, and in a **6-function, single-file,
>   import-free program** where the arena machinery never runs.
> - **Root cause:** the parser desugars `let (cert_inner, e0) = der_enter(...)`
>   to `let __tup0 = der_enter(...)` + `let cert_inner = __tup0.0`, and
>   `lower_let_stmt_ref` (`self-hosted/ir/lower.sio`) has no rule that records a
>   struct type for a tuple-index field-access initialiser. `cert_inner.pos`
>   then falls through `field_idx_for_base_ref` to `field_idx_from_name_simple`
>   — a global, name-only, **first-registered-match** scan over every struct
>   layout — and picks `HsBuf.pos` (index 3, declared in `stdlib/tls/client.sio`)
>   instead of `DerReader.pos` (index 1). Reading index 3 of a 24-byte
>   `DerReader` lands 8 bytes past its end: deterministically 255, whence
>   `tbs_len = 527 - 255 = 272`. This is exactly **Finding 25**, the gap
>   `docs/audit/X509_ARRAY_STRUCT_FIELD_CORRUPTION_DISPATCH_2026-08-24.md`
>   deliberately left open.
> - **Minimal repro:** `tests/known_failures/madaros_tuple_destructure_field_name_collision_probe.sio`
>   (30 lines, no imports). Swapping the two struct declarations makes it pass —
>   the proof that resolution is declaration-order first-match.
> - **Proof the memory is intact:** an instrumented `der_pos_of(&cert_inner)`
>   (reading `r.pos` through a typed `&DerReader` parameter) returns **4** in the
>   same run where the inline `cert_inner.pos` returns **255**. Only the caller's
>   field-load instruction is wrong.
> - **Fix:** `lower_let_stmt_ref` now recovers element `k`'s struct type at the
>   desugared `let x = __tupN.k`, from a per-callee (interned-**name**-keyed,
>   because fn_ids are remapped by the merge and names are not) table of
>   declared tuple-return element types. Every step is guarded; a miss falls
>   back to the previous behaviour, so it can only add resolution.
> - **Still open:** `field_idx_from_name_simple`'s global first-match fallback
>   itself is unchanged — this fix removes the tuple-destructure route into it,
>   not the fallback. The proposed ambiguity diagnostic is not implemented.
>   Tuple arity > 4 is not covered. See the dispatch's "Still open" section.

### Original entry (2026-08-25), preserved

### Symptom
Discovered 2026-08-25 during the Madaros TLS 1.3 handshake plan, Task 7
(`stdlib/tls/client.sio`, `tls_connect`), while driving a real handshake
against a live local `openssl s_server -tls1_3`. `x509_parse_certificate(buf,
len)` (`stdlib/x509/cert.sio`) returns a `Certificate` whose `tbs_start`/
`tbs_len` fields are WRONG (255/272) instead of the correct values (4/523)
for a real, independently-generated, byte-verified-identical 803-byte RSA
certificate DER buffer — while other fields of the SAME returned struct
(`modulus` — correct 256-byte RSA-2048 modulus, `public_key_algorithm` —
correct `PUBKEY_ALG_RSA`, `outer_signature_len` — correct 256) come back
right. The wrong `tbs_start`/`tbs_len` values make `x509_verify_chain`
(`stdlib/x509/chain.sio`) hash the wrong byte range for the certificate's own
self-signature check, so a certificate that is genuinely self-signed and
genuinely present (byte-for-byte) in its own trust store is rejected with
`CHAIN_ERR_BAD_SIGNATURE` — a real, exploitable-looking failure with no
actual security cause; the certificate and signature are both entirely
valid, and the compiler is just computing the wrong hash input.

### A distinct, more severe symptom found and worked around during the same
### investigation: a real SEGFAULT via a D9-class field-name collision
Before finding the above, the same test SEGFAULTED (rc=139) deeper in the
handshake, inside `bytes_be_to_bigint` while converting `CertificateVerify`'s
signature bytes to a `BigInt` — traced to `cv_info.signature_len` (a `tls::
handshake::CertificateVerifyInfo` field, `i64`) printing a huge garbage value
(54287, not the real 256) immediately after `decode_certificate_verify`
returned, while the SAME tuple's `signature_scheme` field printed correctly
(2052 = `0x0804` = `rsa_pss_rsae_sha256`). This is D9
(`docs/handoff/souc_v0800_defects.md`, above) recurring via a DIFFERENT,
accidental field-name collision than the one D9 itself was filed for:
`x509::cert::Certificate` and `x509::sct::SctEntry`/`x509::cert::SctEntry`
both already declare a `signature_len: i32` field, and `x509::cert::
Certificate` also has a fixed-array `signature: [u8;128]` field — both names
collide with `tls::handshake::CertificateVerifyInfo`'s own (pre-existing,
Task-6-authored) `signature: RawBuf` / `signature_len: i64` fields, and all
of these types are compiled into the same final linked program by
`client.sio` (which needs `x509::cert::Certificate` for chain verification
AND `tls::handshake::CertificateVerifyInfo` for the TLS layer). Renaming
`CertificateVerifyInfo`'s two colliding fields to `cv_signature`/
`cv_signature_len` (`stdlib/tls/handshake.sio`) made the segfault disappear
completely and the crypto-critical CertificateVerify signature check start
succeeding correctly against the real server (confirmed: RSA-PSS-SHA256
verification against the live openssl-signed transcript hash returned
`true`). This is the SAME general defect class as D9 (cross-struct
field-name collision corrupting a struct at a tuple-return/cross-module
boundary), just a fresh, previously-unknown pair of colliding names —
recorded here rather than reopening D9, since D9 is specific to the
`ServerHelloInfo`/`ClientHelloParams` pair and this is a different pair with
a different, additionally worse symptom (SIGSEGV, not just wrong data).
**The rename fix for the segfault is real, verified, and kept in `stdlib/
tls/handshake.sio` and its own RFC 8448 test — it is not a workaround this
entry recommends undoing.**

### The `tbs_start`/`tbs_len` corruption (this entry's primary finding) is NOT explained by D9's field-name-collision mechanism
Exhaustively checked: no other struct compiled into this program declares a
field named `tbs_start` or `tbs_len`. This corruption survives the D9-style
rename fix intact (confirmed: still 255/272 after the `signature`/
`signature_len` rename above). It is a DIFFERENT mechanism from D9.

### Minimal repro / isolation (the load-bearing finding of this entry)
1. A small, standalone test (`x509::chain`/`x509::cert`/`net::socket` only,
   ~20 modules merged, "`Merged IR: 238 functions`" per the build log)
   calling `x509_parse_certificate` ONCE on this exact 803-byte DER buffer
   (embedded as a byte-array literal) returns the CORRECT
   `tbs_start=4, tbs_len=523, outer_signature_len=256`, and a full
   `x509_verify_chain` call against a trust store containing this same
   certificate as its own trust anchor returns `CHAIN_OK` (0).
2. The SAME exact byte content (verified byte-for-byte identical via an
   independent Python diff of all 803 bytes — zero differences), parsed by
   the SAME `x509_parse_certificate` function, but as part of the full
   `tls_client_handshake_loopback.sio` program (36 modules merged,
   `Merged IR: 455 functions`), returns the WRONG `tbs_start=255,
   tbs_len=272` — deterministically and repeatably (confirmed: two
   consecutive calls on the identical buffer within the same run both
   return 255/272, not two different garbage values — this is not random
   heap garbage, it's a wrong-but-consistent computation for this program).
3. This reproduces at the VERY FIRST call the test program makes to
   `x509_parse_certificate` — before `tls_connect` is even called, before
   any networking/crypto has executed — ruling out "prior heavy computation
   corrupts shared state" as the mechanism. The only variable that changed
   between the passing (238-function) and failing (455-function) case is
   the TOTAL SIZE of the merged/linked program.
4. Every build of the failing program's own compiler diagnostics show:
   `lower_array: arena_reset_totals ok=0 skip=35 sites_reclaimed=0` — EVERY
   SINGLE arena reset across all 35 non-test modules was skipped
   (`arena_reset_skipped (call-arg scratch overflow)`, printed once per
   module with an escalating `sites` count, e.g. `module 35 sites 1513`).
   The smaller, passing isolated test's build log shows the same kind of
   message but far fewer sites-per-module and, critically, still succeeds —
   consistent with this being a real, size/allocation-pressure-dependent
   arena/scratch-memory-reuse defect, not a one-off fluke.

### Impact
This blocks `tls_connect` (`stdlib/tls/client.sio`, Task 7 of this plan)
from ever reaching `CHAIN_OK` against a real server at this program's scale
(36 modules / 455 functions merged) — confirmed identically for BOTH an RSA
test certificate (`tests/run-pass/tls_client_handshake_loopback.sio`) and an
independently-generated ECDSA P-256 test certificate (`tests/run-pass/
tls_client_handshake_ecdsa_loopback.sio`), both failing with
`CHAIN_ERR_BAD_SIGNATURE` at the identical point for the identical reason.
The adversarial case (`tests/run-pass/tls_client_adversarial_untrusted_loopback.sio`,
an EMPTY trust store) is NOT affected — it fails closed correctly
(`chain_build_candidates` finds no path to a trusted root before ever
reaching the corrupted-signature-check code, so `x509_verify_chain` returns
`CHAIN_ERR_NO_PATH_TO_ROOT` well before the D11 corruption would matter) —
this is a genuine, unaffected-by-D11 positive security result.

Every other real, live-network-verified piece of the handshake up to this
point works correctly against the actual openssl server: ClientHello
encode/send, ServerHello receive/decode (with the D9 x25519_public/random
verification below passing), the full TLS 1.3 key schedule (HKDF Early/
Handshake Secret, traffic secrets, traffic key/IV derivation), AES-128-GCM
record encryption/decryption of the real EncryptedExtensions message, TLS
Certificate message decode, and CertificateVerify's own RSA-PSS-SHA256
signature verification against the live transcript hash (a real
cryptographic check, genuinely passing). Only the X.509 chain's own
self-signature check — a different code path, using `Certificate.tbs_start/
tbs_len` rather than the TLS transcript — is affected.

### Suspected area
`self-hosted/compiler/module_frontend.sio` / `self-hosted/ir/lower.sio`'s
arena/scratch allocation-and-reset machinery for imported/cross-module
calls (the `lower_array: arena_reset_skipped (call-arg scratch overflow)`
diagnostic family) — the same general subsystem already implicated in
`stdlib/x509/chain.sio`'s own in-code comment about needing to inline
`chain_build_candidates` to avoid a related (but distinct-symptom) defect,
and in this repo's CLAUDE.md SS13 "residual D3 family: multi-module
memory-wall / exclusive-ref fragile chains" note. Not investigated further
inside compiler internals as part of Task 7 — this is a workaround-and-record
entry per this file's own established convention (D1, D8, D9), not a
root-caused one.

### Filed
NOT filed as a GitHub issue as of this entry (2026-08-25) — no isolated
minimal repro below the size of this actual 36-module program was found in
the time available (attempts to reproduce with smaller synthetic multi-
module programs of increasing size were not completed); recorded here per
this file's own honest-negative-result convention (D1, D8, D9) so the next
agent has the full forensic trail rather than rediscovering it. **This
defect BLOCKS Task 7 of the Madaros TLS 1.3 handshake plan from reaching a
fully working real-server handshake and is reported as such in that task's
own completion report.**

---

## D12 — real-CA TLS handshakes exhaust the never-reclaimed process arena after 2 connections (exit 181)

**Status:** RESOLVED for the reported workload (ceiling **2 → 95**
handshakes/process). The underlying lifetime defect is UNCHANGED and still
open — the wall moved ~47×, it did not disappear.

> **Amended 2026-08-27 — this entry's ceiling is real but not the whole
> budget. See [D16](#d16--trust_store_load-costs-13-gb-of-never-reclaimed-arena-and-a-process-survives-only-six-calls-exit-181).**
> The "95 handshakes/process" figure is measured with the trust store loaded
> **exactly once, outside** the probe's handshake loop
> (`tests/interop/tls_arena_multi_handshake_probe.sio:36`). That is the
> correct usage — and therefore the one that never measures the cost of the
> wrong usage. A single `trust_store_load()` costs **~1.3 GB** (150 real CA
> roots × this entry's own per-parse figure), so a process admits only **six**
> loads. Do not read "95 handshakes" as "95 TLS-using operations": a consumer
> that loads the trust store per stage exhausts the arena long before the
> 95th handshake, which is exactly what `conclave-search` did on one query.
> The two secondary figures below are also refined by D16's measurements:
> `certificate_zero()`'s cost does **not** apply to the
> `[certificate_zero(); N]` array-repeat form (measured: ~0), and the
> per-handshake cost is chain-dependent by ~4.8× (~70 MB vs `8.8.8.8`,
> ~335 MB vs `1.1.1.1`), not a single constant.

**Full forensic dispatch:**
[`docs/audit/ARENA_EXHAUSTION_TLS_HANDSHAKE_CHAIN_VERIFICATION_DISPATCH_2026-08-26.md`](../audit/ARENA_EXHAUSTION_TLS_HANDSHAKE_CHAIN_VERIFICATION_DISPATCH_2026-08-26.md)

This is **Finding 12** of
[`docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`](../audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md)
("the Madaros runtime arena is never reclaimed") realised on the TLS path,
with a real measurement in place of that finding's projection. Finding 12
predicted the ceiling "will return for any long-running process (e.g. a TLS
server handling many handshakes)"; it returned at **two** handshakes, not at
the ~460,000 `bigint_add` calls its own headline figure suggests.

### Symptom
```
arena_probe: handshake attempt 3
madaros: arena full          (exit 181, uncatchable)
```
Repro: `tests/interop/tls_arena_multi_handshake_probe.sio`.

### Why chain verification is so expensive
- `GeneralName` carried `directory_name: X509Name` — **never populated,
  never read** (the parser deliberately skips copying it because a
  doubly-indexed struct-in-array write corrupts on this compiler). It was
  ~87% of every `Certificate`, since `Certificate` holds 64 `GeneralName`s.
- `stdlib/x509/chain.sio` copied `Certificate` **values** in the hot path,
  including once per trust-store root (150) per DFS step.
- **A `[u8; N]` field costs 8N bytes** — one 8-byte slot per element. Every
  size estimate in this file made against logical struct size is 8× low.

### Cost attribution (measured, `tests/interop/x509_arena_cost_probe.sio`)
| Operation | Iterations/process | Implied cost |
|---|---:|---:|
| `certificate_zero()` | 5,254 | ~352 KB |
| `x509_parse_certificate()` | 161 | ~11.5 MB |
| `x509_verify_chain()` | 44 | ~30 MB |

### Fix
| Commit | Change | Ceiling |
|---|---|---:|
| `c9bd996b2` | drop `GeneralName.directory_name` | 2 → 11 |
| `976e3e399` | `&Certificate` refs + virtual issuer pool in `chain.sio` | 11 → 11 |
| `eea3a449f` | arena 2 GiB → 8 GiB, handle table 2^22 → 2^24 (Linux) | 11 → **95** |

`976e3e399` showing no end-to-end movement is reported as measured, not
hidden: the attribution table above shows parsing and crypto dominate path
building. It was kept because it removes a real ~50 MB/verification cost.

### Filed
Not a GitHub issue. Recorded here and in the dispatch per this file's
convention. **Next highest-value work: `x509_parse_certificate`'s ~11.5 MB
per call**, which is now the single largest lever and is mostly parser
scratch rather than the returned value.

## D13 — `env_get` discards its output buffer and reports success unconditionally on hit

`stdlib/os/env.sio::env_get` (line 24) takes `out: &![u8; 256]` but never
writes to it — the parameter is discarded with `let _ = out` on line 26. The
function still returns `0` (success) whenever the underlying `getenv(key)`
succeeds, so every caller sees "found" with a permanently-empty/garbage
`out` buffer. This is a fail-open no-op, not a fail-closed error: callers
cannot detect it from the return code alone.

Found while reviewing `Sounio-lang/conclave-search`'s Task 7 (CLI
orchestrator) — it could not use `env_get` for host/port configuration and
fell back to argv-only configuration as a workaround. Not fixed here: out
of scope for that repo, and this file is shared stdlib.

## D14 — no connect or receive timeout anywhere in `stdlib/net`/`stdlib/tls`

`stdlib/net/socket.sio::tcp_connect` (line 124) calls the raw `connect`
syscall with no `SO_RCVTIMEO`/`SO_SNDTIMEO` socket option set anywhere in
this file, and `stdlib/tls/client.sio`'s `tls_connect`/`tls_recv` inherit
that: a black-holed (silently dropped, as opposed to actively refused) host
or a peer that accepts but never sends hangs the calling process
indefinitely, with no way for the caller to bound the wait.

Found and independently confirmed while reviewing `Sounio-lang/conclave-search`'s
Task 7 — a CLI that fetches attacker- or network-controlled URLs (e.g. from
search results) has no way to bound a single fetch's wall-clock cost.
Distinct from D12 (arena exhaustion): this is a missing timeout primitive,
not an allocation-budget defect. Not fixed here: out of scope for a
consumer repo, and no CLI flag or stdlib option currently exists to opt in
to a bounded wait.

## D15 (fixed) — `x509_verify_hostname` never matched `iPAddress` SubjectAltName entries

> **UPDATE 2026-08-27 — the ECDSA-SHA384 gap (item 3 below) is RESOLVED, and
> item 3's root cause for `1.1.1.1` is CORRECTED.** Full dispatch:
> [`docs/audit/D20_ECDSA_SHA384_X509_SIGNATURE_VERIFICATION_DISPATCH_2026-08-27.md`](../audit/D20_ECDSA_SHA384_X509_SIGNATURE_VERIFICATION_DISPATCH_2026-08-27.md).
>
> **Fixed:** `x509_verify_signature`'s unconditional `return false` for
> `ecdsa-with-SHA384` is gone. `stdlib/crypto/ecdsa_p256.sio` gained
> `ecdsa_p256_verify_sha384`, applying FIPS 186-4 §6.4.1's leftmost-256-bits
> truncation to SHA-384's 48-byte digest; both ECDSA entry points now share
> one private verification core (DER parsing, `1 ≤ r,s < n`, curve math), so
> the reviewed security steps exist in exactly one copy. Proven against a
> real chain generated **and independently validated** by OpenSSL 3.5.6
> (`openssl verify … OK`), pinned as
> `tests/run-pass/x509_chain_ecdsa_sha384_signature.sio`. Two deliberate
> mis-truncations (rightmost-32, off-by-one) both make that test fail, so it
> genuinely pins the rule. Suite: `x509` 24 → 26 pass, 0 fail, 3 skip against
> a clean-tree baseline on the identical invocation; `tls`/`crypto`/`pem`
> unchanged.
>
> **Corrected — and this matters more than the fix:** item 3 below names the
> SHA-384 hash as the reason `1.1.1.1` failed. That is the smaller half.
> Fetching the real chain shows the leaf is signed by **SSL.com SSL
> Intermediate CA ECC R2, whose key is P-384** — verifying that link needs a
> different *curve*, not a different *hash*, and only P-256 is implemented.
> Measured on the real bytes: both certificates parse `X509_OK`, the leaf's
> key is `EC_P256`, the intermediate's is `PUBKEY_ALG_UNKNOWN`, and
> `x509_verify_signature` returns `false` — the new algorithm-confusion guard
> correctly refusing to run a P-384 key through P-256 arithmetic. Pinned as
> `tests/run-pass/x509_ecdsa_sha384_p384_issuer_rejected.sio`.
>
> **So `1.1.1.1` is still blocked (`tls_connect` → `-8`), and DoH with it.**
> The remaining work is **P-384 curve support**, not a hash path. Live
> measurement: `tests/interop/tls_connect_1111_sha384_probe.sio`. Anything in
> this file that cites "D15's ECDSA-SHA384 gap" as the `1.1.1.1` blocker
> (lines below, and D19's entry) should be read as "D15's P-384 gap".
>
> **CLOSED 2026-08-27 — the residual P-384 gap D20 found is now resolved.**
> Full spec:
> [`docs/superpowers/specs/2026-08-27-madaros-ecdsa-p384-design.md`](../superpowers/specs/2026-08-27-madaros-ecdsa-p384-design.md).
> NIST P-384 (secp384r1) curve constants and a shared-core verifier landed
> in `stdlib/crypto/ecdsa_p384.sio` (`ecdsa_p384_verify_sha384`), the
> secp384r1 OID in `stdlib/x509/oid.sio` (`oid_secp384r1`), 97-byte
> uncompressed-point public-key parsing and `PUBKEY_ALG_EC_P384`
> recognition in `stdlib/x509/cert.sio`, and dispatch by issuer key
> algorithm (P-256 vs P-384, fail-closed for anything else) in
> `x509_verify_signature` plus the matching trust-anchor guard in
> `stdlib/x509/trust_store.sio`. Proven three ways: (1) a real,
> openssl-generated, `openssl verify`-confirmed P-384 root signing a P-256
> leaf (the same shape as SSL.com's real intermediate) verifies end to end
> — `tests/run-pass/x509_chain_ecdsa_p384_signature.sio`; (2) the same
> P-384-signed leaf is correctly *rejected* against an unrelated, real
> P-256 root, proving the dispatch is a genuine per-curve check and not
> "any EC key accepted" — `tests/run-pass/x509_chain_ecdsa_p384_cross_curve_rejected.sio`;
> (3) the original motivating live case now succeeds for real:
> `tls_connect` against Cloudflare's `1.1.1.1:443` (SSL.com SSL
> Intermediate CA ECC R2, a genuine P-384 issuer) returned **rc 0**,
> "1.1.1.1 chain VERIFIED", re-measured live on 2026-08-27 — see the
> updated `tests/interop/tls_connect_1111_sha384_probe.sio`.
>
> **Fix-round update:** the stale test flagged above,
> `tests/run-pass/x509_ecdsa_sha384_p384_issuer_rejected.sio` (committed
> before P-384 recognition existed, asserting the real Cloudflare
> intermediate stayed `PUBKEY_ALG_UNKNOWN` and that
> `x509_verify_signature` returned `false`), has now been updated in place
> and renamed to `tests/run-pass/x509_chain_ecdsa_p384_cloudflare_accepted.sio`.
> It keeps its real, live-fetched Cloudflare leaf/intermediate DER bytes —
> the single most valuable fixture in this plan, the literal motivating
> case — but now asserts the correct, post-fix behaviour:
> `inter.public_key_algorithm == PUBKEY_ALG_EC_P384` and
> `x509_verify_signature(...) == true`. Suite re-verified green with this
> flip in place (see this plan's task-6-report.md, fix round).

**Symptom chain, and two wrong diagnoses corrected along the way**: while
testing `Sounio-lang/conclave-search`'s Task 9 (a DNS-over-HTTPS resolver)
against Cloudflare's real, publicly-trusted `1.1.1.1:443` endpoint,
`tls_connect` returned `TLS_CONNECT_ERR_CERT_CHAIN` (-8). This symptom was
independently misdiagnosed twice before the real cause was found by direct
instrumentation:

1. The `conclave-search` task implementer attributed it to D11 (this
   file's tuple-destructure field-resolution bug) via a function-count
   correlation. **Refuted**: D11's own fix (`3ec2d971d`) is present in the
   binary that produced this failure, and a direct re-parse of the exact
   certificate in question (`tbs_start=4, tbs_len=1292`, cross-checked
   against `openssl asn1parse`) showed no TBS corruption at all.
2. A second review attributed it to `x509_verify_hostname`
   (`stdlib/x509/chain.sio`) only ever checking `GENERAL_NAME_DNS_NAME` SAN
   entries and never `GENERAL_NAME_IP_ADDRESS` — measured directly:
   `verify_hostname("1.1.1.1")` returned `false` in isolation. **This gap
   is real and has been fixed** (commit `7b819069e`: `x509_verify_hostname`
   now parses the connect hostname as dotted-decimal IPv4 and compares it
   octet-for-octet against any `iPAddress`-tagged SAN). But fixing it did
   **not** change the `1.1.1.1` symptom — instrumenting the real
   `chain_result` (not just the collapsed `TLS_CONNECT_ERR_CERT_CHAIN`)
   showed `CHAIN_ERR_BAD_SIGNATURE` (-6), returned well before
   `x509_verify_hostname` is ever reached for this chain.
3. **Actual root cause**: Cloudflare's certificate chain for `1.1.1.1` is
   signed `ecdsa-with-SHA384` (an SSL.com ECC intermediate). This is a
   pre-existing, already self-documented scope limitation from the
   original TLS handshake plan: `x509_verify_signature`
   (`stdlib/x509/cert.sio` ~line 1566) explicitly fails closed on
   `ecdsa-with-SHA384` with an inline comment stating the SHA-384 hashing
   path for `ecdsa_p256_verify` was never built (only SHA-256 was, per
   that plan's Task 2). This is **not new** and **not fixed here** — it is
   a known, deliberate gap, now confirmed to be reachable against a
   real-world, commonly-deployed CA configuration (SSL.com's ECC
   intermediates routinely sign with SHA-384), not just a theoretical one.

**Net effect**: the `iPAddress`-SAN fix (item 2) is real and kept — it
unblocks any future target whose leaf/intermediate chain signs with
`ecdsa-with-SHA256` or RSA. But `1.1.1.1` specifically, and any other
target with an SHA-384-signed link in its chain, remains blocked on the
pre-existing ECDSA-SHA384 gap, not on anything newly found here.

**Filed**: fix at `7b819069e`. The SHA-384 gap itself is not re-filed as a
new entry — it was already self-documented at its own definition site in
`stdlib/x509/cert.sio` when the original TLS plan scoped ECDSA to SHA-256
only; this entry exists to record that it is now confirmed load-bearing
against real-world traffic, not just theoretical.

## D16 — `trust_store_load()` costs ~1.3 GB of never-reclaimed arena, and a process survives only six calls (exit 181)

**Status:** RESOLVED for the reported workload (ceiling **6 → unbounded**
loads/process, via a process-lifetime memo). The underlying lifetime defect
is UNCHANGED and still open, and the post-fix budget for the reported
workload is adequate but not comfortable — see the dispatch's residual-risk
section.

**Full forensic dispatch:**
[`docs/audit/D16_TRUST_STORE_LOAD_PER_STAGE_ARENA_EXHAUSTION_DISPATCH_2026-08-27.md`](../audit/D16_TRUST_STORE_LOAD_PER_STAGE_ARENA_EXHAUSTION_DISPATCH_2026-08-27.md)

### Symptom
`Sounio-lang/conclave-search` aborts on a **single ordinary query**, far short
of D12's 95-handshake ceiling:
```
conclave-search: fetch_memory_context failed (unreachable beagle-core, or no atoms)
conclave-search: continuing with empty enrichment
madaros: arena full          (exit 181)
```
Not an OOM — 10 Gi container limit, container does not crash, the Madaros
runtime aborts internally.

### Root cause — distinct from D12, not a recurrence of it
D12 measured the per-HANDSHAKE cost. This is a per-TRUST-STORE-LOAD cost that
D12's probe was structurally blind to, because
`tests/interop/tls_arena_multi_handshake_probe.sio:36` loads the store exactly
**once, outside** its handshake loop — the correct usage, and therefore the one
that never measures the wrong usage's cost.

`trust_store_load()` parses all **150** roots of the real Debian CA bundle, at
D12's own ~8.8–11.5 MB per `x509_parse_certificate`: **~1.3 GB per load**.
`conclave-search` calls it once per TLS-using stage — including **once per
candidate URL** in `search/dns_resolve.sio:248` (`resolve_a_record`) — so one
query issues `4 + N_candidates` loads, up to **14** at `MAX_RESULTS = 10`.
The process admits six; it dies at load #7, the fourth candidate's DoH
resolution.

### Measurements (`tests/interop/trust_store_load_arena_cost_probe.sio`)
| Measurement | Value |
|---|---:|
| `trust_store_load()` calls before exit 181 | **6** (7th aborts) |
| Implied cost per load (8 GiB arena) | ~1.3 GB |
| `TrustStore` value construction alone | 20,000 iterations, **no exhaustion** (~0) |
| Real handshakes vs `8.8.8.8` (rc 0) | 95 → ~70 MB each |
| Real handshakes vs `1.1.1.1` (rc −8, D15's ECDSA-SHA384 gap) | **20 → ~335 MB each** |

Two secondary findings, both measured: the `[certificate_zero(); 512]`
array-repeat is **not** 512 allocations (a natural suspicion, refuted), and a
**failing** chain verification is not cheap — Cloudflare's chain costs ~4.8×
Google's per handshake and still fails.

### Fix
`stdlib/x509/trust_store.sio`: the previous body is preserved verbatim as
`trust_store_load_uncached()`; `trust_store_load()` now memoises it in
module-level state (failures memoised too — neither failure mode can be fixed
by a retry, and a retry costs another 1.3 GB). Sound because the store is
immutable after load and its `bufs` are arena handles that are never
reclaimed, so nothing can dangle. Returning the memo by value is arena-cheap
on this compiler — **measured**: 3,000 consecutive returns did not move the
arena.

Pinned by `tests/run-pass/trust_store_load_cached_identity.sio`: field-for-field
comparison of cached vs fresh across all 150 roots, live trust-store queries
against the cached store, and 50 loads in one process (fatal before, free
after). `tests/run-pass/{x509_*,tls*,pem_*}`: **35 pass, 0 fail**.

### Filed
Not a GitHub issue. Recorded here and in the dispatch per this file's
convention. **Next highest-value work is unchanged from D12's own nomination:
`x509_parse_certificate`'s ~8.8–11.5 MB per call**, now the sole term in a
trust-store load and the dominant term in every handshake.

## D17 — the reported empty-Subject parsing defect does not exist; the EC trust-anchor bypass found while disproving it does

**Status:** reported defect **NOT REPRODUCED** (hypothesis refuted by
measurement in four configurations); a **separate, security-critical
chain-validation bypass** found during the investigation is **RESOLVED**.

**Full forensic dispatch:**
[`docs/audit/D17_EMPTY_SUBJECT_CERTIFICATE_AND_EC_TRUST_ANCHOR_DISPATCH_2026-08-27.md`](../audit/D17_EMPTY_SUBJECT_CERTIFICATE_AND_EC_TRUST_ANCHOR_DISPATCH_2026-08-27.md)

### Reported symptom
The live `searxng-https-tls` leaf (namespace `beagle`) — a real
`cert-manager` certificate with an **empty Subject DN**, valid
`Aug 27 01:09:33 2026` → `Nov 25 01:09:33 2026` — rejected by
`x509_verify_chain` with `CHAIN_ERR_EXPIRED` (−2) at ~`Aug 27 09:00 2026`.
Hypothesis under test: the zero-length Subject `SEQUENCE` throws off
`x509_parse_certificate`'s Validity byte offsets.

### Part 1 — refuted, structurally and by measurement
The hypothesis cannot hold: in a `TBSCertificate` the Validity field
**precedes** the Subject (RFC 5280 §4.1.2). `openssl asn1parse` on the
reported certificate puts the two `UTCTIME`s at offsets **89** and **104**
and the empty Subject at **119** — the Subject's length is not an input to
any offset the date decode uses.

Measured live on this worktree's `bin/souc` (Madaros v0.80.0):

| Configuration | Result |
|---|---|
| `x509_parse_certificate` standalone | `status = 0`, dates **exactly** `1787792973` / `1795568973` |
| `x509_verify_chain`, trust store = the real internal CA | **`CHAIN_OK`** |
| `x509_verify_chain`, trust store = `trust_store_load()`'s 150 real Debian roots **+** the internal CA | **`CHAIN_OK`** |
| Live **TLS 1.3 handshake** via `tls_connect` against an equivalently-shaped empty-Subject P-256 leaf with an IP SAN | **`err = 0`** |

On the −2 itself: `grep -rn "= 0 - 2$" stdlib/` returns **ten distinct
constants** across `chain`/`cert`/`ocsp`/`trust_store`/`pem`/`der`/`socket`/
`tls`, so a bare −2 is not evidence for `CHAIN_ERR_EXPIRED` specifically —
the same lesson D15 filed. Two measured data points: a too-early `now_unix`
yields **−3** (`NOT_YET_VALID`), not −2; and a `GeneralizedTime`-encoded
validity field **does** yield a genuine spurious `CHAIN_ERR_EXPIRED` (the
parser is UTCTime-only), which remains the leading candidate for any real
instance of this symptom — though not for the reported certificate, whose
dates are both UTCTime.

### Part 2 — the real defect (found in the code the reporter was exercising)
`chain_is_trusted_root`'s "byte-identical public key" requirement — added
2026-08-25 to close a trust-anchor forgery bypass — compared **only**
`modulus` and `public_exponent`, which `x509_parse_certificate` **never
populates for an EC certificate** (a P-256 key goes to `ec_public_key`; both
BigInts stay `bigint_zero()`). For every EC certificate the guard was
`bigint_cmp(zero, zero) == 0` twice — vacuously true — so it silently
degraded to the pre-fix "any self-signed cert whose Subject DN matches"
behaviour. Same hole for `PUBKEY_ALG_UNKNOWN`.

That is the key type this entire sub-project uses (P-256 chosen deliberately
to dodge D15's ECDSA-SHA384 gap).

**Proof of concept, executed.** Genuine P-256 root (key A) alone in the trust
store; attacker's self-signed P-256 root with the **identical Subject DN** and
key B supplied as an intermediate; leaf issued by the forged root:

```
before:  verify result = 0    (CHAIN_OK -- attacker root accepted)
after:   verify result = -6   (CHAIN_ERR_BAD_SIGNATURE -- rejected)
```

### Fix
`stdlib/x509/trust_store.sio`: `trust_store_has_matching_key_for_subject` now
takes the whole candidate `&Certificate` instead of
`(subject, modulus, exponent)`, and requires, on the **same** entry, matching
`public_key_algorithm` **and** matching Subject DN **and** byte-identical key
material *for that algorithm* (RSA modulus+exponent, or `ec_public_key.x`/`.y`).
Algorithms this parser does not decode are rejected up front, fail-closed —
all their key fields are zero, so any two would compare equal. Sole non-test
caller updated (`chain.sio:145`). The rewrite also drops a
`let entry = store.certs[i]` by-value bind that burned ~6.9 MB of
never-reclaimed arena per call (~46 KB × ~150 roots), against this module's
own ARENA DISCIPLINE note and D16.

### Regression tests
- `tests/run-pass/x509_empty_subject_certificate.sio` — the real live
  empty-Subject leaf + its CA: exact dates, `subject.count == 0`, everything
  after the empty Subject still decoding, `CHAIN_OK` inside the window **and**
  `CHAIN_ERR_EXPIRED` one second past `notAfter` (so the positive assertion is
  a live check, not one that never runs).
- `tests/run-pass/x509_chain_forged_ec_root_dn_collision.sio` — the EC half of
  the forged-root bypass, mirroring the RSA test, with its preconditions
  asserted explicitly so it cannot silently stop testing them. `-days 8400`
  fixtures, fixed `now_unix`: it does not rot.

### Test evidence
`x509` **24 pass / 0 fail** · `tls` **7/0** · `pem` **2/0** ·
`trust_store` **2/0**. The 151-root chain verification and the live TLS 1.3
handshake were both re-run after the fix and are unchanged — no legitimate EC
trust anchor is broken.

### Filed
Not a GitHub issue. Recorded here and in the dispatch per this file's
convention. **The originally reported failure remains unexplained**: this
entry refutes the stated mechanism, it does not explain what the reporter
observed. If it recurs, print the `x509_verify_chain` return value *together
with the name of the function that produced it*, plus `not_before_unix`,
`not_after_unix` and `now_unix` from the same process, before re-opening a
parser hypothesis.

## D18 — the live-handshake `CHAIN_ERR_EXPIRED` is a scalar-field offset defect on `ChainCandidate`, not a handshake-framing defect

**Status:** reported symptom **ROOT-CAUSED and FIXED** (Parts 1 and 2 stand).
**Part 3 is WITHDRAWN — see D19.** Its claimed second defect
(`x509_parse_certificate` returning `tbs_start`/`tbs_len` = 255/44) does not
reproduce on any surface, including the very endpoint it was measured on, and
its conclusion that "a real TLS handshake against these endpoints cannot
complete until this compiler defect is fixed" is false: the handshake returns
`rc 0` / `CHAIN_OK`. Do not open work against Part 3. Read
[D19](#d19--d18-part-3-refuted-and-the-arena-full-traced-to-a-stale-2-gib-binary)
first.

**Full forensic dispatch:**
[`docs/audit/D18_CHAIN_CANDIDATE_SCALAR_FIELD_OFFSET_DISPATCH_2026-08-27.md`](../audit/D18_CHAIN_CANDIDATE_SCALAR_FIELD_OFFSET_DISPATCH_2026-08-27.md)

### Reported symptom
D17 refuted the empty-Subject hypothesis by testing the parser on a
statically-extracted PEM, but never drove a live handshake. Re-run through a
real `tls_connect` against the deployed `10.96.250.10:443`, on D17's own fix
commit `ece1420be` and a byte-identical CA bundle, it still fails:
`chain_result = -2` for a chain `openssl verify` accepts.

### Part 1 — the framing hypothesis is refuted
The proposed cause was that `decode_certificate_message` mis-strips the TLS
1.3 CertificateEntry framing (3-byte cert length, 2-byte extensions block).
Measured on the live wire bytes, it does not:

| Measurement | Expected | Observed |
|---|---:|---:|
| `cert_len` | 398 | **398** |
| first 4 bytes of `leaf_buf` | `30 82 01 7D` | **48, 130, 1, 125** |
| `leaf_buf.cap` | 393 | **393** |
| byte-sum of `leaf_buf[0..385]` | 30515 | **30515** |
| `intermediate_count` | 0 | **0** |

The reference values come from the server's own DER, captured separately with
`openssl s_client -showcerts`. The framing also reconciles exactly:
`4+1+3+3+385+2 = 398`, nothing left over. **Extraction is byte-perfect.**

### Part 2 — actual root cause of the −2 (fixed)
`ChainCandidate` is `{ certs: [Certificate; 11], bufs: [RawBuf; 11], len: i32 }`
— ~506 KB per element, ~8 MB for the `[ChainCandidate; 16]` array. **Scalar-field
access through an element of that array resolves to a wrong offset landing
inside the element's own `bufs`:**

| Measurement | Expected | Observed |
|---|---:|---:|
| `rc.len` at record time (plain local) | 2 | **2** ✓ |
| `candidates[i].len` right after `candidates[i] = rc` | 2 | **503099** ✗ |
| `candidates[0].len` straight out of the `len: 0` initialiser | 0 | **495821** ✗ |
| `candidates[0].bufs[0].cap` (from `rawbuf_new(1)`) | 1 | **255** ✗ |
| `path_len` reaching `chain_verify_path` | 2 | **504725** ✗ |

The garbage tracks arena state (it moves run to run; the two candidates differ
by a constant 428 every time) — they are arena pointers. `chain_verify_path`
then walks past the real path into zeroed slots whose `not_after_unix` is `0`,
and `now_unix > 0`, so it reports `CHAIN_ERR_EXPIRED`.

Reads and writes share the same wrong offset, so they are self-consistent —
which makes the obvious repair a trap, and it was measured and rejected:
`candidates[0].len = 7` reads back `7`, but clobbers `bufs[0].cap` from 1 to 0
and makes `bufs[10].cap` **segfault**. Taking a *reference* to a whole array
field (`&candidates[c].certs`) is unaffected.

**Why D17 could not see it:** it is a cross-module / imported-module native
path defect (CLAUDE.md §13's D3/D4 family). From a test module the same calls
give `CHAIN_OK`. `chain.sio`'s own 2026-08-25 comment already recorded this
symptom — a known-good chain returning −2 — and guessed "a second, still-live
instance of the same defect class". It was right; this entry gives the
mechanism. **The existing `tests/run-pass/x509_*` tests structurally cannot
catch it**, and no new single-module test can either.

**Fix:** `x509_verify_chain`'s inlined DFS no longer materialises paths into
`[ChainCandidate; 16]`. Each path is verified in place, at the moment the DFS
reaches a trusted root, while `path_certs`/`path_bufs`/`depth` are still plain
locals. Search order, first-`CHAIN_OK` early return, hostname/OCSP checks, the
`MAX_CHAIN_CANDIDATES` cap and "deepest failure wins" reporting are all
preserved. It also drops ~8 MB of never-reclaimed arena per verification
(D12/D16). `chain_build_candidates`, the non-shipping duplicate, is left alone
per the standing instruction that kept the inlined copy.

### Part 3 — the next blocker (characterised, NOT fixed)
With the path length correct, verification proceeds and fails
`CHAIN_ERR_BAD_SIGNATURE` (−6), then exhausts the arena on the second
candidate. This is **not** a crypto gap: the chain is `ecdsa-with-SHA256` over
P-256 throughout (deliberately chosen to dodge D15's SHA-384 hole) and
`openssl verify` returns OK.

Instrumented **at `x509_parse_certificate`'s return** inside
`stdlib/tls/handshake.sio`:

| Field of the returned `Certificate` | True | Observed |
|---|---:|---:|
| `tbs_start` | 4 | **255** ✗ |
| `tbs_len` | 295 | **44** ✗ |
| `not_before_unix` | 1787792973 | **1787792973** ✓ |
| `not_after_unix` | 1795568973 | **1795568973** ✓ |

Two things make this distinct from Part 2: it is corrupt **at the callee's
return**, before any tuple/array/struct copy — so the large 6-tuple return is
**exonerated**, not implicated; and the corrupt fields are `Certificate`'s
**2nd and 3rd** members while much later members are correct, so it is not a
monotonic layout divergence. `x509_verify_signature` hashes `tbs_start..+tbs_len`,
so it hashes the wrong 44 bytes and fails.

**Consequence: a real TLS handshake against these endpoints cannot complete
until this compiler defect is fixed**, regardless of further `stdlib/` work.
`10.96.250.20:443` presents an identically-shaped leaf and behaves the same.

**This is D2's family** (this file, "large-aggregate struct cross-module
arity-mismatch segfault"): fixed-size-array struct, corrupt across a module
boundary, and D2's own repros crash **only** cross-module, never
single-module — the same split seen here. Start from
`docs/handoff/repros/d2_*.sio`, not from scratch. D2's variable is aggregate
*parameter* arity; the natural next experiment is aggregate *return* values,
which is the shape `x509_parse_certificate` has.

Four single-module synthetic repros failed to reproduce the offset defect
(including at the real ~46 KB element magnitude with the real field shape);
the one shape that did misbehave returned **255** — the same value seen for
`bufs[0].cap` and `tbs_start` on the live path.

### Test evidence
Worktree compiler (`SOUC_BIN=$PWD/bin/souc`), `SOUNIO_TEST_JOBS=4`, only
`chain.sio` differing between columns:

| Pattern | Clean tree (`ece1420be`) | With the D18 fix |
|---|---|---|
| `x509` | 11 pass / 13 fail / 3 skip | **13 pass / 11 fail / 3 skip** |
| `tls` | — | **7 / 0** |
| `pem` | — | **2 / 0** |

No regressions — the 11 remaining failures are a **strict subset** of the
baseline 13; `x509_chain_adversarial.sio` and
`x509_chain_forged_root_dn_collision.sio` flip fail → pass. The 11 are
pre-existing and consistent with Part 3 (`tests/run-pass/` programs are
cross-module too); `x509_chain_verify_positive.sio` now fails one step later,
at `-7` rather than in path verification.

> Two harness notes that cost time here. (1) `run_sio_test_suite.sh` defaults
> to the **shared** `/workspace/sounio/bin/souc`, not your worktree's — from a
> worktree that fails *every* test, including ones unrelated to your diff,
> which reads as a catastrophic regression. Always pass
> `SOUC_BIN=$PWD/bin/souc SOUNIO_STDLIB_PATH=$PWD/stdlib`. (2) It defaults to
> `nproc` parallel jobs (64 on this pod) and **recycled the workspace pod**
> mid-run, exactly as CLAUDE.md §4 warns; use `SOUNIO_TEST_JOBS=4`.
>
> D17 recorded `x509` at 24 pass / 0 fail; re-measured at D17's own commit it
> is 11 pass / 13 fail / 3 skip. Most likely the two figures are measurements
> of two different binaries (see note 1), not a regression between them —
> flagged rather than silently reconciled.

### Filed
Not a GitHub issue. Recorded here and in the dispatch per this file's
convention. ~~**Next highest-value work: the Part 3 compiler defect**~~ —
**withdrawn, see D19.** Part 3 does not reproduce; the handshake it declared
impossible now completes.

---

## D19 — D18 Part 3 refuted, and the `arena full` traced to a stale 2 GiB binary

**Status:** **NOT A DEFECT.** The dispatched compiler defect does not exist on
this branch; the live TLS handshake it declared impossible **succeeds**; and
the `madaros: arena full` regression is a **stale deployed artefact**. No
`self-hosted/` change was made, and none is warranted.

**Full forensic dispatch:**
[`docs/audit/D19_AGGREGATE_RETURN_REFUTATION_AND_STALE_ARENA_DISPATCH_2026-08-27.md`](../audit/D19_AGGREGATE_RETURN_REFUTATION_AND_STALE_ARENA_DISPATCH_2026-08-27.md)

### The dispatched claim
D18 Part 3: `x509_parse_certificate` returns a `Certificate` whose
`tbs_start`/`tbs_len` are `255`/`44` against a true `4`/`295` when called from
`stdlib/tls/handshake.sio`, blocking every real handshake — an untested
variant of D2's large-aggregate cross-module family, this time for aggregate
*return* values.

### Exact stdout — the same endpoint, this branch, instrumented

Probe built from this worktree, run inside the `epistemic-search` pod (whose
CA bundle carries `conclave-search-internal-ca`, 152 entries):

```
P2 trust_store err 0 count 152
D19 inline_pos 4 typedref_pos 4 tbs_start_pos 4 top_pos 0 cs 8 cl 291
D19 stored tbs_start 4 tbs_len 295
D19 cvp path_len 2 c0.tbs 4/295 c0.pka 1
D19 chain_result 0 intermediate_count 0 leaf.tbs_start 4 leaf.tbs_len 295
P4 tls_connect rc 0
```

and with the instrumentation removed and `stdlib/` pristine:

```
trust_store err 0 count 152
tls_connect rc 0
```

`255` → **4**, `44` → **295**, `-6` → **`CHAIN_OK`**. `10.96.250.20:443`
behaves the same. Offline, the same 385-byte leaf (byte-sum 30515) driven
through `decode_certificate_message` gives `4`/`295` too.

### Why the original reading could not have meant what it was taken to mean
`tbs_len` is computed as `(content_start + content_len) - tbs_start_pos`, and
for this certificate `content_start + content_len = 299`. `299 - 255 = 44`.
D18's two "independently corrupt" fields were **one** wrong scalar plus its own
subtraction — so the "2nd and 3rd members corrupt, later members fine" shape,
the whole basis for calling it a new non-monotonic aggregate-return defect,
never existed.

Both standing compiler hypotheses are eliminated by measurement, not by
argument: `tbs_start`/`tbs_len` are declared by exactly one struct in the
tree, so `field_idx_from_name_simple`'s global first-match scan has nothing to
collide with; and D11's tuple-destructure struct-type recovery
(`LOWER_TUPLET_TMP_ID`) is present and demonstrably works, confirmed by two
purpose-built repros — single-module and cross-module — that plant a decoy
struct declaring `pos` at a different index and still read the right value.

### The `arena full` (exit 181, empty stdout)
Three hypotheses, all settled by measurement:

- **Buffering hides the status lines** — refuted. A program that prints three
  lines then exhausts the arena, with stdout redirected to a *file*, still
  shows all three. `print` is unbuffered; empty stdout means nothing printed,
  so the abort is *earlier* than it looks.
- **A regression from D18's restructuring** — refuted. Rebuilt against
  pre-D18 `chain.sio` (`ece1420be`), the binary behaves identically and does
  **not** abort.
- **Downstream of the `tbs_start` corruption** — moot; there is none.

The cause, measured in one pod on one query with only the binary differing:

| Binary | Arena reserved | Peak RSS | Result |
|---|---:|---:|---|
| deployed image `v4` | **2.0 GiB** | — | exit 181, empty stdout |
| built from this worktree | **8.0 GiB** | **3.2 GiB** | full pipeline, exit 1, honest report |

The deployed binary predates `eea3a449f` (arena 2 GiB → 8 GiB). The workload
needs 3.2 GiB. **Rebuild the image and the abort is gone.**

### What still stops a useful answer (neither is a compiler defect)
- beagle-core's `POST /api/exocortex/v1/graphrag/query` returns **HTTP 401** —
  it wants a Bearer token, and `conclave-search` sends only
  `X-Beagle-Consumer`, exactly as `search/memory_context.sio`'s own comment
  predicted. TLS to that host verifies `CHAIN_OK`, so this is not transport.
- `1.1.1.1`'s DoH leaf is **ecdsa-with-SHA384 over P-256** — D15's known hole —
  so `tls_connect` returns `-8`, DNS resolution fails, and every domain-name
  search result is dropped. SearXNG discovery itself works (five candidates
  over a verified connection).

### Test evidence
No `self-hosted/` or `stdlib/` file was modified; the only source change is one
added test, `tests/run-pass/tls_decode_certificate_message_p256_leaf.sio`,
which pins the offline half of the refutation. Suite results are in the
dispatch.

### Filed
Not a GitHub issue. Recorded here and in the dispatch per this file's
convention.
