<!-- docs:meta
topic_id: repo.docs.audit.madaros-print-int-dispatch-2026-06-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-print-int-dispatch-2026-06-20
-->

# Madaros native_v2: `println`/`print` of an integer literal — fix (2026-06-20)

Branch `fix/madaros-print-int-dispatch` off `origin/main` @ `659492156`.

## Bug
`println(42)` / `print(42)` SIGSEGV at runtime of the generated ELF (not at compile).
`println("…")` works. Reproduced and isolated by core-dump disassembly of the *generated*
program (`/tmp/pi.elf`):

```
main:            mov $0x2a,%rax ; ... ; mov 42,%rdi ; call print_str_helper
print_str_helper(0x401067):  mov %rdi,%rsi ; movzbq (%rdi),%rax  <-- CRASH (rdi=42 as char*)
                             ... strlen then write(fd, ptr, len)
```

`println(x)` is lowered (`self-hosted/ir/lower.sio`, println branch) to
`print(x); print_char('\n')`, and `print` resolves to the `print_str` builtin
(strlen+write). For an integer argument the raw value is handed to `print_str` as a
`char*`, so the strlen loop dereferences address `0x2a` → SIGSEGV. There is no
int→string dispatch. (Original code comment even said: *"println works for any argument
type that print accepts (primarily strings)."*)

## Why lowering, not codegen
The native_v2 backend already has a correct integer printer — `print_int`
(`emit_builtin_print_int_into`, div-by-10 itoa). Proven locally with the prebuilt binary:
`print_int(42)` → `42`, `print_int(var)` → works; `print(42)` → SIGSEGV. The backend
only tracks int-vs-float per reg (`is_float_reg`), **not** string-vs-int, so it has no
marker to dispatch on at codegen time. The dispatch must happen in lowering, where the
argument AST is available.

## Fix (this commit)
In `lower.sio`, route `println`/`print` of an **integer literal** (`ExprKind::ExprIntLit`)
to the `print_int` builtin instead of `print`/`print_str`. New free helper
`lower_call_first_arg_is_int_literal`. Defaults to `print` (string) for every other
argument form → **zero string regression**.

## Scope / known limitation (honest)
- `struct Expr` (parser/ast.sio) carries **no inferred type** on a general expression, so
  at lowering only an integer *literal* is unambiguously typed. `println(n)` where `n` is
  an int **variable** or an arithmetic result is **not** covered and still uses `print_str`
  (i.e. still crashes). Fixing the general case needs checker type info threaded into
  lowering (or a string-vs-int reg/local marker added to lowering+codegen).
- `println(<float>)` is a separate bug: `print_f64` is not wired as a native_v2 builtin
  (a direct `print_f64(3.14)` build fails). Not addressed here.
- Likely a user-facing correctness/completeness win, **not** a gen2==gen3 fixed-point
  unblocker (the compiler formats its own numbers via buffers, not `println(int)`).

## Validation status
- `--check` of `self-hosted/compiler/main.sio` with the prebuilt madaros: **755 errors
  both with and without this change** (pre-existing prebuilt-vs-full-bundle noise) →
  the edit is **type-check-neutral** (adds no new errors).
- End-to-end behaviour (int literal actually prints) is **unverified pending a madaros
  rebuild** from this source — the prebuilt binary cannot exercise the source change.
  Verify after rebuild (CI `madaros-prebuilt-refresh.yml` ref=`fix/madaros-print-int-dispatch`):
  `println(42)` prints `42\n` rc 0; `println("hi")` still prints `hi`; full census unchanged.
