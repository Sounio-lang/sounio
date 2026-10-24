<!-- docs:meta
topic_id: repo.docs.audit.lean-single-string-literal-codegen-sqw-overflow-2026-09-03
authority: repo_only
audience: users
last_validated: 2026-09-03
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-string-literal-codegen-sqw-overflow-2026-09-03
-->

# lean_single.sio string-literal codegen: rel8 overflow segfault, and three sibling landmines it uncovered

**Status:** FIXED. Root cause, evidence, and fix are below. Filed retroactively:
the segfault was root-caused and fixed first (PR #2396), then this dispatch was
written up alongside a code review of that PR that surfaced three more defects
in the same code path -- also fixed here, per the same review. `self-hosted/`
is meant to be patched from a dispatch recorded *first*; this one is after the
fact because the review that found the sibling bugs happened after the initial
fix already landed. Recording it now rather than not at all.

**Scope:** `self-hosted/compiler/lean_single.sio`'s string-literal decode/codegen
paths: `compile_primary()` / `compile_primary_a64()` (string literal as a
function-call argument), `emit_print()` / `emit_print_a64()` (string literal in
`print("...")`/`println("...")`), and `compile_all_arm64()`'s pass-2 gate.

Not the same subsystem as
[`LONG_STRING_LITERAL_DIAGNOSTIC_CENSUS_2026-08-17.md`](LONG_STRING_LITERAL_DIAGNOSTIC_CENSUS_2026-08-17.md),
which covers the Madaros/native-v2 IR pipeline's separate `Name`/arena string
handling (`ir_normalize_string_literal_name`, `ir_arena_put_name`) -- a
different code path with a coincidentally similar symptom (long literals
silently truncated), not a shared root cause.

## The fact (bug 1: rel8 jump overflow, the original segfault)

`compile_primary()`'s `k == 5` (string literal) branch embeds a literal's bytes
inline in generated x86-64 code with a "jump over the bytes, then `lea` back at
them" trick:

    em(0xeb)          // jmp rel8
    em(slen + 1)      // skip distance, as a SIGNED BYTE (-128..127)
    ...string bytes...
    em(0)             // null terminator
    em(0x48); em(0x8d); em(0x05); em32(disp)   // lea rax, [rip + disp32]

`em(v: i64)` stores `v as i8`. Once a literal's decoded content reached 127
bytes, `slen + 1` (128) no longer fit a signed byte and wrapped to `-128..-1`.
The `jmp` then jumped *backward* into whatever code was emitted just before it,
instead of forward past the string data -- corrupting execution at the call
site, before the callee ever ran.

Any `tc_error`/`tc_error_hard` diagnostic message (or any other string-literal
call argument) past ~125 characters reliably segfaulted the compiled binary.
Discovered while implementing #2058, worked around there by keeping the new
diagnostic short.

## Measured

Repro (a compiler built from unmodified `lean_single.sio`, given a source file
whose call site passes a 143-byte string literal):

    $ /tmp/souc-host-broken.elf tests/compiler/imported_string_literal_overflow/main.sio /tmp/out.elf --target x86_64-linux
    ...
    $ /tmp/out.elf
    before call
    before call
    before call
    ...(loops, then)
    Segmentation fault (core dumped)   # exit 139

Disassembly confirmed the mechanism: the jump instruction preceding the
embedded string was `eb <negative displacement>`, landing back inside the
`print("before call\n")` call that ran just before it -- an infinite
re-execution of unrelated code, not the callee.

## Fix (bug 1)

Switch the skip-jump from `jmp rel8` (`em(0xeb); em(slen+1)`, 2 bytes) to
`jmp rel32` (`em(0xe9); em32(slen+1)`, 5 bytes) in `compile_primary()`. The
skip *distance* (`slen + 1`) is unchanged -- only the instruction encoding
that carries it changes, so no other offset math in the function needed to
move. `compile_primary_a64()` (arm64) was never affected: its `b`
(unconditional branch) instruction already carries a 26-bit word-offset
immediate, far more range than any literal `SQW` can hold.

Also added a bounds check in the same decode loop: the shared scratch buffer
`var SQW: [i64; 256]` (256 slots) was written byte-at-a-time with no check
against its own declared size, so a literal over 256 bytes -- independent of
the rel8/rel32 question -- would corrupt `FN_NS: [i64; 65536]`, the next
global declared after `SQW`. Applied to both `compile_primary()` and
`compile_primary_a64()`.

## The fact (bug 2: the new bounds check was itself silently bypassable)

The bounds check calls `tc_error_hard()`, which calls `tc_mark_failed()`:

    // Guard: errors in imported module functions should NOT block ELF emission.
    fn tc_mark_failed() with Mut {
        TC_FN_ERR_COUNT = TC_FN_ERR_COUNT + 1
        if MAIN_SRC_END > 0 && CURRENT_FN >= 0 && CURRENT_FN < 65536 {
            if (FN_EFFECTS[CURRENT_FN as usize] & 2048) != 0 {
                return
            }
        }
        TYPECHECK_FAILED = 1
    }

Bit 2048 marks a function pulled in via `use` (an imported module). This
early-return exists so a cosmetic type error in unused imported code doesn't
block the whole build -- deliberate, and correct for that case. But it also
covers the new string-literal-length check, whose "error" is not cosmetic:
the literal has *already been silently truncated to empty* by that point
(`slen = 0` is set unconditionally so codegen has something well-formed to
emit). For a >=256-byte literal inside an imported-module function with <=10
other errors in that function, the result was: an `error:` line printed, but
`TYPECHECK_FAILED` never set, so `compile_all()`'s failure gate never tripped,
and a working binary shipped with the argument silently replaced by `""`.

## Measured (bug 2)

Two-file fixture (`lib.sio` defines `pub fn use_helper()` that calls another
function with a 260-byte literal; `main.sio` imports and calls it):

    $ /tmp/souc-host-baseline.elf tests/compiler/imported_string_literal_overflow/main.sio /tmp/out.elf --target x86_64-linux
    error: string literal exceeds maximum of 256 bytes at .../lib.sio:7 (bundle line 7)
    elf: /tmp/out.elf 36715 bytes (bss=1048984)
    $ echo $?
    0
    $ /tmp/out.elf
    main reached
    $ echo $?
    0

A binary was written and ran successfully despite the printed error.

## Fix (bug 2)

Set `TYPECHECK_FAILED = 1` directly in the bounds-check branch, bypassing
`tc_mark_failed()`'s import-tolerance for this one error class -- the same
idiom already used by `tc_unbalanced_braces()` (#1634) for the same reason
(that function's own comment: unbalanced braces in an imported module used to
fail the build silently, with no `error:` line at all, via the same early
return). Applied to both `compile_primary()` and `compile_primary_a64()`.

Regression coverage: `scripts/ci/imported_string_literal_overflow_gate.sh`
against the `tests/compiler/imported_string_literal_overflow/` fixture --
asserts nonzero exit, the diagnostic, `typecheck: failed`, and no ELF written.

## The fact (bug 3: two sibling decoders had no bounds check at all)

`SQW` is a shared global, not local to `compile_primary`/`compile_primary_a64`.
`emit_print()` and `emit_print_a64()` -- the codegen for the `print("literal")`
/`println("literal")` fast path, reached via a *different* dispatch than a
call argument -- decode the same escape syntax into the same buffer with no
bounds check whatsoever, in either the pre-fix or the bug-1-fixed version:

- `emit_print_a64()`: `SQW[slen as usize] = byte`, one byte per slot, same
  256-byte threshold as the just-fixed decoders, still unguarded.
- `emit_print()`: packs 8 decoded bytes per `i64` slot (`qi = slen / 8`), so
  its overflow threshold is ~2048 bytes rather than 256, but the write
  (`SQW[qi as usize] = SQW[qi as usize] | (byte << (bi * 8))`) is equally
  unguarded.

Both overflow into `FN_NS` exactly like bug 1's pre-fix buffer, just reached
through `print("...")` instead of a call argument.

## Fix (bug 3)

Added the same bounds-check idiom (diagnostic + direct `TYPECHECK_FAILED = 1`
+ truncate-and-stop-decoding) to both functions, at their respective
thresholds (256 for `emit_print_a64`, 2048 for `emit_print`). Both functions'
effect signatures gained `IO` (needed to call `tc_error_hard`/`print`); their
only callers (`compile_primary`/`compile_primary_a64`) already declare `IO`,
so this is a strict-superset change with no cascading signature edits.

Regression coverage: `tests/compile-fail/print_literal_exceeds_sqw_capacity.sio`
(the x86-64 `emit_print` threshold; runs in the standard suite). The arm64
`emit_print_a64` threshold was verified manually (below) rather than wired
into CI: this repo's existing arm64 correctness tests
(`scripts/ci/arm64_nested_store_witness_gate.sh`) are already attest-only off
Apple Silicon, and building new arm64 execution CI infrastructure was out of
scope for this fix.

## The fact (bug 4: the arm64 pass-2 gate didn't check TYPECHECK_FAILED either)

Fixing bug 3 for `emit_print_a64` exposed a fourth, independent gap.
`compile_all_arm64()` runs Pass 1 (`compile_all()`, the x86-64 pass, whose
code output is discarded) first and returns early if it fails -- but Pass 1's
`emit_print()` has a 2048-byte threshold, so a 260-byte literal sails through
Pass 1 untouched. Pass 2 (the real arm64 codegen) then calls `emit_print_a64()`,
which correctly detects the 260-byte overflow and sets `TYPECHECK_FAILED = 1`
-- but `compile_all_arm64()`'s Pass 2 loop has no check of that flag before
falling through to patch/emit and returning `0` unconditionally:

    $ /tmp/souc-host-r2.elf print_overflow_arm64.sio /tmp/out.elf --target aarch64-linux
    arm64: re-emitting pass 2
    error: string literal exceeds maximum of 256 bytes at <main>:2 (bundle line 2)
    arm64_compile: fns=22 code=35068 main=fn0 patches=23
    elf_arm64: /tmp/out.elf 39164 bytes (bss=1048984)
    $ echo $?
    0

(`souc-host-r2.elf` here already has bug 3's fix but not bug 4's.)

## Fix (bug 4)

Added `if TYPECHECK_FAILED != 0 { print_limit_error(); print("typecheck: failed\n"); return 1 }`
in `compile_all_arm64()` immediately after Pass 2's per-function codegen loop
and before BL-instruction patching begins -- mirroring `compile_all()`'s own
gate for Pass 1. Re-ran the same repro after this fix: exit 1, `typecheck:
failed`, no `elf_arm64:` line, no file written.

## Files touched

- `self-hosted/compiler/lean_single.sio` -- all four fixes above.
- `tests/run-pass/long_string_literal_call_arg.sio` -- bug 1 regression (from
  the original PR #2396).
- `tests/compile-fail/print_literal_exceeds_sqw_capacity.sio` -- bug 3 (x86-64
  `emit_print`) regression.
- `tests/compiler/imported_string_literal_overflow/{lib.sio,main.sio}` +
  `scripts/ci/imported_string_literal_overflow_gate.sh` -- bug 2 regression.
- Bug 4 (`compile_all_arm64`'s missing Pass-2 gate) and bug 3's
  `emit_print_a64` threshold: verified manually per the transcripts above;
  not wired into automated CI (see bug 3's fix note).
