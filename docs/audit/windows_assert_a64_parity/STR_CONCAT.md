<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.str-concat
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.str-concat
-->

# A64 PARITY — `str_concat` builtin

**Opened / closed.** 2026-05-21.
**Status.** RESOLVED — CODE CHANGE LANDED.
**Class.** Codegen on `self-hosted/compiler/lean_single.sio` (new
`emit_byte_copy_loop_a64` + `emit_str_concat_slots_a64` + dispatch).
**Branch.** `feat/windows-assert-exit`.
**From the scan.** 3rd of the 4 string/char builtin gaps (after `print_char`,
`str_eq`). `str_slice` is the last.

---

## §1 — The gap

`str_concat(a, b)` existed on x86 but was absent from `compile_primary_a64` —
unknown identifier on `aarch64-*`.

## §2 — The fix

- `emit_byte_copy_loop_a64` — copies x11 bytes from x9 (src) to x10 (dst),
  advancing both (`cbz`/`ldrb`/`strb`/`add`/`add`/`sub`/`b`).
- `emit_str_concat_slots_a64` — `len_a=str_len(a)`, `len_b=str_len(b)`,
  `res=heap_alloc(len_a+len_b+1)`, copy a then b (x10 persists across both so the
  second copy continues at `res+len_a`), `strb wzr,[x10]` NUL terminator, return res.
- `str_concat` dispatch mirroring x86 (compile a → slot, compile b → slot,
  EXPR_TY=3 string).

**Bug found and avoided during the port:** `emit_load_var_a64` uses **x9 as
address scratch**, so the copy-setup must load the src into x9 **last** —
otherwise a later `emit_load_var_a64` clobbers it. (First cut set x9 first → only
1 byte "copied" from a frame address; caught on M3, `str_len(concat)`=1.)

## §3 — Verification

- **Self-host fixed point.** PASS: stage1==stage2==stage3,
  `md5=1a8700c745f903a71612f6fbc63771c6`; binary rebuilt.
- **Real Apple M3** (`aarch64-macos`): `str_concat` of `("foo","bar")`,
  `("","xyz")`, `("abc","")`, `("Hello, ","world!")` print `foobar`/`xyz`/`abc`/
  `Hello, world!`, byte-matching x86. `str_len(str_concat("12345","67890"))`=10.
- **x86 non-regression.** `text_interpolate`, `stdlib_time_basic`,
  `epistemic_hessian_transcendentals` exit 0.

## §4 — a64 is *more* correct than x86 on nested calls

`str_concat(str_concat("a","b"), str_concat("c","d"))` → a64 `abcd` (correct);
**x86 prints `ccd`** because x86 uses fixed global `BUILTIN_SLOT_A/B` that inner
calls clobber. The a64 port allocates per-invocation `NEXT_SLOT` locals, so
nesting composes correctly. (Not a parity defect to "fix" — x86's global-slot
reuse is the latent bug; a64 sidesteps it. Pre-existing x86 issue, out of scope.)

## §5 — Remaining from the scan

`str_slice` is the last confirmed a64 builtin gap (mmap + a single bounded copy).
