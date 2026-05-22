<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.str-slice
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.str-slice
-->

# A64 PARITY — `str_slice` builtin (last of the scan's 4)

**Opened / closed.** 2026-05-21.
**Status.** RESOLVED — CODE CHANGE LANDED.
**Class.** Codegen on `self-hosted/compiler/lean_single.sio` (new
`emit_str_slice_range_slots_a64` + dispatch).
**Branch.** `feat/windows-assert-exit`.
**From the scan.** 4th and final string/char builtin gap (after `print_char`,
`str_eq`, `str_concat`).

---

## §1 — The gap

`str_slice(s, start, end)` (substring `[start, end)`; 2-arg form `(s, start)`
returns the suffix pointer `s+start`) existed on x86 but was absent from
`compile_primary_a64`.

## §2 — The fix

`emit_str_slice_range_slots_a64`: `len = end - start`, `res = heap_alloc(len+1)`,
copy `len` bytes from `s+start` via the shared `emit_byte_copy_loop_a64`,
NUL-terminate, return res. Dispatch mirrors x86's 3-arg parse with the 2-arg
fallback (`add x0, x1, x0` → `s+start`), EXPR_TY=3. Same x9-last discipline as
`str_concat` (the src pointer is computed into x0 then `mov x9,x0` last, since
`emit_load_var_a64` clobbers x9).

## §3 — Verification

- **Self-host fixed point.** PASS: stage1==stage2==stage3,
  `md5=98f267a4875c9c05d78e5b5822ecbaf5`; binary rebuilt.
- **Real Apple M3** (`aarch64-macos`): `str_slice("hello",1,3)`=`el`,
  `("hello",0,5)`=`hello`, `("hello",2,2)`=`` (empty), `("abcdef",3)`=`def`
  (2-arg suffix), `str_len(str_slice("hello world",0,5))`=5 — all byte-match x86.
- **x86 non-regression.** `text_interpolate`, `stdlib_time_basic`,
  `epistemic_hessian_transcendentals` exit 0.

## §4 — Scan complete

This closes the four string/char builtin gaps the `_x86`-only helper scan found:
`print_char` (`20ffa61c7`), `str_eq` (`771d5a7e0`), `str_concat` (`eff2c2b5e`),
`str_slice` (this commit). The remaining `_x86`-only helpers are arch internals
(rcx/rax ops, ABI shuffling, syscall6, macho writer), the GPU subsystem (x86-host
only), and Windows PE — none are a64 language-feature gaps. The ARM64 backend now
has no known builtin/language-feature parity gaps; the only documented residual
is the absence of a GTT (gradient-topology-type) refusal layer (low priority).
