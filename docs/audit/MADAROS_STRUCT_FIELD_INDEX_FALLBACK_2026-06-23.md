<!-- docs:meta
topic_id: repo.docs.audit.madaros-struct-field-index-fallback-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-struct-field-index-fallback-2026-06-23
-->

# Madaros by-value-struct "SIGSEGV" — root cause is struct field-index fallback (2026-06-23)

*Branch:* investigation off `main` (madaros built from source, `artifacts/self-hosted/madaros-sret`)
*Status:* ROOT CAUSE FIXED + VERIFIED (see "Resolution" below).

## Resolution (applied)

The root was the **two-level nested store** `(*box).entries[i].fields[fc] = …` (and the
local-table form `table.entries[i].fields[fc] = …`) being **silently dropped** by the
lean_single by-value-aggregate miscompile, so struct field registration never landed →
`field_idx_from_name*` fell back to the `name[0] % 64` hash → out-of-bounds field access.

Fixed in `self-hosted/ir/lower.sio` by rewriting the three registrant sites to the proven
**single-level idiom** (read the entry into a local, set the field + count, assign the
entry back once) — the same dodge as the enum-variant fix `62f2b3a28`, and the same shape
already used safely in the summary→lowerer copy (376–392):
- `ir_fast_summary_register_struct_fields_owned` (~1231)
- `ensure_struct_literal_layout_ref` (~3746)
- `register_struct_fields_ref` (~3843)

Plus a gated observation dump (`SOUNIO_DUMP_LAYOUTS=1`) in `lower_field_access_expr_ref`.

**lean_single is untouched** (the deeper miscompile remains; dodged in madaros source per
advisor — editing the bootstrap seed would risk the `gen2==gen3` fixed point).

### Verified (fresh madaros built from this exact source, on a clean `main` base)
- Dump confirms `struct_layouts` now contains `P2` with **relative** indices `a:0, b:1`
  (was: absent → hash fallback 33,34).
- Repros: `let p = P2{7,35}; print p.a,p.b` → `7 35` (was `7 0`); `P3{11,22,33}` →
  `11 22 33` (was `11 0 0`); `direct p.b / getb(p) / suma(p)` → `35 / 35 / 42`.
- madaros self-builds with the fix (full compiler bundle compiles).

### NECESSARY but not SUFFICIENT for the headline tests
`hyperbolic_geodesic.sio` / `lyapunov_*.sio` require **two** same-class fixes:
1. **this** struct field-index registration fix (`lower.sio`), and
2. the independent **`machine_ir` block-mutation** copy-modify-writeback fix (Codex's
   `4e07642df`, in `self-hosted/native/machine_ir.sio`).

On a clean `main` base with *only* fix (1), the repros pass but `hyperbolic_geodesic`
still SIGSEGVs (139) — it also hits the `machine_ir` bug. With *both* fixes present, all
three run to exit 0. Codex independently converged on the same copy-modify-writeback root
for `machine_ir`, corroborating the diagnosis. Both fixes should land together.

### Honest scope — one root of several
This removes the **struct-field-index** root. Other madaros codegen holes remain (enum/
method/Box construction, larger SRET shapes); a curated 15-test struct-returning sample
stayed 0/15 because those tests hit additional holes. Triage continues on the path to
Madaros-official.

---

## Summary

The "by-value-large-struct-return SIGSEGV" that fails ~70 % of run-pass tests under
`madaros run` is **not** primarily a return-ABI or stack-overflow bug. The decisive
reproduction shows it is a **struct field-index resolution bug**: user structs are **not
present in the `struct_layouts` table that field-access lowering consults**, so every
field index falls back to the hash `name[0] % 64`. Those huge indices (e.g. `a`→33,
`b`→34) drive **out-of-bounds heap field reads/writes** on an object allocated for only
`field_count` slots. The access "works" when the out-of-bounds memory happens to survive,
and yields wrong values or a SIGSEGV when it has been reused — exactly the observed
flakiness (wrong values for small i64/f64 structs, crashes for larger ones).

## Decisive reproduction (printed values, not exit codes)

```
struct P2 { a: i64, b: i64 }
let p = P2 { a: 7, b: 35 }; print p.a, p.b        →  7 0     (b lost)
var p = P2 { a: 7, b: 35 }; print p.a, p.b        →  7 0     (not let-vs-var)
var p = ...{0,0}; p.a=7; p.b=35; print p.a,p.b    →  7 35    (field-assign OK)
let p = P2{7,35}; getb(p) /* returns p.b */       →  35      (pass-by-value OK)
let p = P2{7,35}; suma(p) /* returns p.a+p.b */   →  42      (pass-by-value OK)
let p: P2 = P2{7,35}; print p.b                   →  0       (annotation no help)
struct P3{a,b,c}; let p=P3{11,22,33}; print a,b,c →  11 0 0
```

Key invariants:
- The struct **storage is intact** — a byte-copy (pass-by-value to a fn) carries all
  fields, and the callee reads them correctly when it reads immediately after the copy.
- The **direct read of a non-first field of a literal-bound local is wrong** (0), and is
  **time-sensitive** (works in `var+assign` where the write is immediately before the
  read; fails in `let = literal` where a `print` call intervenes and clobbers the OOB
  heap).
- `-O` is **off** by default, so this is **base codegen**, not the optimizer.

## Root-cause chain

1. Field read lowers via `lower_field_access_expr_ref` → `field_idx_for_base_ref`
   (`lower.sio:3622`). With `SOUNIO_LOWER_AGG_TRACE=1`, `p.a`→idx **33**, `p.b`→idx **34**.
2. `33 = 'a'(97) % 64`, `34 = 'b'(98) % 64` — the **last-resort hash** at the end of
   `field_idx_from_name` / `field_idx_from_name_simple` (`lower.sio:3496`, `:3528`).
3. That hash is only reached when the struct/field is **not found in
   `(*self.struct_layouts)`**. So the layout table consulted by field access **does not
   contain `P2`**.
4. Codegen (`codegen_x86_linux.sio:6383` `IrFieldGet`, `:6394` `IrFieldSet`) computes the
   slot as `field_idx + header/8` on a handle-resolved object base; the object was
   allocated `aggregate_storage_bytes(field_count)` = `count*8` (`lower.sio` ~7105). Index
   34 on a 2-field object ⇒ **far out of bounds**.
5. `SOUNIO_LOWER_ORDERED_TRACE=1` shows `register_struct_fields_ref`'s trace (which prints
   `struct_fields_begin/done`) **never fires** for the test's `struct P2`. Registration
   therefore goes through a *different* spine (`ir_fast_summary_register_struct_fields_owned`,
   `lower.sio:1199`, used by the summary/seed pre-pass) — into a table that is **not the
   `self.struct_layouts`** field access reads, or is dropped. The two registration spines
   that assign **relative** indices (`ensure_struct_literal_layout_ref` and the
   declaration path, both starting at 0) are not the table that the read sees.

## Why it presents as "by-value-struct return SIGSEGV"

Returning a struct by value, enum/Box construction, and method calls all build or copy
multi-field aggregates and then read their fields — every such read goes through the same
broken index resolution. With OOB slot indices, larger aggregates or already-reused heap
produce a hard SIGSEGV (the previously-documented `mov 0x0(%rdx)` null-derefs in
`MADAROS_METHOD_CALL_SIGSEGV` / `MADAROS_SRET_ROOT_SYNTHESIS`), while small ones silently
mis-read. The earlier "SRET/by-value-return" framing was a *symptom* class; the shared
root is the field-index fallback.

## Proposed fix (to validate)

Make the `struct_layouts` table that `field_idx_from_name*` reads actually contain the
program's structs with **relative** (0-based) field indices, so the hash fallback is never
taken. Candidate approaches, smallest first:
1. Ensure the lowerer's `self.struct_layouts` is seeded from the same registration that the
   summary spine performs (route the declaration registration into `self.struct_layouts`,
   not only the summary table), mirroring the enum-variant registration fix (`62f2b3a28`,
   "extract Box ptr first" to avoid the by-value-table store being dropped).
2. As a guard, make the last-resort hash a **loud codegen error** instead of a silent OOB
   index, so any unregistered struct fails at compile time rather than miscompiling.

## Verification gate (per advisor)

A fix is judged by the **real tests**, not the minimal repro:
- Re-run the 30-file run-pass sample under `madaros run` (currently 9/30 pass) — expect a
  large jump.
- `hyperbolic_geodesic.sio` (SIGSEGV 139) and `lyapunov_closed_forms.sio` must run to
  correct results.
- Self-host CI legs + madaros `--check` self-test must stay green; `gen2==gen3` fixed point
  is unaffected (lean_single is untouched; only modular files feed madaros).

## AI disclosure

Forensic investigation by AI agent (Claude) under human direction; all claims backed by
re-runnable `madaros compile`/`run` commands and source line references above.
