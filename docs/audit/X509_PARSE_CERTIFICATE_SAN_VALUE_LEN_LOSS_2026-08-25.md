<!-- docs:meta
topic_id: repo.docs.audit.x509-parse-certificate-san-value-len-loss-2026-08-25
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.x509-parse-certificate-san-value-len-loss-2026-08-25
-->

# Madaros forensic dispatch — `x509_parse_certificate` SAN `value_len` loss

**Date:** 2026-08-25
**Branch:** `tls-on-madaros`
**Engine:** default Madaros (`bin/souc`, v0.80.0), imported-module full-IR path
**Status:** ROOT-CAUSED and FIXED (one line, `self-hosted/ir/lower.sio:2555`).
Two independent defects were found behind one symptom; Defect B is fixed and
unblocks `tests/run-pass/x509_chain_verify_positive.sio`, Defect A remains open
with an exact, already-in-use source workaround (§3, §7).
**Related prior dispatches:**
`docs/audit/X509_ARRAY_STRUCT_FIELD_CORRUPTION_DISPATCH_2026-08-24.md` (Finding 24),
`docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`,
`.superpowers/sdd/2026-08-24-madaros-x509-chain-validation-plan/task-4-report.md`
(“Second, separate compiler/stdlib defect”, item 3 — UNRESOLVED).

---

## 1. Reported symptom (as inherited)

`stdlib/x509/cert.sio`’s

```sounio
pub fn x509_parse_certificate(buf: &RawBuf, len: i64) -> (Certificate, i64) with IO
```

writes `cert.san_entries[0].value_len` correctly — verified by an in-function
`print_int` immediately before its own `(cert, X509_OK)` return — but *every*
caller read `san_entries[0].value_len` back as `0`.

The prior investigation reported one confirmed correlate and no explanation:
the loss reproduced for every REAL certificate tried (513–838 bytes) and did
NOT reproduce for the 197-byte hand-built fixture in
`tests/run-pass/x509_parse_full_certificate.sio`. That report hypothesised a
size-dependent return-value marshalling cap (“in the spirit of `MF_AREBOX_CAP`”).

**That correlate is a red herring.** Certificate size is causally irrelevant.
There are two independent, purely *static* defects, both in the READ path, both
size-independent. Neither loses any data: the returned `Certificate` value is
byte-correct in memory in every case measured below.

---

## 2. Decisive measurement — the data is not lost

Harness (`/tmp/rep/h1.sio`): the real 835-byte leaf DER from
`tests/run-pass/x509_chain_verify_positive.sio`, parsed, then read back in the
caller. A temporary accessor was added inside `stdlib/x509/cert.sio`:

```sounio
pub fn dbg_gn_value_len(c: &Certificate, i: i64) -> i64 {
    c.san_entries[i as usize].value_len as i64
}
```

Measured output:

| Read site | Expression | Result | Correct? |
|---|---|---|---|
| caller (`main`) | `leaf_cert.san_count` | `1` | yes |
| caller (`main`) | `leaf_cert.san_entries[0].tag` | `2` | yes |
| caller (`main`) | `leaf_cert.san_entries[0].value_len` | `0` | **NO** (expect 21) |
| caller (`main`) | `leaf_cert.san_entries[0].value[0]` | `0` | **NO** (expect 99 = `'c'`) |
| `cert.sio` accessor | `dbg_gn_value_len(&leaf_cert, 0)` | `21` | yes |

Same object, same instant: the accessor inside `cert.sio` sees `21`, the direct
read in the caller sees `0`. Therefore no write is lost, no return value is
truncated, and no capacity cap is involved. **The caller reads the wrong byte
offset.**

Note which fields survive: `san_count` (a `Certificate` field with a globally
unique name) and `tag` (`GeneralName` field index 0, offset 0 — correct under
*any* misresolution). The fields that fail — `value_len`, `value` — are exactly
the ones whose *names are shared* with other structs at *different* field
indices:

| Field name | `GeneralName` idx | `RdnEntry` idx | `ExtensionEntry` idx |
|---|---:|---:|---:|
| `value` | 1 | 3 | 3 |
| `value_len` | 2 | 4 | 4 |

This is the exact signature of Finding 24’s name-only global field lookup
(`Lowerer::field_idx_from_name_simple`) being reached because the *typed*
resolution path returned “unknown”.

---

## 3. Defect A — struct type lost through tuple destructuring (Finding 25)

`let (leaf_cert, e3) = x509_parse_certificate(...)` is desugared by the parser
(`self-hosted/parser/stmts.sio`, lines 40–41, 100, 314–315) into

```
let __tupN = x509_parse_certificate(...)
let leaf_cert = __tupN.0
let e3        = __tupN.1
```

The let-binding struct-type propagation in `self-hosted/ir/lower.sio`
(≈ lines 13401–13525) handles RHS kinds `ExprStructLit`, `ExprCall`,
`ExprFieldAccess` (Box only), `ExprMethodCall`, `ExprIdent` and `ExprIndex` —
but the tuple-element read is an `ExprFieldAccess` whose base is the tuple temp,
and `IrFunction` carries only a single `return_struct_name` (`self-hosted/ir/ir.sio:1847`),
which `lower_opt_type_named_name` leaves EMPTY for a tuple return type. So
`leaf_cert` is bound with **no** struct type, and every later
`leaf_cert.san_entries[i].field` falls through
`expr_struct_type_ref` → `array_index_base_elem_struct_type` →
`field_idx_for_array_index_base` (all return −1/empty) into the name-only
global lookup.

**Evidence.** Applying the existing Finding-25 workaround in the caller —

```sounio
var leaf_cert = certificate_zero()
let (leaf_raw, e3) = x509_parse_certificate(&leaf_buf, leaf_der_len)
leaf_cert = leaf_raw
```

— makes the same reads return `value_len = 21`, `value[0] = 99`. Nothing else
changed.

**Why the 197-byte fixture “passed”.** `tests/run-pass/x509_parse_full_certificate.sio`
(lines 95–103) already applies that workaround, with an explicit “Per Finding
25’s workaround” comment. `tests/run-pass/x509_chain_verify_positive.sio`
(line 264) and every ad-hoc bisection harness used in the prior investigation
used the bare tuple destructure. The correlate was *test-file binding style*,
not certificate size. Verified directly: the 835-byte real cert reads back
correctly under the workaround, and reads back `0` without it.

**Disposition:** NOT fixed at the compiler level in this dispatch. A proper fix
needs per-tuple-slot return type information on `IrFunction`, which does not
exist today (`return_struct_name` is one `Name`; `returns_float` carries a
1024+bitmask for tuple slots, but there is no name equivalent). Adding it
touches `ir.sio`, `ssa.sio`, `optimize.sio`, `serialize.sio`, `module_frontend.sio`
and six sites in `lower.sio` — a large, SIGSEGV-prone change to a self-hosting
compiler, which this repository’s own discipline says not to attempt
speculatively. The documented `var x = T_zero(); let (raw, st) = f(); x = raw`
workaround is exact and already used across the x509 tests.

---

## 4. Defect B — array-element type name dropped when preseeding EXTERNAL structs

Applying the Defect-A workaround to `tests/run-pass/x509_chain_verify_positive.sio`
did **not** make it pass: it still returned `-7`
(`CHAIN_ERR_HOSTNAME_MISMATCH`). A second, independent instance lives in
`stdlib/x509/chain.sio`:

```sounio
pub fn x509_verify_hostname(leaf: &Certificate, hostname: &RawBuf, hostname_len: i64) -> bool with IO {
    var i: i32 = 0
    while i < leaf.san_count {
        let san = leaf.san_entries[i as usize]
```

Here `leaf` is a properly typed `&Certificate` parameter — no tuple
destructuring anywhere. Instrumented measurement inside `x509_verify_hostname`,
called on a leaf whose `Certificate` value was already proven correct:

```
san.tag                              = 2    (correct)
san.value_len                        = 0    (WRONG, expect 21)
leaf.san_entries[i as usize].value_len = 0   (WRONG, expect 21)
```

while the byte-identical expression `c.san_entries[i as usize].value_len` inside
`stdlib/x509/cert.sio` returned `21`. The difference is the **module**:
`GeneralName`/`Certificate` are *declared* in `cert.sio` and merely *imported*
into `chain.sio`.

### Root cause

`self-hosted/ir/lower.sio`, `lowerer_preseed_external_items_into_acc_mut`
(the into-acc preseed that registers a dependency module’s struct layouts —
this is the path the imported-module full-IR compile takes; the build log
prints `lower_array: dep_mode into_acc`). Its `StructFieldEntry` construction,
at line 2555, read:

```sounio
elem_type_name_id: ir_intern_name(ir_empty_name()),
```

Every sibling registration site that has the declared `FieldDef` type in hand
uses the real element type name instead:

- line 2369 — `elem_type_name_id: ir_intern_name(lower_type_expr_array_elem_name(&(*flist).head.ty))`
- line 2647 — same (`lowerer_register_struct_fields_early_mut`)
- line 2704 — same
- line 3521 — same
- line 11452 — same

(Line 11314 is the struct-*literal*-derived registration, which genuinely has no
declared type; its empty value is correct and is documented as such in place.)

The preseed at 2555 **does** populate `named_type_name_id` from the same
`(*flist).head.ty` on the very next line, so the omission is an oversight in
that one field, not a deliberate limitation.

Consequence: for any struct registered through the external preseed,
`Lowerer::field_array_elem_type_name_for_struct` (line 10790) returns the empty
name for every array-of-struct field. `array_index_base_elem_struct_type`
(10858) therefore returns empty, `field_idx_for_array_index_base` (10679)
returns −1, and `arr[i].field` in *any consuming module* falls back to the
name-only global lookup — Finding 24, re-entered through a different door.

This is not x509-specific. It affects every `owner.arrayfield[i].name` access
where `owner`’s struct is declared in another module and `name` is not globally
unique across all registered structs.

### Fix

One line, `self-hosted/ir/lower.sio:2555`:

```sounio
-  elem_type_name_id: ir_intern_name(ir_empty_name()),
+  elem_type_name_id: ir_intern_name(lower_type_expr_array_elem_name(&(*flist).head.ty)),
```

bringing this registration site into line with all five siblings that have the
declared field type available.

---

## 5. Minimal reproducer

Committed as a three-file regression test:

- `tests/run-pass/imported_array_struct_field_decl.sio` (`//@ ignore`) — declares
  `Other` (fields `value` idx 3, `value_len` idx 4) then `Leaf` (`value` idx 1,
  `value_len` idx 2) and `Owner { leaves: [Leaf; 4], count: i32 }`; writes
  `o.leaves[0].value_len = 21` **in the declaring module**.
- `tests/run-pass/imported_array_struct_field_mid.sio` (`//@ ignore`) — a module
  that only `use`s the above and reads `o.leaves[0].value_len` from a
  `&Owner` parameter.
- `tests/run-pass/imported_array_struct_field.sio` (`//@ run-pass`) — drives both
  and asserts the declaring-module read and the importing-module read agree.

Measured, same source, same command (`./bin/souc run`):

| Madaros build | `mid_read_tag` | `owner_read_here` | `mid_read_value_len` |
|---|---:|---:|---:|
| before fix (`a13da0b63`) | 7 | 21 | **64** |
| after fix | 7 | 21 | **21** |

`64` is a byte out of `Leaf.value` — i.e. the consumer read `Other`’s field index
4 instead of `Leaf`’s index 2.

The three essential ingredients, each verified necessary by falsification during
bisection:

1. Two structs sharing a field NAME at different field INDICES, with the *wrong*
   one declared first (declaring `Leaf` first makes the buggy fallback land on
   the correct index by luck, and the repro goes green on the broken compiler).
2. The owner struct declared in a module OTHER than the one doing the
   `arr[i].field` access, and that access in a module other than the seed/main
   file (a `use decl::*` directly from `main.sio` did **not** reproduce).
3. Asymmetry between the write site and the read site. When both the write and
   the read use the same wrong offset the value round-trips and nothing is
   observable — this is why the defect only ever surfaced as “the parser wrote
   it, the consumer can’t see it”.

Certificate size, DER byte count, loop-iteration count, extension count and RSA
key size are all irrelevant — every one of them was independently ruled out, and
none appears in the reproducer.

---

## 6. Verification

All commands from the repo root with `SOUNIO_STDLIB_PATH=$PWD/stdlib`.

1. **Madaros rebuilds itself.** `bash scripts/ci/build_modular_madaros.sh
   artifacts/self-hosted/madaros` → `compile: fns=10989`, `Madaros ready
   (100659745 bytes)`. Self-compilation is the single largest cross-module
   struct consumer in the tree, so this is the strongest smoke test available
   for a struct-layout change.
2. **Minimal repro.** `./bin/souc run tests/run-pass/imported_array_struct_field.sio`
   → `imported array-element field index OK` (before: read `64`, not `21`).
3. **The blocking test now passes.**
   `./bin/souc run tests/run-pass/x509_chain_verify_positive.sio` →
   `x509_chain_verify_positive: full chain verified` (before: exit 1, printed
   `-7` = `CHAIN_ERR_HOSTNAME_MISMATCH`). Note that this test still uses the
   *bare* tuple destructure — Defect A does not bite it, because the leaf is
   only ever passed by reference into typed parameters.
4. **x509 suite, no regressions.**
   `SOUNIO_TEST_JOBS=2 bash scripts/run_sio_test_suite.sh --filter-prefix x509_`
   → Pass 16 / Fail 0 / Skip 3. (Before the fix the same run was Pass 15 /
   Fail 1, the failure being `x509_chain_verify_positive`.)
   At `SOUNIO_TEST_JOBS=8` this same set reports 14 spurious
   `run timed out after 30s` failures on this pod — contention, not the fix;
   each of those tests completes in ~5 s when run alone. Use `--jobs 2`.
5. **`asn1` suite:** Pass 5 / Fail 0.
6. **`struct` suite:** Pass 5 / Fail 1 — `struct_missing_field.sio`
   (`missing error: missing field`). Verified PRE-EXISTING by rebuilding Madaros
   from the unpatched tree and re-running: identical result.
7. **`epistemic` suite:** Pass 20 / Fail 13 / Skip 3, with a byte-identical
   failure list before and after the fix (verified by a full
   stash → rebuild → run → unstash → rebuild cycle). No regression.
8. **`import` suite:** Pass 6 / Fail 4 — all four are `error[E175] function is
   private in its defining module`, the known stdlib-visibility issue. E175 is
   emitted by `check`, which runs strictly before lowering, so a change to
   `lower.sio` cannot cause it. Pre-existing.

---

## 7. Residual risk and follow-up

- **Defect A (§3) is still open.** `let (x, st) = f()` still loses `x`’s struct
  type, so any *direct* deep field read on a tuple-destructured struct local in a
  consuming module remains wrong. Measured after this fix: the harness reading
  `leaf_cert.san_entries[0].value_len` directly off a bare tuple destructure
  still returns `0`. Workaround (already the documented house style, used by
  `tests/run-pass/x509_parse_full_certificate.sio`):

  ```sounio
  var cert = certificate_zero()
  let (cert_raw, status) = x509_parse_certificate(&buf, len)
  cert = cert_raw
  ```

  A real fix needs per-tuple-slot return type information on `IrFunction`.
  Suggested shape, for whoever picks this up: mirror how `returns_float` already
  carries a `1024 + mask` per-tuple-slot encoding, but for names this needs an
  actual `[Name; 8]` (or a parallel layout-index array) plus propagation at the
  six `return_struct_name` assignment sites in `lower.sio`, the two summary-copy
  sites (`lower.sio:1395`, `lower.sio:2062`), `module_frontend.sio:2458`, and the
  four `IrFunction` literal constructors (`ir.sio:5346`, `ssa.sio:1072`,
  `optimize.sio:619`, `serialize.sio:693`). That is a large change to a
  self-hosting compiler and should be its own dispatch with its own bisection
  budget.
- **The fix widens what resolves correctly; it never narrows it.** Before, an
  imported struct’s array-of-struct field had NO element type and every consumer
  fell to the by-name fallback. After, consumers use the declared element type
  when one exists and fall back exactly as before when it does not (the empty
  name is still produced for non-array and unnamed-element field types by
  `lower_type_expr_array_elem_name`). No call site changes behaviour unless it
  was previously guessing.
- **`self-hosted/ir/lower.sio:11314`** keeps its empty `elem_type_name_id`
  deliberately: that path registers a layout derived from a struct *literal*’s
  value list, which carries no declared `FieldDef` type. Its in-place comment
  already says so. Not touched.
