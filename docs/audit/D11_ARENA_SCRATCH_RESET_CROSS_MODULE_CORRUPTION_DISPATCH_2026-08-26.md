<!-- docs:meta
topic_id: repo.docs.audit.d11-arena-scratch-reset-cross-module-corruption-dispatch-2026-08-26
authority: repo_only
audience: users
last_validated: 2026-08-26
validated_by: controller (tls-on-madaros branch, TLS 1.3 handshake sub-project)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.d11-arena-scratch-reset-cross-module-corruption-dispatch-2026-08-26
-->

# Forensic dispatch — D11: a tuple-destructured local loses its struct type, so a field read resolves by NAME across every struct in the linked program (filed as "arena/scratch reset cross-module corruption"; the arena is not involved)

**Filed:** 2026-08-26 · **Status:** RESOLVED (fixed in `self-hosted/ir/lower.sio`, commit `3ec2d971d`; dispatch filed as `f83b20ce3`) · **Protocol:** CLAUDE.md §8 · **Supersedes the suspected area recorded in** `docs/handoff/souc_v0800_defects.md` §D11.

**The name of this file is kept as originally dispatched. It is wrong about the
mechanism and is retained only so the handoff entry's cross-reference resolves.
The arena/scratch reset machinery in `self-hosted/compiler/module_frontend.sio`
is NOT involved — see "What the arena diagnostic actually means" below.**

---

## Root cause

`let (r, status) = f()` is desugared **by the parser**
(`self-hosted/parser/stmts.sio`, the "Tuple-let desugaring side table" block)
into

```
let __tup0 = f()
let r      = __tup0.0
let status = __tup0.1
```

`self-hosted/ir/lower.sio`'s `lower_let_stmt_ref` has binding rules that record
a local's struct type for a struct-literal initialiser, a call initialiser
(via `expr_call_return_struct_name_ref`), a method-call initialiser, an
identifier initialiser, and an `arr[i]` initialiser — but **no rule for a
tuple-index field-access initialiser** (`__tup0.0`). `r` is therefore left with
an empty struct-type slot.

When `r.pos` is later lowered, `field_idx_for_base_ref` takes its
`ExprIdent` branch, `lookup_local_struct_type("r")` returns the empty name, and
it falls through to `field_idx_from_name_simple("pos")` — a **global,
name-only, first-registered-match scan over every struct layout in the linked
program**. Whichever struct happens to have been registered first and declares
a field literally named `pos` wins, at *its* field index.

This is exactly the fallback that `docs/audit/X509_ARRAY_STRUCT_FIELD_CORRUPTION_DISPATCH_2026-08-24.md`
fixed for `arr[i].field` bases, and exactly the gap that dispatch's own
resolution section left open and named:

> "one further gap deliberately left open (**Finding 25: tuple-destructured
> locals don't propagate struct types**)"

**D11 is Finding 25 realized on a security-critical path.** It is not a new
defect class; it is the known-open half of an already-fixed one.

### Why it looked like a scale / arena problem

In the real TLS program, the colliding pair is:

| Struct | Module | index of field `pos` |
|---|---|---|
| `HsBuf` | `stdlib/tls/client.sio` | **3** |
| `DerReader` | `stdlib/asn1/der.sio` | **1** |

`stdlib/x509/cert.sio`'s `x509_parse_certificate` does

```sio
let (cert_inner, e0) = der_enter(&top, &cert_seq_tag)
...
let tbs_start_pos = cert_inner.pos
```

`HsBuf` only enters the program when `tls::client` is imported. That is the
*only* thing that changes between the passing 20-module test and the failing
36-module program — not the module count, not the function count, not the
arena. `cert_inner.pos` then loads field index 3 of a 24-byte `DerReader`,
i.e. 8 bytes past its end, which on this stack layout is deterministically
`255`. `tbs_len` is computed from it (`(content_start + content_len) -
tbs_start_pos` = `527 - 255` = `272`), which is why the two wrong values are
consistent with each other and stable run to run. Every other field of the
returned `Certificate` is correct because no other read on the path goes
through a type-less base with a colliding field name.

## Minimal repro — 30 lines, one file, no imports, 6 merged functions

`tests/known_failures/madaros_tuple_destructure_field_name_collision_probe.sio`:

```sio
//@ run-pass
struct Decoy { d0: i64, d1: i64, d2: i64, pos: i64 }   // `pos` at index 3
struct Reader { base: i64, pos: i64, end: i64 }        // `pos` at index 1

fn decoy_touch() -> i64 {
    let d = Decoy { d0: 10, d1: 20, d2: 30, pos: 40 }
    d.pos
}
fn make_reader() -> (Reader, i64) { (Reader { base: 111, pos: 222, end: 333 }, 0) }
fn reader_pos(r: &Reader) -> i64 { r.pos }

fn main() -> i64 with IO {
    let direct = Reader { base: 111, pos: 222, end: 333 }
    let (r, status) = make_reader()
    print_int(direct.pos)        // 222  correct (struct-literal binding records the type)
    print_int(reader_pos(&r))    // 222  correct (typed &Reader parameter)
    print_int(r.pos)             // 0    WRONG  (expected 222)
    print_int(r.end)             // 333  correct (`end` collides with nothing)
    ...
}
```

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc run tests/known_failures/madaros_tuple_destructure_field_name_collision_probe.sio
```

Measured on Madaros v0.80.0, 2026-08-26:

```
touched=40 direct.pos=222 helper_pos=222 r.pos=0 r.end=333 status=0
```

`r.pos` reads `0` — `Decoy`'s index 3, eight bytes past the end of the 24-byte
`Reader`. Exit code 1.

**Order dependence (the proof that it is first-registered-match, not size):**
swap the two `struct` declarations so `Reader` is declared first and the same
program prints `r.pos=222` and exits 0. Nothing else changes.

### Reduction trail from the original 36-module report

Every step below was measured on this branch on 2026-08-26 with the committed
`artifacts/self-hosted/madaros`:

| Program | modules | merged fns | `arena_reset_totals` | `tbs_start/tbs_len` |
|---|---:|---:|---|---|
| `tls_client_handshake_loopback.sio`, parse-only, **no networking** | 36 | 241 | `ok=35 skip=0` | **255 / 272 (WRONG)** |
| same, minus `use tls::client::*` | 35 | 152 | `ok=18 skip=0` | 4 / 523 (correct) |
| same, plus **all 13 of `tls::client`'s own imports** | 35 | 224 | `ok=35 skip=0` | 4 / 523 (correct) |
| `use x509::cert::*` only | 16 | 145 | `ok=15 skip=0` | 4 / 523 (correct) |
| 36-module program with **`HsBuf.pos` renamed to `hs_pos`** | 36 | 242 | `ok=35 skip=0` | **4 / 523 (correct)** |

Instrumented reads inside the failing 36-module build:

```
D11DBG der_enter child.pos=4 tag.content_start=4          <- der_enter builds it correctly
D11DBG via_helper=4 e0=0 cert_inner.pos=255 cert_inner.end=803 seqtag.cs=4 seqtag.cl=799
```

`der_pos_of(&cert_inner)` (a helper reading `r.pos` through a typed
`&DerReader` parameter) returns **4**; the inline `cert_inner.pos` in the same
function returns **255**. The struct's memory is intact — only the caller's
field-load instruction is wrong. That single measurement rules out every
memory-corruption theory at once.

## What the arena diagnostic actually means (and why §D11's suspected area was a dead end)

`lower_array: arena_reset_skipped (call-arg scratch overflow)` is emitted by
`module_frontend.sio:5628/5791` when `mf_arebox_scan` overflows
`MF_AREBOX_CAP = 8192` recorded call-arg sites. The skip path is **fail-safe**:
it *declines* to call `__arena_reset`, so post-mark allocations stay live and
nothing can dangle. A skipped reset cannot corrupt anything — it only raises
peak memory. `skip=35` in the original report is a scale *indicator*, not a
cause.

Independently: the corruption reproduces at `ok=35 skip=0` (see the table
above) and in a 6-function program where the arena machinery never runs at all
(`ok=0 skip=0`). The arena hypothesis in `docs/handoff/souc_v0800_defects.md`
§D11 is disproved.

## Impact

Any `let (a, ...) = f()` where `a` is a struct and one of its field names is
also declared by *any other* struct anywhere in the linked program, at a
different field index, silently reads the wrong slot. Consequences observed on
this branch:

- `x509_parse_certificate` returns `tbs_start=255, tbs_len=272`, so
  `x509_verify_chain` hashes the wrong bytes and rejects a genuinely valid,
  genuinely self-signed, genuinely trusted certificate with
  `CHAIN_ERR_BAD_SIGNATURE`. Confirmed for both the RSA and the ECDSA P-256
  loopback tests.
- The failure mode is **silent and load-bearing on security**: no diagnostic,
  no crash, a plausible-looking wrong number, and a cryptographic verdict
  computed from it.
- The blast radius scales with program size in the worst possible way — adding
  an unrelated import can flip a previously-correct read, and removing one can
  hide the bug again.

Tuple-destructuring a struct is idiomatic and pervasive in this stdlib
(`x509/`, `asn1/`, `tls/`, `bignum/` all use `let (value, status) = ...`
throughout), so this is not a corner case.

## Proposed fix (compiler, `self-hosted/ir/lower.sio` + `self-hosted/parser/stmts.sio`)

Conservative and targeted, in the same shape as commit `88f91fae6`'s fix for
the `arr[i].field` base:

1. **Record the callee's tuple return element types per `fn_id`.** Every site
   that already does
   `fn_slot.return_struct_name = lower_opt_type_named_name(&fd.return_type)`
   (lower.sio:1777, 2256, 3115, 3249, 3377, 3695, 3789) additionally records,
   for a `TypeTuple` return type, the interned name-id of each element's named
   struct type into a module-level side table
   `LOWER_RET_TUPLE_NAMEID: [i64; IR_MAX_FUNCS * 4]` (512 KB BSS; interned ids,
   not `Name`s, because `Name` is 136 bytes). This mirrors the existing
   `MF_AREBOX_*` global-side-table idiom rather than growing `IrFunction`,
   whose struct literal is constructed at six sites across `ir.sio`,
   `optimize.sio`, `serialize.sio` and `ssa.sio`.

2. **Remember which function produced a tuple-typed local.** Add
   `tuple_ret_fn_id: [i64; 4096]` to the Lowerer's locals table with
   `bind_local_tuple_ret_fn_id` / `lookup_local_tuple_ret_fn_id`, an exact
   mirror of the already-proven `tuple_float_mask` pair (lower.sio:36, 7487,
   7503). Bind it in `lower_let_stmt_ref`'s `ExprCall` branch when
   `expr_call_return_struct_name_ref` comes back empty (i.e. a tuple return).

3. **Bind the element's struct type at the desugared read.** In
   `lower_let_stmt_ref`'s existing `ExprFieldAccess` branch, before the Box
   check: if `lower_name_tuple_index((*expr_box).name)` is `k >= 0` and the
   base is an `ExprIdent` with a recorded `tuple_ret_fn_id`, look up
   `LOWER_RET_TUPLE_NAMEID[fn_id * 4 + k]`, and if it names a registered struct
   layout, `bind_local_struct_type(s.name, that_name)`.

Nothing else changes: `field_idx_for_base_ref`'s existing `ExprIdent` branch
then finds a non-empty type name and routes through the struct-scoped
`field_idx_from_name`, exactly as it already does for
`let d = Decoy { ... }`.

**Defensive companion change (recommended, cheap, independent):** make
`field_idx_from_name_simple` emit a diagnostic (behind the existing
`lower_aggregate_diag_enabled()` gate, or a new `SOUNIO_WARN_FIELD_AMBIGUITY=1`)
when two or more registered struct layouts declare the queried field name at
**different** indices. That turns every remaining instance of this class from
silent-wrong-answer into an observable event, which is what cost this
investigation and the two before it most of their time. It changes no codegen.

**Not recommended:** hard-erroring on the ambiguity. Measured collisions in
this stdlib alone (`pos`, `len`, `value`, `signature`, `signature_len`) are
numerous enough that a hard error would break a large amount of currently
correct code that never reaches the fallback.

### Verification plan for whoever implements it

1. `tests/known_failures/madaros_tuple_destructure_field_name_collision_probe.sio`
   exits 0, and moves to `tests/run-pass/`.
2. The parse-only 36-module reduction (`/tmp/d11/repro_full.sio` shape in this
   dispatch's trail; regenerate from `tests/run-pass/tls_client_handshake_loopback.sio`
   by replacing everything after the `x509_parse_certificate` call with prints)
   reports `tbs_start=4 tbs_len=523`.
3. `tests/run-pass/tls_client_handshake_loopback.sio` reaches `CHAIN_OK`
   against a live `openssl s_server -tls1_3` (needs the server running and the
   cert regenerated — it expires 2026-08-27).
4. `bash scripts/run_sio_test_suite.sh` over at least the `x509_`, `tls_`,
   `asn1_`, `der_`, `struct_` and `knowledge_` prefixes, plus
   `tests/run-pass/knowledge_array.sio` specifically — commit `88f91fae6`'s own
   notes record that growing `field_idx_for_base_ref`'s body in place
   SIGSEGV'd that test, so any change in this neighbourhood must re-check it.
5. Rebuild via `bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros`
   — **called directly, never wrapped in `scripts/dev/souc-build-lock.sh`**,
   which self-deadlocks (CLAUDE.md, "Concurrency discipline").

## Immediate stdlib mitigation (available now, not applied by this dispatch)

Renaming `HsBuf`'s `pos` field to `hs_pos` in `stdlib/tls/client.sio` (four
call sites, all inside `hsbuf_try_take_message`/`hsbuf_new`) makes the full
36-module program return the correct `tbs_start=4, tbs_len=523` — verified
2026-08-26. This is a workaround for one collision, not a fix: any future
struct anywhere in the closure that declares `pos` at an index other than 1
re-arms it, with no warning.

## What was ruled out

- **The arena/scratch reset machinery** (`module_frontend.sio`'s
  `mf_arebox_scan` / `__arena_reset` / `arena_reset_skipped`) — the skip path is
  fail-safe, and the defect reproduces with `skip=0` and with the machinery
  never running.
- **Total module count / merged function count.** 35 modules and 224 merged
  functions with all of `tls::client`'s imports present: correct. 6 functions
  in one file: wrong.
- **D9's cross-struct field-name collision at tuple *return* boundaries.** D9's
  own mechanism (a colliding field name corrupting the returned aggregate's
  memory) is not what happens here: the memory is provably intact
  (`der_pos_of(&cert_inner) == 4`); only the caller's field-index resolution is
  wrong. Same *ingredient* (a colliding field name), different stage.
- **Prior heavy computation / heap state.** The corruption is present at the
  very first call, in a program that does nothing else.

## Resolution (2026-08-26, commit `3ec2d971d`)

Implemented as proposed above, with one simplification: no per-`__tupN` table
is needed. `parse_block` drains the desugared element bindings immediately
after the `let __tupN = f()` they belong to and in source order, so remembering
only the **most recent** tuple-producing temp (`LOWER_TUPLET_TMP_ID`) and its
callee (`LOWER_TUPLET_CALLEE_ID`, both interned ids) is sufficient to link
them, and a name mismatch simply declines.

Landed in `self-hosted/ir/lower.sio`:

- `LOWER_RTT_NAMEID: [i64; 262144]` — element struct name-ids for up to 4
  tuple-return elements, keyed by the callee's **interned name**, not by
  `fn_id`. `fn_id`s are remapped by `ir_merge_place_and_remap_function`; names
  are not — name identity is exactly what the cross-module merge dedups on.
- `lower_type_list_named_name_at`, `lower_record_ret_tuple_names`,
  `lower_ret_tuple_elem_name` — record/read helpers.
- One `lower_record_ret_tuple_names(...)` call added beside each of the eleven
  existing `return_struct_name = lower_opt_type_named_name(...)` assignments.
- `lower_let_stmt_ref`'s `ExprCall` branch remembers the temp + callee when the
  return is not a single named struct; its `ExprFieldAccess` branch binds
  element `k`'s struct type when the field name is a tuple index and the base
  is that remembered temp.

Every step is guarded — non-tuple return, non-struct element, unregistered
layout, name-pool overflow, or a temp-name mismatch all fall through to the
previous behaviour. The table can only ever *add* resolution, never
mis-resolve.

### Verification

| Check | Before | After |
|---|---|---|
| `tests/run-pass/tuple_destructure_field_name_collision_regression.sio` | exit 1, `r.pos=0` | **exit 0, `r.pos=222`** |
| 36-module TLS parse-only repro, unmodified stdlib | `tbs_start=255 tbs_len=272` | **`tbs_start=4 tbs_len=523`** |
| `souc check self-hosted/compiler/main.sio` (120-module compiler closure) with the fixed compiler | — | `run_check_mode: verdict=0`, `check: OK` |
| `scripts/run_sio_test_suite.sh --filter-prefix {a..i} --jobs 3` (886 tests) | 391 pass / 129 fail | **byte-identical** pass/fail counts and failing-test identities |
| `--filter-prefix {x509_,tls_,asn1_,der_,struct_,knowledge_,tuple_,cert_}` | 61 pass / 13 fail | **byte-identical** |

`tests/run-pass/knowledge_array.sio` SIGSEGVs (`run exited 139`) on **both**
the pre-fix and post-fix compilers — pre-existing, unchanged, and unrelated;
noted here because commit `88f91fae6` records that test as the canary for
changes in this neighbourhood.

The compiler ELF itself (`artifacts/self-hosted/madaros`) is gitignored, so the
fix ships as source: re-run
`bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros`
(directly — never wrapped in the build lock, which self-deadlocks) to pick it
up. Build wall time measured here: ~4 minutes on the 64-core pod.

### Still open after this fix

- **The fallback itself.** `field_idx_from_name_simple` remains a global,
  name-only, first-registered-match scan for every base whose struct type still
  cannot be resolved. This fix removes the tuple-destructure route into it; it
  does not remove the fallback. The defensive diagnostic proposed above
  (warn when two layouts declare the same field name at different indices) is
  **not** implemented and remains the cheapest way to make the remaining
  instances of this class observable rather than silent.
- **Tuple arity > 4** and tuple elements that are not plain named types are not
  recorded — they fall back to the old behaviour, silently.
- **Finding 26** (a struct containing an array-of-structs field, written into a
  doubly array-indexed target) from the 2026-08-24 dispatch is untouched.
- `tests/run-pass/tls_client_handshake_loopback.sio` reaching `CHAIN_OK`
  against a live `openssl s_server -tls1_3` has **not** been re-run: it needs
  the server up and the embedded test certificate regenerated (it expires
  2026-08-27). The parse-only reduction proves the corrupted input to
  `x509_verify_chain` is gone; the end-to-end handshake verdict is not yet
  re-measured.

## Trail

- `docs/handoff/souc_v0800_defects.md` §D11 (original report, superseded here)
- `docs/audit/X509_ARRAY_STRUCT_FIELD_CORRUPTION_DISPATCH_2026-08-24.md`
  (the sibling defect this one is the open half of; Finding 25)
- `docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md` Findings 24-26
- `self-hosted/parser/stmts.sio` — tuple-let desugaring side table
- `self-hosted/ir/lower.sio` — `lower_let_stmt_ref`, `field_idx_for_base_ref`,
  `field_idx_from_name_simple`, `bind_local_struct_type`
