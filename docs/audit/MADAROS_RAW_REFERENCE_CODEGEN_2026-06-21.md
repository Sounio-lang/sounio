<!-- docs:meta
topic_id: repo.docs.audit.madaros-raw-reference-codegen-2026-06-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-raw-reference-codegen-2026-06-21
-->

# Madaros raw-reference codegen — forensic dispatch

**Date:** 2026-06-21
**Author:** Claude (Codex-coordinated; this is the object-model/codegen lane)
**Status:** SCOPED — implementation is mechanically tractable but **soundness-blocked**; do not land the codegen until the frontend gains escape analysis (see §4).
**Related:** #392 (box-deref codegen + raw-ref loud-fail), #393 (line-start `*b`/`&x` parse), #394 (`Box<T>` param deref E005).

---

## 1. Symptom

Raw references are rejected at native_v2 codegen:

```sounio
fn load(r: &i64) -> i64 { *r }
fn main() -> i64 { let x = 42; load(&x) }
```

→ `Native compilation failed: native_driver_function_codegen_failed`.

Both `&x` (OpRef) and `*r` (deref of a true `&T`) fail. `&!x` (OpRefMut) too.

## 2. Where it fails (contradiction resolved)

The active native_v2 path is the **core_ir** loop (`nc_core_*`), not the
`native_v2_emit_*` / MIR `MachineInstr` layer. In
`self-hosted/native/codegen_x86_linux.sio`, the `IrUnaryOp` case:

```
6460  } else if instr_op == IrOpcode::IrUnaryOp {
6469      if uop == UnaryOp::OpNeg { ... }
6474      else if uop == UnaryOp::OpNot { ... }
6479      else {
6480          return false        // <-- OpRef / OpRefMut / OpDeref(raw &T) -> codegen_failed
6481      }
```

This is #392's deliberate fail-loud. Note: the `native_v2_emit_unary_code`
helper (line ~4990) *does* have an `OpDeref` arm, but it is on the **inactive**
MIR path — it does not run for these programs. So both `&x` and `*r` die at 6480.

Lowerer side (`self-hosted/ir/lower.sio` `lower_unary_expr_ref`, ~6772): box
deref is already rerouted to `IrFieldGet(0)`; everything else emits
`ir_unaryop(dst, e.un_op, src)` where for `&x`, `src` is x's **loaded value**,
not its address.

## 3. The implementation is mechanically tractable (~10 lines)

Decisive fact: in the core_ir path **every IR temp/register is an addressable
stack slot**:

```
nc_core_load_temp_to_rax(temp)  = load  rax, [rbp + ir_slot_offset(temp)]
nc_core_store_rax_to_temp(temp) = store [rbp + ir_slot_offset(temp)], rax
```

So:
- **`*r` (OpDeref of raw &T):** load src (the pointer) to rax, `emit_load_rax_mem_rax` (rax ← [rax]), store dst. The load primitive already exists (used at line 4997). Always safe given a valid pointer.
- **`&x` (OpRef):** the operand already sits at `rbp + ir_slot_offset(src)`, so `lea rax, [rbp + ir_slot_offset(src)]` (the existing `MIR_OP_LEA_STACK` primitive at line 5333), store dst. Mechanically trivial.

(Lowering nuance: the lowerer copies x's value into a fresh temp before `&`, so
`&x` would yield the address of that **copy slot**, not x's canonical slot. That
is fine for read-only `*(&x)` — you read back the same value — but means
mutation through `&!T` would not write x. `&!T` is deferred regardless.)

## 4. SOUNDNESS BLOCKER — why this must NOT land yet

Enabling `&x` re-introduces a **silent dangling-pointer miscompile** for
escaping references — the worst class for an epistemic language, and exactly
what the loud-fail prevents.

**Verified on a clean current-main build** (run 27920658457, off
`probe/rawref-baseline` @ `ad69883df`, so #393's parser fix is in — no
line-start `*r` contamination), `--check` **ACCEPTS** all of:

| Program | Frontend verdict | If codegen enabled |
|---|---|---|
| `fn bad() -> &i64 { let x = 7; let r = &x; r }` | **ACCEPT** | returns a dangling stack address |
| `struct Holder { p: &i64 } fn bad() -> Holder { let x=7; let r=&x; Holder{p:r} }` | **ACCEPT** | escapes a dangling ref in a struct |
| `fn ok() -> i64 { let x=7; let r=&x; *r }` (safe, in-frame) | ACCEPT | correct |
| `fn load(r:&i64)->i64{*r} ... load(&x)` (safe) | ACCEPT | correct |

The borrow-checker (`try_shared_borrow` / `try_exclusive_borrow`) tracks
**in-scope aliasing** but performs **no escape/lifetime analysis** — it does not
reject returning or storing a reference to a local. So the loud-fail at 6480 is
the *only* thing preventing the dangling-pointer miscompile, and it is
**correct** until escape analysis exists.

There is no safe useful subset to ship early: `*r` alone is useless without
`&x` to create a reference, and `&x` is precisely the escape-prone operation.

## 5. Recommended path (dependency order)

1. **Frontend escape analysis (prerequisite).** Reject a reference to a local
   when it can outlive the local: returning a `&T`/`&!T` to a local, storing it
   in a returned/heap aggregate, etc. This is the real work and the gate.
2. **Then** add the ~10-line core_ir codegen from §3 (lea for OpRef, load for
   OpDeref) and remove the 6480 loud-fail for the now-provably-safe cases.
3. Lowering: make `&<local ident>` lea the local's **canonical** slot (not a
   value copy) so a future `&!T` can mutate through the reference.

**Scope even after escape analysis** — land `&<local>` + `*ref`, read-only,
non-escaping, primitive inner first. Defer: `&!T` mutation-through-ref, refs to
struct fields / array elements, reference returns, any escaping ref.

## 6. Escape analysis — completeness requirement + channel inventory

A **partial** escape analysis does NOT unblock codegen: if it misses any escape
channel, enabling ref codegen still ships a silent dangling pointer. So the
loud-fail (§2) must stay until escape analysis is **complete** across every
channel by which a local-rooted reference can outlive its frame.

"Complete" = control every escape channel. Channels you can cheaply **forbid**
(reject constructing) cost far less than channels you must **analyse** — and a
measurement of `self-hosted/` + `stdlib/` shows the forbiddable channels are
near-empty, so this is bounded, not a multi-session lifetime system:

| Channel | Count | Disposition |
|---|---:|---|
| ref-typed struct fields (`field: &T`) | 4 | forbid (or special-case); near-empty |
| ref-typed globals | 5 | referent is static → benign |
| `Box<&T>` / refs in arrays | 0 | forbid; none exist |
| ref returns (`-> &T`) | 5 | **analyse** — all 5 are param-derived (valid), must accept |

The only channel needing real dataflow is **return** (and escaping stores):
distinguish a reference whose place-root is a **frame-bound local** (reject) from
one rooted in a **reference parameter** / global (accept). The 5 stdlib ref
returns (`get(self: &MetaResult) -> &Epistemic`, `atlas_get_region(atlas: &Atlas)
-> &AtlasRegion`, `current_token`/`peek_token`, `het`) are all param-derived and
**must not** be false-rejected.

### Proposed return-provenance analysis (bounded)
1. Tag each binding's reference **provenance**: `&<local>` → LOCAL; `&<param>` /
   `&param.field` / `&param[..]` → PARAM (outlives); global → STATIC.
2. Propagate through `let r = &x` so `r` carries x's provenance.
3. At a `return <e>` where the return type is `&T`/`&!T` (or an escaping store):
   reject if `<e>`'s provenance is LOCAL.
4. Forbid the exotic channels (ref struct fields beyond the 4, `Box<&T>`, ref
   arrays) outright until they're analysed.

Borrow infra note: `borrow.sio` `BorrowEnv` tracks borrow **state**, not ref
**provenance** — provenance is a new per-binding tag. `borrows.sio`
(`BorrowChecker` v2) is more elaborate but appears not fully wired; confirm
before building on it.

## 7. Q1 literature grounding (deep-research, 2026-06-22)

Two adversarially-verified deep-research passes (PLDI/POPL/OOPSLA/ICFP/TOPLAS;
3-vote verification) converge on a design that **matches §6's return-provenance
proposal** and supplies the soundness theorem to mirror. Verified claims:

**Established floor.**
- *Second-class values* (Osvald, Essertel, Wu, Rompf — OOPSLA 2016): one binary
  qualifier `n` (first/second-class) on STLC; Coq-mechanized type soundness
  (Thm 3.5), leak-freedom "evaluation never leaks stack references" (Thm 3.2),
  and Cor 3.6 "well-typed programs respect stack-based lifetimes for second-class
  values". Three baseline rules ≈ our spec: (i) first-class fn may not capture a
  second-class free var, (ii) fns may not **return** second-class values, (iii)
  second-class may not be **stored** in fields / mutable vars. [verified 2-0]
- *Tofte–Talpin region inference* (Inf.&Comp. 1997; Tofte–Birkedal TOPLAS 1998):
  `letregion` LIFO + **region polymorphism** (region vars as extra params) is the
  theory for "reject frame-local escape, accept parameter-rooted pass-through";
  co-inductive consistency Thm 6.1. [verified 3-0]

**Verified frontier — the propagation rule (answers §6's open design question).**
- *Capture calculus* CC<:□ (Boruch-Gruszecki, Odersky, et al. — TOPLAS 2023,
  arXiv 2105.11896): capture sets on types; Coq progress+preservation; the
  **capture-prediction lemma** (Cor 2.6 / Lemma 4.11) = the capture set is a
  *sound over-approximation* of a value's free vars. [verified 3-0]
- **The exact decidable propagation function** `cv(t)` — syntax-directed, no
  fixpoint/quantifier: `cv(x)={x}`; `cv(let x=s in t)=cv(s) ∪ cv(t)\{x}`;
  `cv(λ(x).t)=cv(t)\{x}`; `cv(box)={}`. The let-rule (**union-minus-binder**) is
  exactly taint-through-binding: any subterm rooted in a frame-local taints the
  binding; a parameter-rooted ref carries only the parameter. [verified 3-0]
- **Escape side-condition**: reject a result whose type would capture the root
  capability `cap`; *avoidance* widens an escaping local's type to the smallest
  supertype not mentioning it (→ `{cap}` when fresh) → rejected. Scala 3 capture
  checking ships this diagnostic ("…persists longer than its allowed lifetime").
  [verified 3-0]
- **Effects** (`&!T` / algebraic effects, our Q4): capture calculus already
  handles effect handlers via a *single* non-escape side-condition in the
  `(handle)` rule (handler's `{x}` not a subcapture of the result type) — the
  light machinery suffices for effects; full reachability is only forced by
  genuinely **shared mutable** state. [verified 3-0]

**Caveats:** Scala cc tracks capabilities, not literal stack-frame refs (sound
*analogy*, not identity). Reachability-types specifics (Bao/Wei/Rompf, freshness,
Preservation-of-Separation) and the exact Q2 undecidability bound (semi-unification,
Kfoury 1990) were rate-limited to 0-0 *abstain* (not refuted) — treat as leads.

### Recommended design (research-grounded, supersedes §6 step list)
**Land first — binary provenance qualifier, decidable, sound:**
- Tag each binding LOCAL vs FIRST-CLASS (outlives-frame): `&<local>` → LOCAL;
  `&<param>` / `&param.field` / global → FIRST-CLASS.
- Propagate by the `cv` taint rule: `let r = e` ⇒ `r` inherits LOCAL if any
  frame-local is in `cv(e)` (union-minus-binder over let/field-projection/reborrow).
- **Reject** any `return <e>` (ret type `&T`/`&!T`) or escaping store where
  `cv(<e>)` contains a frame-local; **accept** parameter/global-rooted.
- Keep regions **monomorphic** to stay decidable (avoid the semi-unification
  boundary). Soundness theorem to mirror: second-class Thm 3.5 + Cor 3.6.
- This is a one-bit collapse of a capture set — coarser, so it will reject a few
  valid mixed-origin programs a capture set would accept (acceptable first cut).

**Upgrade path (only if the binary qualifier over-rejects in practice):**
binary → finite **capture sets** (CC<:□: per-variable identity, fewer false
rejects, effects via `(handle)`) → full **reachability qualifiers** (shared
mutable `&!T`, Preservation-of-Separation; System Capless, Lean-mechanized,
SPLASH 2025).

Sources: arXiv 2105.11896 (CC<:□/TOPLAS'23), dl.acm.org/10.1145/3022671.2984009
(Osvald'16), ToTa Inf.&Comp.1997 / TOPLAS'98, arXiv 2509.07609 (Capless),
arXiv 2307.13844 (reachability — lead), docs.scala-lang.org scala3 cc.

## 8. Implementation status — increment 1 LANDED (2026-06-22)

The frontend escape analysis (§7 recommended design) is implemented on branch
`feat/escape-analysis-raw-refs` (commit 648d79300, `borrow.sio` + `check.sio`,
+129): a binary provenance qualifier on `BorrowEntry` (0=first-class, 1=LOCAL),
propagated by `checker_expr_provenance` (the cv-taint rule), with the body-tail
provenance captured in a `Checker.escape_provenance` side-channel (mirrors
`last_literal_kind`) and checked at the return-compat + explicit-return sites.
New diagnostic **E091** "reference to a local variable escapes its scope".

**Verified (off-pod build run 27937904382, build + Madaros gate green):**
- The 3 escape cases REJECT with E091: `let r=&x; r` returned, `&x` returned,
  `Holder { p: &x }` returned.
- The 5 valid cases ACCEPT: `&self.f` returned, `&self.f` via let, ref-param
  pass-through, in-frame `*r`, `&x` passed to a call.
- **Zero false-rejects** across a 160-file sweep (all stdlib + self-hosted
  check/parser/ir) incl. the 5 real stdlib `-> &T` returns.

**Deferred channels (increment-1 boundary — INCOMPLETENESS, not unsoundness;**
**these slip the frontend but the codegen loud-fail at §2 still guards them, so**
**no silent miscompile — they MUST be closed before raw-ref codegen is enabled):**
- Interprocedural: a call returning a ref tainted by a `&local` argument
  (`fn id(r:&i64)->&i64{r}; id(&x)` returned) — needs call-result taint = OR of
  ref-arg provenances. CONFIRMED slips (ACCEPT).
- Nested block / if tails (`{ let x=7; &x }` as the function tail) — needs
  `checker_expr_provenance` to recurse through `ExprBlock`/`ExprIf`. CONFIRMED
  slips (ACCEPT).
- Arrays of refs and ref-typed struct-field reads (the 4 `field: &T` sites) —
  needs array-lit taint + ref-field projection.

**Property:** the analysis only ADDS rejections (E091); it never enables codegen,
so it cannot introduce a miscompile. Worst case is a false-reject (verified 0) or
a missed escape (caught later by the loud-fail). The unwired 1846-line
`self-hosted/check/lifetimes.sio` NLL checker remains the precision/completeness
upgrade path.

## 9. Reproducers

`/tmp/rrtest/{esc_return,esc_struct,safe_inframe,safe_param}.sio` (see §4 table).
Baseline compiler: artifact `madaros-built` from run 27920658457.
