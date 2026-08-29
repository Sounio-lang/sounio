<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.tuple-match-feature-design-2026-06-03
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.tuple-match-feature-design-2026-06-03
-->

# Tuple-patterns-in-match-arms — implementation design (canonical compiler) 2026-06-03

Implements the last 2 modular survivors' root cause (Knowledge: types_equal uses
`match (a.inner, b.inner) { (Some(ia), Some(ib)) => ... }`). PRE-EXISTING canonical bug (crashes
old bin/souc). Repro: TUPLE_MATCH_DEREF_REPRO_2026-06-03.sio. HIGH RISK: touches lean_single.sio
match codegen → must preserve the bootstrap fixed point (gen2==gen3) + run-pass + examples.

## Confirmed facts (from reading lean_single.sio)
- Match handler: arm loop at ~20253 (`compile_stmt`/match-expr). Scrutinee compiled via compile_or,
  stored in `match_slot` (a local holding a POINTER to the value/tuple slots).
- Single tagged pattern emission (20519+): tag = `[match_slot_ptr + 0]`; `cmp disc; jne skip`;
  payload bind = `[match_slot_ptr + 8]` → var. Merge via `em(0xe9); match_ends[arm_count]=CL; em32(0)`.
- Arm parser (20253-20320) handles tk 57=Some(/58=None/59=Ok(/60=Err(/3=ident|wild/4=lit/or-pattern.
  NO case for tk 6 = `(` → tuple pattern. THIS is the gap.
- Tuple layout (tuple_destructure_from_ptr_x86, 17040): elem0 at byte_off 0, elem1 at
  `tup_first_slots*8` (=8 for a 2x 1-slot tuple). first/total slots from `tup_hash` (MATCH_SCRUT_HASH)
  decode (`(h%1000000)/1000` first, `h%1000` total) or TUP_CACHE. Each Option element is a 1-slot
  POINTER; its tag=`[elem_ptr+0]`, payload=`[elem_ptr+8]`.

## Plan
### 1. Parse (add an `if arm_tk == 6` branch in the arm loop, before the dispatch)
Consume `(`; loop parsing N element sub-patterns (reuse the Some/None/Ok/Err/ident/wild/lit logic
into per-element arrays: TPAT_DISC[i], TPAT_TAGGED[i], TPAT_BIND_NS[i]/NE[i], TPAT_HAS_BIND[i],
TPAT_IS_WILD[i]) separated by `,` (tk 28) until `)` (tk 7). Record elem_count.

### 2. Codegen (emit BEFORE the body; one shared `skip` target)
- Compute per-element byte offsets from MATCH_SCRUT_HASH (mirror tuple_destructure decode:
  off[0]=0, off[1]=tup_first_slots*8; for >2 elems generalise via running slot sum).
- For each element i with TPAT_TAGGED[i]==1 and not wild:
    emit_load_var(match_slot)              ; rax = tuple ptr
    if off[i]!=0: lea rax,[rax+off[i]]     ; rax = &elem slot
    mov rax,[rax]                          ; rax = elem Option ptr
    mov rax,[rax]                          ; rax = tag  ([elem+0])
    push; movabs disc[i]; pop rcx; cmp rcx,rax
    jne skip   (collect ALL element jnes → patch them ALL to the shared skip)
- After all tests pass, for each element with TPAT_HAS_BIND[i]:
    emit_load_var(match_slot); [+off[i]] → elem ptr; mov rax,[rax+8] (payload); var_add+store;
    propagate inner type via option_hash_inner_ty over the tuple element's Option hash.
- `if TK==52 EP++` (=>); compile body (block or single stmt, as the single-pattern path);
  match_arms++; arm type-union as existing; `em(0xe9); match_ends[arm_count]=CL; em32(0)`; arm_count++;
  patch ALL collected element-skip jnes to CL (after the body jmp).
- a64 path: mirror in the a64 dispatch (~29xxx).

### 3. Type check
Verify the scrutinee is a tuple of the right arity; each element sub-pattern's enum matches the
element type. (Minimal: accept; the modular checker already type-checks via the in-place spine.)

### 4. Validate (MANDATORY, in order — do NOT commit if any fails)
- teq repro: souc_gen2 teq.sio → run → correct answer (rc reflects teq, not 139).
- Bootstrap fixed point: gen2==gen3 (md5). THE critical guard.
- run-pass divergence sweep: 0 real (non-deterministic-only) divergences vs old bin/souc.
- examples divergence sweep: 0 real divergences.
- Modular census: confirms epsilon_comparison_valid + knowledge_octonion_inner CRASH→non-crash
  (CRASH 2→0).

## Status
Design complete (codegen offsets + parse + integration points confirmed by reading). Execution is a
focused ~60-100 line codegen addition to the arm loop (x86 + a64) + full bootstrap/sweep validation —
nested-write-scale, best done with fresh context given the bootstrap-fixed-point risk.
