<!-- docs:meta
topic_id: repo.docs.audit.r2-3-compiler-tuple-return-bug.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-3-compiler-tuple-return-bug.dispatch
-->

# DISPATCH R.2.3 — Compiler bug: `(Struct, f64)` tuple-return scalar-field corruption

**Opened:** 2026-05-16
**Predecessor:** R.2.2 (synthesis at `/tmp/sounio_pure_1/r2_2_b/SYNTHESIS.md`)
**Class:** souc compiler internals. Outside R.2.1 §7 scope; this dispatch is the carve-out.
**Priority:** P2 — does NOT block PBPK28 (workaround shipped in `stdlib/random/park_miller.sio`); does block any future stdlib code that returns `(Struct, scalar)` tuples.
**Bootstrap risk:** HIGH. Any compiler patch MUST clear `bash scripts/ci/lean_single_fixed_point_gate.sh` and the full `native_v2_cpu_compiler_umbrella_gate.sh` (6 sub-gates) before commit.

---

## §0 — Sounio-Pure constraint (preserved from R.2.1)

This dispatch may read `self-hosted/compiler/native_compile_driver.sio` and other compiler sources, run `gdb` and `objdump` against compiler outputs, and propose patches in the form of unified diffs. **No Python/JS/R for diagnostic instrumentation.** GDB scripting (`.gdb` files / `-batch -ex`) is permitted as an inspection-only tool — same exception class as `objdump`. Compiler patches land via standard Sounio compiler-source edits.

If the fix requires Python anywhere in the diagnostic loop (e.g. parsing core dumps), **PARA E REPORTA** — that signals the surface needs a different attack.

---

## §1 — Bug definition

**Symptom.** Two consecutive `println(r1.1)` calls, where `r1: (S4, f64)` is the result of a function returning a 5-field struct-tuple computed via a chained inner call:

```sounio
let r1 = step_outer(rng)
println(r1.1)    // correct: 1362.073482
println(r1.1)    // wrong: 0.000000
```

The first read is correct; the second (and all subsequent) reads return 0.0. Once corrupted, the storage stays corrupted until the call site is re-entered.

**Trigger conditions** (empirically narrowed; see R.2.2 SYNTHESIS table):
- Return type is `(S4, f64)` with `S4 = {a,b,c,d: i64}` — 5-field tuple, last field f64.
- Outer body chains an inner `(S4, i64)` call: `let inner = step_inner(s); (inner.0, (inner.1 as f64) / 1e9)`.
- Inputs trigger big-magnitude arithmetic (seed 20260516 fires; seed 10 does NOT). Magnitude itself isn't the cause — it shifts stack layout in a way that exposes a latent invariant violation.
- The CALL `println(r1.1)` is what corrupts. A bare read `let _ = r1.1` does NOT corrupt (`bare_read_safe.sio`).
- Binding `let f1 = r1.1` BEFORE the first `println(r1.1)` preserves a stable scalar copy in `f1`. Binding AFTER captures the already-corrupted 0.

**Canonical repro.** `repro/canonical.sio` — two `println(r1.1)`, no other code, deterministically reproduces.

---

## §2 — What R.2.2 ruled out (do not re-litigate)

1. **NOT `UFN_CALL_STRUCT_RESULT` slot reuse** in `native_compile_driver.sio:6943`. Static analysis of `repro/canonical.sio` (`nt10.elf`) shows two SRET buffers at non-overlapping offsets (-0x90 for r1, -0x100 for r2 in `nested_tuple.elf`). No slot collision in caller.
2. **NOT slot reuse of the saved-pointer slot `-0x98(%rbp_main)`.** `objdump | grep` over main confirms exactly one write at the post-call save; no other write between the two `println` calls.
3. **NOT raw memory writes to `-0x70(%rbp_main)`** (the f64 storage byte) from any in-main instruction. Static asm grep returns zero such writes between the calls.
4. **NOT a stdlib `println` body bug touching caller frame.** `println`'s frame is below caller's frame; no positive-offset `(%rbp)` accesses; syscall buf is local.
5. **NOT a nested-tuple-return aliasing issue.** Bug fires from a single `step_outer` call; the two-call discriminator (nt3) showed both calls produce same-pattern corruption regardless of call shape.
6. **NOT a `dst_pcg64` source bug.** Repro contains zero PCG code.

**Implication:** corruption is in code that static asm reading didn't surface. Candidates: ABI-level temporary slot the compiler uses without referencing `(%rbp)` (e.g. via `(%rsp)`), some indirect write through a register set up earlier and not visible in the local window, or a write through a register that aliases to an unexpected stack location during the println call setup.

---

## §3 — Attack plan (multi-session OK)

### Phase A — UPDATE 2026-05-16: Sounio-native run executed; surface tightened

Ran Sounio-native Phase A in environment without gdb. Findings narrowed the bug surface significantly. See `instrumentation/field_scope.sio` and the §3.A-results subsection below before doing the gdb run.

**Key constraint to falsify in 30s with gdb (next session, top priority):**

After the first `println(r1.1)` call returns, main's SRET buffer at offsets +0/+8/+16/+32 hold println-internal-scratch values; offset +24 is **untouched** from its pre-call value. The four corrupted slots are not a wholesale buffer overwrite — three consecutive 8-byte writes (at +0/+8/+16 from some base) plus a separate write at +32. The "writer" register base needs gdb identification:

```
gdb -batch \
    -ex 'starti' \
    -ex 'b *main+<offset-of-add-rsp-0x10-after-println-call>' \
    -ex 'c' \
    -ex 'awatch *(uint64_t*)($rbp - 0x90)' \
    -ex 'awatch *(uint64_t*)($rbp - 0x78)' \
    -ex 'c' \
    /tmp/canonical.elf
```

Hit on `-0x90` will fire; hit on `-0x78` should NOT fire during println execution (confirms the "+24 untouched" constraint). The instruction address from the `-0x90` hit + the register state at that instruction pins the writer.

**Sounio-native Phase A (executed):**

### Phase A — pin the writer with gdb watchpoint (1 session) [ORIGINAL PLAN — partially executed; gdb portion deferred]

Goal: identify the exact instruction address that writes 0x0000000000000000 to the f64 storage byte.

1. `./bin/souc compile /tmp/sounio_pure_1/r2_3_compiler_tuple_bug/repro/canonical.sio -o /tmp/canonical.elf`
2. Determine `&r1.1` in main's frame. From R.2.2 disasm: it's at `%rbp_main - 0x70` (or whatever the new compile produces — re-derive each rebuild).
3. Run under gdb:
   ```
   gdb -batch \
       -ex 'starti' \
       -ex 'b *main+OFFSET_AFTER_FIRST_CALL' \
       -ex 'c' \
       -ex 'awatch *(double*)($rbp - 0x70)' \
       -ex 'c' \
       /tmp/canonical.elf
   ```
   The `awatch` triggers on read+write. Filter to writes by checking value-change in the hit record.
4. Record: instruction address, instruction bytes, %rsp / %rbp at hit, register state.
5. Map the instruction address back to compiler-driver source. Each emitted instruction in `native_compile_driver.sio` is produced by a specific UFN handler — grep for the byte sequence to find the emit site.

Acceptance: a single `(emit_site, register_used, intended_target_slot)` triple that explains why the write lands at `-0x70(%rbp_main)` when it shouldn't.

### §3.A-results — Sounio-native Phase A findings (2026-05-16)

Ran `instrumentation/field_scope.sio` (reads `r1.0.{a,b,c,d}` and `r1.1` before AND after the first `println(r1.1)`). Output values:

| Offset (rel SRET base) | Field | Before f64 println | After f64 println |
|---|---|---|---|
| +0 | r1.0.a | 340518492412 | **4696837146684686336** (= 0x4131F50DDDF50000, f64-bit-pattern of ≈1362073, i.e. 1362.073482 × 1000) |
| +8 | r1.0.b | 340518778132 | **1000000** (= 10^6, the float-format scale at precision 6) |
| +16 | r1.0.c | 340517971397 | **6** (the precision argument passed to println) |
| +24 | r1.0.d | 340518240310 | 340518240310 **UNCHANGED** |
| +32 | r1.1 (f64) | 1362.073482 | 0.0 |

**Asm-level observations:**

1. `println(r1.0.X)` for i64 fields is **inlined** as itoa+syscall in main; does not call any function; does not corrupt.
2. `println(r1.1)` for f64 **calls** an actual function (in `field_scope.elf`, at 0x786d). All corruption stems from this call.
3. f64 println body (`/tmp/fs_println.txt`, 952 lines) accesses locals only at NEGATIVE `(%rbp)` offsets in range -0x8..-0x120. Statically, its frame should be disjoint from caller's frame.
4. main's SRET buffer for `r1` is at `-0x90(%rbp_main)`; f64 println's frame in main coordinates is `[rbp_main - 0x1c0, rbp_main - 0x370]`. **No static (%rbp)-relative overlap** between println's accessed addresses and main's SRET buffer.
5. The corruption values **ARE** println's scratch (precision, 10^precision, intermediate float ×10^precision for digit extraction). So either:
   - (a) Some register-indirect write inside println aliases main's SRET buffer address (likely candidate: a register that step_outer left holding the SRET pointer is not restored across the call, and println uses it as a scratch base); OR
   - (b) main's `%rbp` is itself modified during the call (so post-call `-0x98(%rbp)` accesses different memory), explaining why "static analysis says nothing writes to -0x70" — main itself is reading from the wrong location.
6. Grep for the 0/+8/+16 cluster-write pattern in println body returned **no matches**. Either the cluster uses a different encoding form (e.g., 32-bit imm rather than 8-bit displacement) or hypothesis (b) is correct.

**Constraint for gdb session:** +24 is untouched. Three consecutive writes at +0/+8/+16 from some register base + one separate write at +32. Plus: hypothesis (b) is gdb-cheap to falsify — set `awatch` on `$rbp` itself; if it fires inside println, that's the answer.

### Phase B — minimal source patch (1 session)

Goal: change one emit site so it writes to its intended slot, not the f64 storage byte.

1. Inspect the emit site from Phase A. Likely candidates by symptom class:
   - cvtsi2sd / f64-cast lowering reusing a stack slot.
   - `push %rax` for arg-passing whose offset calculation collides because of `add $0xa0` frame size vs SRET-buffer-extends-by-one-f64-field interaction.
   - Tuple `.1` field-extract on `(Struct, f64)` where the compiler's field-offset table off-by-one'd.
   - SRET-buffer-size accounting that thinks `(S4, f64)` is 4 fields (32 bytes) instead of 5 (40 bytes) when allocating the caller's slot — making the f64 land in the next local's slot.

   Hypothesis worth testing first: **caller allocates SRET buf as 4 i64s (S4 only) instead of 5 fields (S4 + f64).** Test: dump frame map for main; if `r1` is given 32 bytes instead of 40, the f64 field of step_outer's return lands one slot beyond and aliases the next local.

2. Patch. Recompile compiler.

3. Verify `repro/canonical.sio` prints 1362.073482 twice.

4. Verify `repro/workaround_proof.sio` still prints stable values (no regression).

### Phase C — bootstrap & regression gates (1 session, non-negotiable)

1. `bash scripts/ci/lean_single_fixed_point_gate.sh` — MUST PASS (stage2==stage3 bit-identical).
2. `bash scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` — 6/6 sub-gates MUST PASS.
3. Re-run stdlib test sweep for any regression in `(Struct, scalar)`-returning code: grep `stdlib/` for `-> (` tuple-return signatures, run the corresponding test files.
4. Diff-driven review: smallest patch that fixes the bug, no surrounding cleanup.

### Phase D — restore stdlib PCG64 (1 session)

Once compiler patched:
1. Remove the deprecation header from `stdlib/random/distributions.sio`.
2. Re-validate `dst_pcg64_next_f64` self-test (previously failed under buggy compiler).
3. Update `stdlib/random/lib.sio` guidance header — point users back to PCG64 as default; keep `park_miller` as low-quality-acceptable lightweight option.
4. Update R.2.2 SYNTHESIS.md status to "RESOLVED in R.2.3".

---

## §4 — Out of scope

- Optimization of tuple-return ABI generally. Fix the bug, do not redesign.
- Migrating stdlib code to a different return convention (Result<T, E>, Box<>, etc.). Sounio's tuple-return is a language primitive; the compiler must support it.
- Changes to `lean_single.sio` driver — bootstrap stays frozen.

---

## §5 — Halt conditions (PARA E REPORTA triggers)

- Phase A gdb watchpoint doesn't pin a single writer (multiple instructions write 0x0 to the slot across runs). Surface is broader than expected; needs reframing.
- Phase B patch fixes `canonical.sio` but breaks `lean_single` fixed-point. Patch is wrong direction.
- Any need for Python/JS/R in the diagnostic loop. Signal that the dispatch needs a different toolchain.

---

## §6 — Deliverables on close

1. `self-hosted/compiler/native_compile_driver.sio` — unified diff, smallest possible.
2. Pre/post `bin/souc` md5 + size, recorded in `validation/binary_diff.txt`.
3. Gate logs: `validation/lean_single_gate.log`, `validation/umbrella_gate.log` — both showing PASS.
4. `repro/canonical.sio` runtime output before/after — `validation/before.txt` and `validation/after.txt`.
5. `stdlib/random/distributions.sio` un-deprecated, `stdlib/random/lib.sio` guidance updated.
6. Single commit on a dedicated branch (`codex/r2-3-tuple-return-fix` or similar). NO mixing with other work.

---

## §7 — Acceptance

`./bin/souc compile <canonical.sio>` produces an ELF whose output is exactly:
```
1362.073482
1362.073482
```
(with newlines after each — currently `println(f64)` omits trailing newline; do NOT touch that in this dispatch).

And: park_miller continues to compile and pass its self-test (additive, no-regression check).
