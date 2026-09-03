<!-- docs:meta
topic_id: repo.docs.audit.madaros-f64-bitcast-sitofp-boundary-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-f64-bitcast-sitofp-boundary-2026-08-17
-->

# Madaros f64 bitcast/sitofp boundary — Knowledge.confidence kind-split (F2)

**Date:** 2026-08-17  
**Lane:** grok-cli4 / f64-bitcast-boundary  
**Engine:** default Madaros v0.80.0 (`./bin/souc`); lean_single is the healthy reference  
**Status:** mechanism **NAMED and BOUNDARY-TABLED**. Compiler fix not landed here
(fleet lock — no self-hosted rebuild). Detection is fail-closed via the gate below
and the earlier `epistemic_fabrication_detect` surface.

---

## 0. Name

| | |
|---|---|
| **Mechanism ID** | `KCONF-BITCAST-SITOFP` |
| **Short name** | Knowledge.confidence f64-payload / i64-layout kind-split |
| **Family** | f64 **bitcast → sitofp** (same family as historical D1 `f64 as i64` bitcast) |
| **Not this** | char\* `println` SIGSEGV (fable-1 / `MADAROS_PRINTLN_BOOL_SCALARKIND_SEGV`) |

**One sentence.** Constructors store an f64 probability in `Knowledge.confidence`,
but the IR layout marks that field as integer (`is_float=3`); `print_f64` of the
raw load can look correct, while any f64 arithmetic or compare emits
`sitofp`/`cvtsi2sd` on the IEEE bit pattern and yields magnitude ~4×10¹⁸.

---

## 1. Why F2 is not the char\* printer

| Symptom | char\* family (fable-1) | F2 (this mechanism) |
|---|---|---|
| Runtime shape | **SIGSEGV** (rc=139), strlen on small int as pointer | process lives; prints a tidy decimal |
| Print path | `println` → kind 0 → `print` (char\*) | **`print_f64`** of a live huge float |
| Measured F2 line | — | `AUC confidence: 4604219396932172800.000000` |
| Decode | — | integer `4604…` = IEEE bits of **≈0.671038** after **sitofp** |

A char\* misroute on this bit pattern would fault, not print `….000000`. F2 is
value corruption in the float domain after an integer reinterpretation.

---

## 2. Source locus (do not rediscover)

`self-hosted/ir/lower.sio` — `ir_register_knowledge_layout`:

```text
// confidence is i64 (is_float=3 integer marker). Was wrongly is_float=1 which
// polluted field_is_float_by_name_simple("confidence") for EVERY struct field
// named confidence (MiniEp, Epistemic, …). … See MADAROS_FIELD_IF_I64 / #1496.
fields[2] = … name "confidence", is_float: 3 …
```

Same file — Knowledge constructors still **write an f64 epsilon** into that slot:

```text
// Knowledge(12.5, e=0.65, prov="src") -> .confidence 0.65
// field 2: confidence = epsilon   (conf_reg marked float at store)
```

Cast lowering (`lower_cast_expr_ref`): when the source is not classified float,
`x as i64` is a **no-op copy** (bitcast of bits); `k as f64` on a non-float source
emits `IrIntToFloat` (sitofp). Binary ops that see an integer-kind operand and an
f64 operand take the same sitofp edge before `mulsd`.

**Why confidence was marked i64.** #1496: a name-global `is_float=1` on
`"confidence"` made *every* struct field of that name take the float path, so
i64 0–1000 confidence (`MiniEp`, stdlib `Epistemic`) broke (`conf >= 800` → 0).
The layout flip fixed that class and opened this one for **Knowledge**, whose
surface epsilon is a probability in (0,1].

**Family link to D1.** D1 (`MADAROS_IMPORTED_MODULE_F64_CAST_BITCAST_2026-07-14`):
`f64 as i64` without `cvttsd2si` → `print_int` shows the IEEE payload (GUM k95
stuck at 1.960). Here the dual: f64 bits live in an i64-kind slot → sitofp
before f64 use → `print_f64` shows the integer magnitude. Same kind-confusion
family; different edge of the cast diamond.

---

## 3. Trigger boundary (measured, this worktree, prebuilt Madaros v0.80.0)

Witnesses:

- `tests/run-pass/f64_bitcast_boundary_controls.sio` — R01–R13
- `tests/run-pass/f64_bitcast_boundary_user_conf.sio` — R23–R24
- `tests/run-pass/f64_bitcast_boundary_knowledge_conf.sio` — R20–R22, R25
- Live science: `stdlib/darwin_pbpk/epistemic_pbpk28.sio` TEST 6

| # | Program shape | Madaros | lean_single | Note |
|---|---|---|---|---|
| R01 | `let x: f64 = 4.172; x as i64` | **OK** → 4 | OK | main-local trunc |
| R02 | `fn f(x: f64)->i64 { x as i64 }` | **OK** → 4 | OK | param cast (D1 closed same-module) |
| R03 | `print_f64(0.671038)` | **OK** | OK | plain print |
| R04 | single-field `{ conf: f64 }` return + print | **OK** | OK | not Knowledge |
| R05 | two-field ConfBox | **OK** | OK | prior F2 mini-probe |
| R06 | EpResult-like plain f64 fields | **OK** | OK | multi-field alone insufficient |
| R07 | weighted plain f64 sum | **OK** → 0.66 | OK | no Knowledge |
| R08 | local `f64→i64→f64` roundtrip | **OK** → 4.0 | OK | true trunc+sitofp |
| R09 | param roundtrip | **OK** | OK | |
| R10 | `print_int(42)` after `print_f64(3.5)` | **OK** | OK | garble not on this micro |
| R11 | `(-3.9) as i64` | **OK** → -3 | OK | |
| R12 | field f64 then `as i64` | **OK** | OK | |
| R13 | `let b = a; b as i64` (f64 copy) | **OK** | OK | fable-1 copy gap is println kind |
| R14–R17 | imported plain f64 cast/return/struct | **OK** | OK | D1 residual not required for F2 |
| R20 | `Knowledge(…, e=0.671038).confidence` then `* 0.5` | **BITCAST_SITOFP** | conf often **1.0** (ctor) | Madaros: print OK, arith sitofp; lean e= mapping differs — not the cross-engine oracle |
| R21 | struct-lit Knowledge + weighted sum | **BITCAST_SITOFP** | **OK** | prod/wsum ~4.604e18 |
| R22 | `k.epsilon` vs `k.confidence` | eps=**0.0**, conf print OK | — | separate epsilon gap |
| R23 | user `struct { confidence: f64 }` arith | **OK** | OK | positive control |
| R24 | user `struct { confidence: i64 }` ge/sub | **OK** | OK | #1496 twin — must stay int |
| R25 | `[Knowledge[f64];N]` weighted conf (ep28 shape) | **BITCAST_SITOFP** ~4.604e18 | **OK** 0.66 | **minimal F2** |
| F2 | `epistemic_pbpk28` TEST 6 `auc_blood_conf` | **FABRICATION** ~4.604e18 | OK ~0.67 | science surface |

**Trigger (necessary and sufficient for the F2 shape):**

1. Store an f64 probability into `Knowledge.confidence` / epsilon slot, and  
2. Use that field in **f64 arithmetic or compare** (not merely `print_f64` of the raw load).

**Non-triggers (controls):** plain f64 locals/params/structs; user structs with
`confidence: f64` or `confidence: i64` under typed layout; D1-style `as i64`
truncation on ordinary f64 (closed on this prebuilt for the rows above).

---

## 4. Decode receipt (R21 / R25)

```text
python3 -c "import struct; x=0.671038; print(struct.unpack('<Q', struct.pack('<d', x))[0])"
# 4604219392518779302
# sitofp → 4604219392518779392.0  (matches R21_prod under Madaros)
# weighted sitofp(0.80)*0.5 + sitofp(0.60)*0.3 + sitofp(0.40)*0.2
#   → 4603939827068311040.0       (matches R25)
```

ep28 TEST 6 prints `4604219396932172800.000000` — same class (weighted sitofp of
the live prior confidences), not a print-only lie: the range check
`conf > 0.20 && conf < 0.90` fails because the **register value** is ~4e18.

---

## 5. Fix direction (compiler owner — not this lane)

Scoped to Knowledge layout / field-kind resolution; must preserve R23 and R24.

1. **Preferred:** Knowledge.confidence is f64 at the IR layout (`is_float=1`)
   **and** field kind is resolved from the **typed base**
   (`field_is_float_for_base_ref`), never from the bare name `"confidence"`.
   That is what #1496 actually required — not a permanent i64 lie on Knowledge.
2. **Do not** flip `field_is_float_by_name_simple("confidence")` alone — reopens
   #1496 on MiniI / Epistemic i64 confidence.
3. After fix: R20/R21/R25 → `*_OK`, ep28 TEST 6 probability-scale, fabrication
   gate F2 arm takes the healthy branch. Rebuild Madaros **from source** before
   claiming closure (prebuilt `bin/souc` does not track `self-hosted/` edits).

---

## 6. Gate

```bash
export MADAROS_STACK_KB=524288
bash scripts/ci/f64_bitcast_sitofp_boundary_gate.sh
```

Assertions:

1. CONTROLS R01–R13 all OK under Madaros  
2. USER R23+R24 OK (f64 and i64 confidence)  
3. KCONF either all OK (fixed) or BITCAST_SITOFP **and** rc≠0 (live defect, fail-closed)  
4. lean_single KCONF all OK with R25 ~0.66 (Madaros-only diagnosis)

Related: `scripts/ci/epistemic_fabrication_detect_gate.sh` (F1 zero-var + F2 ep28).

---

## 7. Non-goals

- No `self-hosted/` edit in this delivery (CPU / fleet lock).  
- Do not re-attribute F2 to char\* / `lower_array` status lines.  
- Do not treat D1 imported-module cast as still open on the R02/R14 micro; F2
  does not need D1 to fire once Knowledge.confidence is in the expression.
- R22 `.epsilon == 0.0` is noted, not fixed here.

---

## 8. AI disclosure

Boundary table, witnesses, and gate by AI agent (grok-cli4) under human direction,
measured on the worktree prebuilt Madaros and lean_single. GAIDeT-ICMJE 2025.
No fabricated green: KCONF is fail-closed while the defect is live.
