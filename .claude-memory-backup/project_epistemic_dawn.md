---
name: Operation Epistemic Dawn
description: Native epistemic computing pipeline — struct sret fix, PBPK demo, 20KB ELF, 6/6 PASS
type: project
---

## Operation Epistemic Dawn (2026-03-15)

Native epistemic computing pipeline completed. Three critical lean driver bugs fixed,
then flagship PBPK demo compiled to bare-metal ELF.

### What was fixed
1. **ExprReturn sret copy** — return statements in struct-returning functions skipped the sret
   copy, returning zeros. Fixed by checking `ctx.cur_sret` in ExprReturn handler.
2. **Struct assignment copy** — `s = fn_returning_struct()` stored only the address pointer
   instead of copying struct data. Fixed with rep movsq in StmtAssign for struct types.
3. **Field assignment** — `s.field = value` was silently ignored (no ExprFieldAccess handler
   in StmtAssign). Added field offset computation + store.

### Demo: epistemic_pbpk_native.sio
- 477 lines, 5 structs (up to 9 fields), 16 functions
- Rapamycin 3-compartment PBPK + GUM uncertainty propagation
- 1,920 RK4 integration steps across 4 perturbations
- **20,946 byte ELF**, runs in **3ms**, 6/6 validation tests PASS
- Gate: `bash scripts/epistemic_pbpk_gate.sh` — 7/7 PASS

### Key finding: `step` is a reserved word
The self-hosted parser treats `step` as a keyword. Functions named `step` cause parse_error.
Use `euler_step`, `rk4_advance`, etc. instead.

**Why:** First language to compile uncertainty propagation to native x86-64 code.
**How to apply:** Use `scripts/epistemic_pbpk_gate.sh` to validate. The demo is the
reference for struct-heavy native compilation patterns.
