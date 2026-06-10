<!-- docs:meta
topic_id: repo.docs.self-hosting-punchlist
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.self-hosting-punchlist
-->

# Madáres self-hosting punch-list (2026-06-05)

Goal: make `--native-v2-compile` handle the **whole language**, so Madáres can compile
real programs (and eventually itself) to native ELF, and we retire `mini_native`.

## The wall, measured (corpus = `examples/ + tests/run-pass/`, 1378 files)

Each program run through `souc --native-v2-compile <f> -o /tmp/x.elf`, classified by exit/output:

| Layer | Count | % | Owner file(s) |
|---|---|---|---|
| `parse_failed`     | 402 | 29% | `self-hosted/parser/*.sio` |
| `ir_bodies_failed` | 345 | 25% | `self-hosted/ir/lower.sio` (semantic lowering, NOT missing expr-kinds) |
| `CRASH` (139/124)  | 269 | 20% | `self-hosted/native/codegen_x86_linux.sio` (+ lower) |
| `backend_fail`     | 180 | 13% | `self-hosted/native/codegen_x86_linux.sio` |
| `ELF_OK` ✅         | 178 | 13% | — |

**Partition rule: one agent owns one file/layer end-to-end.** Within a layer most fixes
touch the SAME file; two agents on `lower.sio` (or `codegen_x86_linux.sio`) collide and
re-create the branch sprawl we just cleaned. So we split by file, not by item.

## Shared acceptance metric (run before & after every change)

```bash
cd /workspace/sounio-nv2-consolidate
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
SOUC=<your-built-souc.elf>     # built via: scripts/dev/souc-build-lock.sh ./bin/souc self-hosted/compiler/main.sio <out.elf>
# category census:
git ls-files '*.sio' | grep -E '^(examples|tests/run-pass)/' | while read -r f; do
  rm -f /tmp/x.elf
  out=$(timeout 25 "$SOUC" --native-v2-compile "$f" -o /tmp/x.elf 2>&1); rc=$?
  if [ "$rc" -ge 128 ] || [ "$rc" = 124 ]; then echo CRASH
  elif [ -f /tmp/x.elf ]; then echo ELF_OK
  elif echo "$out" | grep -qi "parse_failed\|parse error"; then echo parse_failed
  elif echo "$out" | grep -qi "ir_bodies_failed"; then echo ir_bodies_failed
  elif echo "$out" | grep -qi "backend\|FAIL to_file\|unsupported"; then echo backend_fail
  else echo other; fi
done | sort | uniq -c | sort -rn
```
A change is good iff its layer's count **drops**, `ELF_OK` rises, and **no other count rises**
(no regressions). Also always: `tests/native_v2_capgate/run.sh <souc>` must stay 31/31.

## Rules of engagement (from `.codex/AGENT_HANDOFF.md`)
- Build is **serialized** through `scripts/dev/souc-build-lock.sh` (~35GB RAM). One driver owns the build cadence (Claude). Agents request a rebuild via the driver or take the lock.
- Before editing a shared file, append a lock entry to `artifacts/omega/agent_handoff.log.md`; release with status.
- Check `.claude/check_sio_integration_window.v1.json` before touching `check.sio`.
- Small reversible commits. One branch per agent, off `feat/exact-orc-machinery`.
- Diagnostic technique (how to find the specific gap in your layer): instrument the
  failing fallback/error site to `print` the offending kind/name, rebuild once, run the
  corpus, tally `sort | uniq -c | sort -rn`. (Revert the probe before committing.)

---

## CARD A — Parser layer  (Codex)
**Own:** `self-hosted/parser/*.sio` only.  **Target:** `parse_failed` 402 → down.
The MODULAR parser rejects programs `mini_native` accepts. Known gaps already seen:
uninitialised `var x: T` (no initializer), const-dimensioned array fields/locals `[T; N]`
where N is a const, sci-notation floats, some keyword/item forms. Find each by reducing a
`parse_failed` file to the minimal rejected snippet (`souc --check <min>` → "parse error at
line L"), fix the parser, re-census. Do NOT touch lower/codegen.
**Accept:** `parse_failed` drops, nothing else rises, capgate 31/31.

## CARD B — Backend layer  (Codex)
**Own:** `self-hosted/native/codegen_x86_linux.sio` only.  **Target:** `backend_fail` 180
→ down, and triage the 269 `CRASH` (many are backend codegen faults, e.g. arrays-of-structs,
that crash mid-emit). For each: minimal repro, find the emit site, fix or fail-closed
(return an error instead of SIGSEGV). Known gaps: arrays-of-structs element store/load,
large struct returns, possibly f64-through-aggregate. Do NOT touch parser/lower.
**Accept:** `backend_fail` + `CRASH` drop, `ELF_OK` rises, capgate 31/31.

## CARD C — Lowerer layer  (Claude/Demetrios — has the build + integration)
**Own:** `self-hosted/ir/lower.sio`.  **Target:** `ir_bodies_failed` 345 → down.
NOTE: these are NOT unhandled expr-kinds (the `lower_expr_ref` else-fallback never fires).
They are semantic — the lowerer handles the syntax but `report_error()`s during lowering.
Prime suspects: unresolved function/method/field references (native-v2 is **single-module**
— imported symbols don't resolve) and specific-handler errors. Refine the diagnostic by
instrumenting the ~15 `report_error()` call sites (give each a code) OR reduce
`ir_bodies_failed` files. The big structural lever here is likely **multi-module symbol
resolution** in the native-v2 front-half (`bridge_lower_single_module_box`).
**Accept:** `ir_bodies_failed` drops, `ELF_OK` rises, capgate 31/31.

---
The four layers are independent and measured by the SAME census, so progress composes:
sum of `ELF_OK` gains across agents = programs that newly compile. When `ELF_OK` covers the
language `main.sio` uses, native-v2 can compile `main.sio` → self-host → retire mini.
