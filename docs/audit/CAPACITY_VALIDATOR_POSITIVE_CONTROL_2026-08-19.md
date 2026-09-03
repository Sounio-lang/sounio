<!-- docs:meta
topic_id: repo.docs.audit.capacity-validator-positive-control-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5 (assumed existing PR #1947; Slurm job 10358)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.capacity-validator-positive-control-2026-08-19
-->

# IrCapacities validator — positive control (PR #1947)

**Filed:** 2026-08-19 · **Lane:** grok-cli5 on existing PR #1947 · **Status:** validator alive, measured on Slurm

## Choice (default vs invariant 2)

**The invariant was fixed. The default was not raised.**

Production keeps `dce_liveness_slots: 8192` and `max_instructions: IR_MAX_INSTRS = 16384`. Main chose that on purpose in `042c29be53`: `dce_run_impl` refuses any function above `DCE_MAX_INSTRS` rather than analysing a prefix of it. A truncated liveness analysis is a wrong analysis. Raising the default would contradict that choice. The runtime object therefore requires only `dce_liveness_slots > 0`. The coupling "IR_MAX_INSTRS cannot rise without a matching DCE refuse" is a policy invariant pinned by `scripts/ci/irfunction_instr_capacity_coherence_gate.sh`.

`ir_test_30_capacities_invariants` is defined once.

## What was required

Issuecomment-5340499111 asked for one compile with a deliberately invalid configuration as a positive control, so the refusal is shown to fire, plus a valid baseline. A validator never seen to refuse is not a validator. The previous receipt on this PR used inlined probes on the pod after six Slurm jobs failed because `/workspace` is not mounted on the node.

This receipt uses `srun` stdin-tarball (`/tmp/cap1947_slurm.sh`), the same transport as `scripts/dev/souc-build-remote.sh`. The node never sees `/workspace`.

## Method (Slurm job 10358)

Host `gpuorangefs-r770-proxmox`, 32 CPUs, partition `all`. Payload: `self-hosted`, `stdlib`, `scripts`, seed ELFs, `bin/madaros`, and the three probes under `docs/audit/repro/`.

1. `bash scripts/ci/build_modular_madaros.sh` on the node (not wrapped in `souc-build-lock.sh`).
2. **VALID compile through the arena:** new Madaros compiles `docs/audit/repro/ir_cap_hello.sio`. `ir_instr_arena_init` calls `ir_capacities_validate` on the production default. If the default still contradicted the runtime invariants, this compile would abort.
3. **VALID probe:** same Madaros compiles and runs `docs/audit/repro/ir_capacities_valid_probe.sio` (replica of the three comparisons on the production numbers).
4. **INVALID probe:** same Madaros compiles `docs/audit/repro/ir_capacities_invalid_probe.sio` (same replica, then `region_slots = max_functions`) and runs it. The run must abort. No PASS line.

## Result (measured 2026-08-19T14:00Z)

```
REMOTE: host=gpuorangefs-r770-proxmox nproc=32 unpacked=72M
REMOTE: madaros_build rc=0 elapsed=226s
REMOTE: elf bytes=100550866

VALID_HELLO_COMPILE rc=0
CAP_HELLO_OK
VALID_HELLO_RUN rc=0

VALID_PROBE_COMPILE rc=0
PASS default capacities validated
VALID_PROBE_RUN rc=0

INVALID_PROBE_COMPILE rc=0
INVALID_STDOUT: (empty)
INVALID_PROBE_RUN rc=132
```

- Default configuration does **not** panic on the compile path. Hello lowered 3 functions and ran.
- Invalid configuration is **refused**. Compile of the bad probe succeeds (the source is well-formed); the run aborts with rc=132 (`128+SIGILL`) and prints neither PASS nor FAIL. The panic path on this Madaros build traps rather than `exit(1)`. That is still a refusal, and it is not a silent clamp.

## Reproduce

```bash
# From the #1947 worktree. Do not point srun at /workspace paths.
bash /tmp/cap1947_slurm.sh
```

Or locally, after a Madaros build from this branch:

```bash
env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
  MADAROS_RAW_BIN=artifacts/self-hosted/madaros \
  bin/madaros compile docs/audit/repro/ir_cap_hello.sio -o /tmp/ir_cap_hello.elf
# expect compile rc=0; /tmp/ir_cap_hello.elf prints CAP_HELLO_OK

bin/madaros compile docs/audit/repro/ir_capacities_valid_probe.sio -o /tmp/cap_valid.elf
/tmp/cap_valid.elf
# expect rc=0, PASS default capacities validated

bin/madaros compile docs/audit/repro/ir_capacities_invalid_probe.sio -o /tmp/cap_invalid.elf
/tmp/cap_invalid.elf
# expect rc!=0, no PASS line
```

## Related

- `self-hosted/ir/ir.sio` — `ir_capacities_validate` (runtime inv 2 is `dce_liveness_slots > 0`)
- `self-hosted/ir/dce.sio` — refuse-function + `DCE_REFUSAL_COUNT`
- `self-hosted/ir/serialize.sio` — `SOIR_DESER_REFUSAL_COUNT`
- `scripts/ci/irfunction_instr_capacity_coherence_gate.sh` — binds serialize / IrModule literals to `IR_MAX_FUNCS`
- `docs/audit/SOIR_CAPACITY_LOCKSTEP_CENSUS_2026-08-19.md` — 103 sites counted; capacity is not unified
