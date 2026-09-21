<!-- docs:meta
topic_id: repo.docs.architecture.ws-c-pr1-payload-census
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.ws-c-pr1-payload-census
-->

# WS-C PR1 Payload Census

Date: 2026-08-16

Source refs: `origin/main` = `03416657fa3e10667867e8a00bbfa4e7ec44261d`; `origin/canon/madaros-v2-sota` = `97b525949765980406a4fefa7f533e9db89721e1`.

Scope: census only for `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh` and the `madaros_v2_e2*` / `madaros_v2_e3*` gate scripts on `origin/canon/madaros-v2-sota`. This follows shell-script references, Python verifier references, verifier `PROGRAMS` fixture expansions, recursive regression-gate calls, and the `self-hosted/enir/driver.sio` import closure required by each gate build. `$TMP_DIR` negative fixtures generated inside scripts are excluded because they are not repository payload.

Headline: the requested E1/E2/E3 gate stack needs **63 files absent from `origin/main`**: 14 gate scripts, 13 Python verifiers, 14 `self-hosted/enir` driver-closure files, and 22 `tools/eisa` oracle/fixture files. The 23rd frontier-only `tools/eisa` file, `tools/eisa/eisa_enir_c2_rump.eisa`, is referenced by the C2 gate, not by the requested E1/E2/E3 gate set.

## Common references

These paths are referenced by most or all gates through seed/build, protected diff scopes, or the frozen METRON corpus. They are present on `origin/main` unless noted; changed means present on both refs with different blob content.

| Path | Present on main | Notes |
| --- | --- | --- |
| `bin/souc-lean-single-x86_64` | yes (changed) | common seed/helper/corpus/protected scope |
| `scripts/dev/souc-build-lock.sh` | yes (changed) | common seed/helper/corpus/protected scope |
| `self-hosted/compiler/main.sio` | yes (changed) | common seed/helper/corpus/protected scope |
| `self-hosted/gpu` | yes (tree) | common seed/helper/corpus/protected scope |
| `self-hosted/ir` | yes (tree) | common seed/helper/corpus/protected scope |
| `self-hosted/native` | yes (tree) | common seed/helper/corpus/protected scope |
| `self-hosted/wasm` | yes (tree) | common seed/helper/corpus/protected scope |
| `stdlib/eisa` | yes (tree) | common seed/helper/corpus/protected scope |
| `stdlib/math/dd64.sio` | yes (same) | common seed/helper/corpus/protected scope |
| `stdlib/math/qd128.sio` | yes (changed) | common seed/helper/corpus/protected scope |
| `tools/eisa/eisa_evm_run.sio` | yes (changed) | common seed/helper/corpus/protected scope |
| `stdlib/runtime` | no | guard-only path named by several scripts; absent on both compared refs, so not an add-list item |

## Common ENIR driver closure

Every gate builds `self-hosted/enir/driver.sio`. Its `use enir::*` closure requires these repository files. All are present on the canon branch and absent from `origin/main`, so each applies to the per-gate transitive missing set.

| Path | Present on main |
| --- | --- |
| `self-hosted/enir/mod.sio` | no |
| `self-hosted/enir/driver.sio` | no |
| `self-hosted/enir/ir.sio` | no |
| `self-hosted/enir/parser.sio` | no |
| `self-hosted/enir/canonical.sio` | no |
| `self-hosted/enir/verify.sio` | no |
| `self-hosted/enir/hash.sio` | no |
| `self-hosted/enir/shadow_fixture.sio` | no |
| `self-hosted/enir/source_lower.sio` | no |
| `self-hosted/enir/interpreter.sio` | no |
| `self-hosted/enir/qd.sio` | no |
| `self-hosted/enir/mir.sio` | no |
| `self-hosted/enir/mir_cfg.sio` | no |
| `self-hosted/enir/mir_join.sio` | no |

## Per-gate transitive census

### E1 shadow

Gate script: `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
Regression chain followed: none.
Transitive files absent from `origin/main`: **16**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e1_enir_shadow_verify.py` | no | gate/verifier |

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E2A lowering

Gate script: `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
Regression chain followed: `E1 shadow`.
Transitive files absent from `origin/main`: **18**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e2_enir_lowering_verify.py` | no | gate/verifier |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E2B CFG

Gate script: `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
Regression chain followed: `E2A lowering`.
Transitive files absent from `origin/main`: **21**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py` | no | gate/verifier |
| `tools/eisa/eisa_enir_v1_oracle.sio` | no | oracle/fixture |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E2C fuel/blockargs

Gate script: `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
Regression chain followed: `E2B CFG`.
Transitive files absent from `origin/main`: **24**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py` | no | gate/verifier |
| `tools/eisa/eisa_enir_v1_loop_oracle.sio` | no | oracle/fixture |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `tools/eisa/eisa_enir_v1_oracle.sio`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E2D rump DD

Gate script: `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`
Regression chain followed: `E2C fuel/blockargs`.
Transitive files absent from `origin/main`: **27**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py` | no | gate/verifier |
| `tools/eisa/eisa_enir_v1_rump_dd.eisa` | no | oracle/fixture |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py`
- `tools/eisa/eisa_enir_v1_loop_oracle.sio`
- `tools/eisa/eisa_enir_v1_oracle.sio`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E2E qd128 arithmetic

Gate script: `scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh`
Regression chain followed: `E2D rump DD`.
Transitive files absent from `origin/main`: **35**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e2e_enir_qd128_verify.py` | no | gate/verifier |
| `tools/eisa/eisa_enir_v2_const_gate.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_add.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_sub.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_mul.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_div.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_sqrt.eisa` | no | oracle/fixture |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py`
- `tools/eisa/eisa_enir_v1_loop_oracle.sio`
- `tools/eisa/eisa_enir_v1_oracle.sio`
- `tools/eisa/eisa_enir_v1_rump_dd.eisa`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E2F rump qd

Gate script: `scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh`
Regression chain followed: `E2E qd128 arithmetic`.
Transitive files absent from `origin/main`: **38**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e2f_enir_rump_qd_verify.py` | no | gate/verifier |
| `tools/eisa/eisa_enir_v2_rump_qd.eisa` | no | oracle/fixture |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py`
- `scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e2e_enir_qd128_verify.py`
- `tools/eisa/eisa_enir_v1_loop_oracle.sio`
- `tools/eisa/eisa_enir_v1_oracle.sio`
- `tools/eisa/eisa_enir_v1_rump_dd.eisa`
- `tools/eisa/eisa_enir_v2_add.eisa`
- `tools/eisa/eisa_enir_v2_const_gate.eisa`
- `tools/eisa/eisa_enir_v2_div.eisa`
- `tools/eisa/eisa_enir_v2_mul.eisa`
- `tools/eisa/eisa_enir_v2_sqrt.eisa`
- `tools/eisa/eisa_enir_v2_sub.eisa`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E2G fuel/control/frail

Gate script: `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh`
Regression chain followed: `E2F rump qd`.
Transitive files absent from `origin/main`: **43**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_verify.py` | no | gate/verifier |
| `tools/eisa/eisa_enir_v2_fuel.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_loop.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_frail.eisa` | no | oracle/fixture |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py`
- `scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e2e_enir_qd128_verify.py`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_verify.py`
- `tools/eisa/eisa_enir_v1_loop_oracle.sio`
- `tools/eisa/eisa_enir_v1_oracle.sio`
- `tools/eisa/eisa_enir_v1_rump_dd.eisa`
- `tools/eisa/eisa_enir_v2_add.eisa`
- `tools/eisa/eisa_enir_v2_const_gate.eisa`
- `tools/eisa/eisa_enir_v2_div.eisa`
- `tools/eisa/eisa_enir_v2_mul.eisa`
- `tools/eisa/eisa_enir_v2_rump_qd.eisa`
- `tools/eisa/eisa_enir_v2_sqrt.eisa`
- `tools/eisa/eisa_enir_v2_sub.eisa`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E2H memory/move/poison

Gate script: `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_gate.sh`
Regression chain followed: `E2G fuel/control/frail`.
Transitive files absent from `origin/main`: **48**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py` | no | gate/verifier |
| `tools/eisa/eisa_enir_v2_mem.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_emov.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_mem_poison.eisa` | no | oracle/fixture |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py`
- `scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e2e_enir_qd128_verify.py`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_verify.py`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_verify.py`
- `tools/eisa/eisa_enir_v1_loop_oracle.sio`
- `tools/eisa/eisa_enir_v1_oracle.sio`
- `tools/eisa/eisa_enir_v1_rump_dd.eisa`
- `tools/eisa/eisa_enir_v2_add.eisa`
- `tools/eisa/eisa_enir_v2_const_gate.eisa`
- `tools/eisa/eisa_enir_v2_div.eisa`
- `tools/eisa/eisa_enir_v2_frail.eisa`
- `tools/eisa/eisa_enir_v2_fuel.eisa`
- `tools/eisa/eisa_enir_v2_loop.eisa`
- `tools/eisa/eisa_enir_v2_mul.eisa`
- `tools/eisa/eisa_enir_v2_rump_qd.eisa`
- `tools/eisa/eisa_enir_v2_sqrt.eisa`
- `tools/eisa/eisa_enir_v2_sub.eisa`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E3A MIR qd128

Gate script: `scripts/dev/madaros_v2_e3a_enir_mir_qd128_gate.sh`
Regression chain followed: `E2H memory/move/poison`.
Transitive files absent from `origin/main`: **50**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e3a_enir_mir_qd128_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e3a_enir_mir_qd128_verify.py` | no | gate/verifier |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py`
- `scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e2e_enir_qd128_verify.py`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_verify.py`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_verify.py`
- `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_gate.sh`
- `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py`
- `tools/eisa/eisa_enir_v1_loop_oracle.sio`
- `tools/eisa/eisa_enir_v1_oracle.sio`
- `tools/eisa/eisa_enir_v1_rump_dd.eisa`
- `tools/eisa/eisa_enir_v2_add.eisa`
- `tools/eisa/eisa_enir_v2_const_gate.eisa`
- `tools/eisa/eisa_enir_v2_div.eisa`
- `tools/eisa/eisa_enir_v2_emov.eisa`
- `tools/eisa/eisa_enir_v2_frail.eisa`
- `tools/eisa/eisa_enir_v2_fuel.eisa`
- `tools/eisa/eisa_enir_v2_loop.eisa`
- `tools/eisa/eisa_enir_v2_mem.eisa`
- `tools/eisa/eisa_enir_v2_mem_poison.eisa`
- `tools/eisa/eisa_enir_v2_mul.eisa`
- `tools/eisa/eisa_enir_v2_rump_qd.eisa`
- `tools/eisa/eisa_enir_v2_sqrt.eisa`
- `tools/eisa/eisa_enir_v2_sub.eisa`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E3B MIR memory

Gate script: `scripts/dev/madaros_v2_e3b_enir_mir_memory_gate.sh`
Regression chain followed: `E3A MIR qd128`.
Transitive files absent from `origin/main`: **52**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e3b_enir_mir_memory_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e3b_enir_mir_memory_verify.py` | no | gate/verifier |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py`
- `scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e2e_enir_qd128_verify.py`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_verify.py`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_verify.py`
- `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_gate.sh`
- `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py`
- `scripts/dev/madaros_v2_e3a_enir_mir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e3a_enir_mir_qd128_verify.py`
- `tools/eisa/eisa_enir_v1_loop_oracle.sio`
- `tools/eisa/eisa_enir_v1_oracle.sio`
- `tools/eisa/eisa_enir_v1_rump_dd.eisa`
- `tools/eisa/eisa_enir_v2_add.eisa`
- `tools/eisa/eisa_enir_v2_const_gate.eisa`
- `tools/eisa/eisa_enir_v2_div.eisa`
- `tools/eisa/eisa_enir_v2_emov.eisa`
- `tools/eisa/eisa_enir_v2_frail.eisa`
- `tools/eisa/eisa_enir_v2_fuel.eisa`
- `tools/eisa/eisa_enir_v2_loop.eisa`
- `tools/eisa/eisa_enir_v2_mem.eisa`
- `tools/eisa/eisa_enir_v2_mem_poison.eisa`
- `tools/eisa/eisa_enir_v2_mul.eisa`
- `tools/eisa/eisa_enir_v2_rump_qd.eisa`
- `tools/eisa/eisa_enir_v2_sqrt.eisa`
- `tools/eisa/eisa_enir_v2_sub.eisa`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E3C CFG memory SSA

Gate script: `scripts/dev/madaros_v2_e3c_cfg_memory_ssa_gate.sh`
Regression chain followed: `E3B MIR memory`.
Transitive files absent from `origin/main`: **56**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e3c_cfg_memory_ssa_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e3c_cfg_memory_ssa_verify.py` | no | gate/verifier |
| `tools/eisa/eisa_enir_v2_mem_phi_zero.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_mem_phi_once.eisa` | no | oracle/fixture |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py`
- `scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e2e_enir_qd128_verify.py`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_verify.py`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_verify.py`
- `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_gate.sh`
- `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py`
- `scripts/dev/madaros_v2_e3a_enir_mir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e3a_enir_mir_qd128_verify.py`
- `scripts/dev/madaros_v2_e3b_enir_mir_memory_gate.sh`
- `scripts/dev/madaros_v2_e3b_enir_mir_memory_verify.py`
- `tools/eisa/eisa_enir_v1_loop_oracle.sio`
- `tools/eisa/eisa_enir_v1_oracle.sio`
- `tools/eisa/eisa_enir_v1_rump_dd.eisa`
- `tools/eisa/eisa_enir_v2_add.eisa`
- `tools/eisa/eisa_enir_v2_const_gate.eisa`
- `tools/eisa/eisa_enir_v2_div.eisa`
- `tools/eisa/eisa_enir_v2_emov.eisa`
- `tools/eisa/eisa_enir_v2_frail.eisa`
- `tools/eisa/eisa_enir_v2_fuel.eisa`
- `tools/eisa/eisa_enir_v2_loop.eisa`
- `tools/eisa/eisa_enir_v2_mem.eisa`
- `tools/eisa/eisa_enir_v2_mem_poison.eisa`
- `tools/eisa/eisa_enir_v2_mul.eisa`
- `tools/eisa/eisa_enir_v2_rump_qd.eisa`
- `tools/eisa/eisa_enir_v2_sqrt.eisa`
- `tools/eisa/eisa_enir_v2_sub.eisa`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E3D multipred scalar/memory SSA

Gate script: `scripts/dev/madaros_v2_e3d_multipred_scalar_memory_ssa_gate.sh`
Regression chain followed: `E3C CFG memory SSA`.
Transitive files absent from `origin/main`: **60**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e3d_multipred_scalar_memory_ssa_gate.sh` | no | gate/verifier |
| `scripts/dev/madaros_v2_e3d_multipred_scalar_memory_ssa_verify.py` | no | gate/verifier |
| `tools/eisa/eisa_enir_v2_join_then.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_join_else.eisa` | no | oracle/fixture |

Inherited missing paths from regression chain:
- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py`
- `scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e2e_enir_qd128_verify.py`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_verify.py`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_verify.py`
- `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_gate.sh`
- `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py`
- `scripts/dev/madaros_v2_e3a_enir_mir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e3a_enir_mir_qd128_verify.py`
- `scripts/dev/madaros_v2_e3b_enir_mir_memory_gate.sh`
- `scripts/dev/madaros_v2_e3b_enir_mir_memory_verify.py`
- `scripts/dev/madaros_v2_e3c_cfg_memory_ssa_gate.sh`
- `scripts/dev/madaros_v2_e3c_cfg_memory_ssa_verify.py`
- `tools/eisa/eisa_enir_v1_loop_oracle.sio`
- `tools/eisa/eisa_enir_v1_oracle.sio`
- `tools/eisa/eisa_enir_v1_rump_dd.eisa`
- `tools/eisa/eisa_enir_v2_add.eisa`
- `tools/eisa/eisa_enir_v2_const_gate.eisa`
- `tools/eisa/eisa_enir_v2_div.eisa`
- `tools/eisa/eisa_enir_v2_emov.eisa`
- `tools/eisa/eisa_enir_v2_frail.eisa`
- `tools/eisa/eisa_enir_v2_fuel.eisa`
- `tools/eisa/eisa_enir_v2_loop.eisa`
- `tools/eisa/eisa_enir_v2_mem.eisa`
- `tools/eisa/eisa_enir_v2_mem_phi_once.eisa`
- `tools/eisa/eisa_enir_v2_mem_phi_zero.eisa`
- `tools/eisa/eisa_enir_v2_mem_poison.eisa`
- `tools/eisa/eisa_enir_v2_mul.eisa`
- `tools/eisa/eisa_enir_v2_rump_qd.eisa`
- `tools/eisa/eisa_enir_v2_sqrt.eisa`
- `tools/eisa/eisa_enir_v2_sub.eisa`

Common missing driver closure applies: `self-hosted/enir/*` listed above.

### E3E equal value distinct event

Gate script: `scripts/dev/madaros_v2_e3e_equal_value_distinct_event_gate.sh`
Regression chain followed: none.
Transitive files absent from `origin/main`: **17**.

| Direct gate-specific path | Present on main | Notes |
| --- | --- | --- |
| `scripts/dev/madaros_v2_e3e_equal_value_distinct_event_gate.sh` | no | gate/verifier |
| `tools/eisa/eisa_enir_v2_equal_then.eisa` | no | oracle/fixture |
| `tools/eisa/eisa_enir_v2_equal_else.eisa` | no | oracle/fixture |

Common missing driver closure applies: `self-hosted/enir/*` listed above.

## Consolidated add-list

These are the exact repository files referenced by the requested gate stack, present on `origin/canon/madaros-v2-sota`, and absent from `origin/main`.

- `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh`
- `scripts/dev/madaros_v2_e1_enir_shadow_verify.py`
- `scripts/dev/madaros_v2_e2_enir_lowering_gate.sh`
- `scripts/dev/madaros_v2_e2_enir_lowering_verify.py`
- `scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh`
- `scripts/dev/madaros_v2_e2b_enir_cfg_verify.py`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh`
- `scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh`
- `scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py`
- `scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e2e_enir_qd128_verify.py`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh`
- `scripts/dev/madaros_v2_e2f_enir_rump_qd_verify.py`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh`
- `scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_verify.py`
- `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_gate.sh`
- `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py`
- `scripts/dev/madaros_v2_e3a_enir_mir_qd128_gate.sh`
- `scripts/dev/madaros_v2_e3a_enir_mir_qd128_verify.py`
- `scripts/dev/madaros_v2_e3b_enir_mir_memory_gate.sh`
- `scripts/dev/madaros_v2_e3b_enir_mir_memory_verify.py`
- `scripts/dev/madaros_v2_e3c_cfg_memory_ssa_gate.sh`
- `scripts/dev/madaros_v2_e3c_cfg_memory_ssa_verify.py`
- `scripts/dev/madaros_v2_e3d_multipred_scalar_memory_ssa_gate.sh`
- `scripts/dev/madaros_v2_e3d_multipred_scalar_memory_ssa_verify.py`
- `scripts/dev/madaros_v2_e3e_equal_value_distinct_event_gate.sh`
- `self-hosted/enir/canonical.sio`
- `self-hosted/enir/driver.sio`
- `self-hosted/enir/hash.sio`
- `self-hosted/enir/interpreter.sio`
- `self-hosted/enir/ir.sio`
- `self-hosted/enir/mir.sio`
- `self-hosted/enir/mir_cfg.sio`
- `self-hosted/enir/mir_join.sio`
- `self-hosted/enir/mod.sio`
- `self-hosted/enir/parser.sio`
- `self-hosted/enir/qd.sio`
- `self-hosted/enir/shadow_fixture.sio`
- `self-hosted/enir/source_lower.sio`
- `self-hosted/enir/verify.sio`
- `tools/eisa/eisa_enir_v1_loop_oracle.sio`
- `tools/eisa/eisa_enir_v1_oracle.sio`
- `tools/eisa/eisa_enir_v1_rump_dd.eisa`
- `tools/eisa/eisa_enir_v2_add.eisa`
- `tools/eisa/eisa_enir_v2_const_gate.eisa`
- `tools/eisa/eisa_enir_v2_div.eisa`
- `tools/eisa/eisa_enir_v2_emov.eisa`
- `tools/eisa/eisa_enir_v2_equal_else.eisa`
- `tools/eisa/eisa_enir_v2_equal_then.eisa`
- `tools/eisa/eisa_enir_v2_frail.eisa`
- `tools/eisa/eisa_enir_v2_fuel.eisa`
- `tools/eisa/eisa_enir_v2_join_else.eisa`
- `tools/eisa/eisa_enir_v2_join_then.eisa`
- `tools/eisa/eisa_enir_v2_loop.eisa`
- `tools/eisa/eisa_enir_v2_mem.eisa`
- `tools/eisa/eisa_enir_v2_mem_phi_once.eisa`
- `tools/eisa/eisa_enir_v2_mem_phi_zero.eisa`
- `tools/eisa/eisa_enir_v2_mem_poison.eisa`
- `tools/eisa/eisa_enir_v2_mul.eisa`
- `tools/eisa/eisa_enir_v2_rump_qd.eisa`
- `tools/eisa/eisa_enir_v2_sqrt.eisa`
- `tools/eisa/eisa_enir_v2_sub.eisa`

## Present-but-changed watchlist

These referenced files already exist on `origin/main` but differ from the canon branch. They are not add-list entries, but a PR that expects canon behavior may need to account for the content delta separately.

- `bin/souc-lean-single-x86_64`
- `scripts/dev/souc-build-lock.sh`
- `self-hosted/compiler/main.sio`
- `stdlib/math/qd128.sio`
- `tools/eisa/eisa_evm_run.sio`

## Explicitly not in this add-list

- `tools/eisa/eisa_enir_c2_rump.eisa`: absent from `origin/main` and present on the canon branch, but referenced by `scripts/dev/madaros_v2_c2_v0_gate.sh`, not by the requested E1/E2/E3 gate set.
- `$TMP_DIR/*.eisa` negative fixtures: generated by the gate scripts at runtime and not repository payload.
