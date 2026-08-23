<!-- docs:meta
topic_id: repo.docs.audit.gate-instrument-census-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.gate-instrument-census-2026-08-18
-->

# Gate instrument census — three patterns, 2026-08-18

**Why.** The E230 ceiling gate was wrong four ways in a row and stayed
green: compile without checking an ELF existed, wrong engine, rc read
through a pipe, compiler that could not construct structs. Each passed
as a measurement. The question is how many *other* `scripts/ci` gates
share the first three of those.

**Scope.** `scripts/ci/*.sh`, 537 files. One pattern at a time. Not a
rewrite of 468 gates.

**The three numbers** (remeasured on `origin/main` `ff5b9295ee`,
`scripts/ci/*_gate.sh` = 474, compile-invoking = 112; workflow-reachable
= **85** from `CI_GATE_WORKFLOW_REACHABILITY_CENSUS_2026-08-18.tsv` —
not 91):

| pattern | raw | of which workflow-reachable |
|---|---:|---:|
| 1. compile, then talk about the artefact without `\x7fELF` | **93** | **19** |
| 2. rc read through a pipe | **0** | **0** |
| 3. never record which engine compiled | **68** | **12** |

Raw 93 and 68 are over twenty → helper, not a hand sweep. The order that
matters is the reachable column: 19 gates on `ci.yml` / prebuilt-refresh
can print a green that did not compile an ELF. The rest are debt. Fix
the 19 by calling `require_elf`, not by rewriting 93.

**Answer in one line.** Pattern 2 is empty. Patterns 1 and 3 are design.
The helper is `scripts/lib/gate_assert.sh`. Its selftest is
`scripts/ci/gate_assert_instrument_selftest.sh`.

## Pattern 4 — skip as green (cursor-2 `measured=no`)

A fourth lie, from the leftover budget: gates with published numbers
that `exit 0` after `SKIPPED (no ptxas)` / `SKIPPED (libcuda not
found)`. That is skip-vacuous. `require_tool` prints `SKIPPED` and
exits **77**, with `measured=no`. Never 0.

The helper's selftest runs two fixture gates: a skip must not be 0,
and a compile-to-void must refuse. If either fixture is accepted, the
selftest fails. One real consumer: `handle_table_ceiling_gate.sh`.
Mass conversion of the 19 reachable P1 gates is a separate decision.

## Pattern 2 — rc through a pipe (done first)

A status assigned from a pipeline is the last command's status
(grep/tee/awk), so the gate is always green.

| class | n | notes |
|---|---:|---|
| `rc=$(tail \| grep \| awk)` / status from a pipe | **0** | the E230 defect was local |
| `rc=$(run_probe …)` | 7 | function, not a pipe |
| `cmd \| tee \|\| RC=$?` | 2 | pbpk28 + frontend parity; both have `pipefail` |
| `if ! souc … \| grep -v warning` | 2 | metal + ptx; they then `[[ -s $ELF ]]` |
| `PIPESTATUS[0]` after tee | 2 | e_series, ade_wildgen — the correct form |
| `\| tee` without `pipefail` | **0** | 25 files tee; all already `pipefail` |

No cleanup. No helper for this pattern. Do not reopen it as a sweep.

## Pattern 1 — compile, then talk about the artefact without `\x7fELF`

Compile-invoking `*_gate.sh`: **110**. Never check `\x7fELF`: **91**.
Of those, **19** are workflow-reachable (false greens on a PR):

| depth | gate | workflow |
|---|---|---|
| 1 | `canonical_compiler_gate.sh` | ci.yml |
| 1 | `epistemic_egraph_rewrite_gate.sh` | ci.yml |
| 1 | `madaros_current_source_f64_lowering_gate.sh` | ci.yml |
| 1 | `madaros_dce_reachability_gate.sh` | ci.yml |
| 1 | `madaros_fixed_point_gate.sh` | ci.yml |
| 1 | `madaros_self_parse_gate.sh` | ci.yml |
| 1 | `ontology_cli_smoke_gate.sh` | ci.yml |
| 1 | `package_pbpk_gum_gate.sh` | ci.yml |
| 1 | `self_falsifying_compilation_line_r1_gate.sh` | ci.yml |
| 1 | `self_falsifying_compilation_line_r29_gate.sh` | ci.yml |
| 1 | `souc_v2_gate.sh` | ci.yml |
| 1 | `sounio_package_support_gate.sh` | ci.yml |
| 1 | `madaros_enum_gate.sh` | madaros-prebuilt-refresh.yml |
| 1 | `madaros_loop_gate.sh` | madaros-prebuilt-refresh.yml |
| 2 | `madaros_global_capacity_gate.sh` | via f64-lowering |
| 2 | `madaros_global_f64_scratch_gate.sh` | via f64-lowering |
| 2 | `madaros_imported_call_arity_13_gate.sh` | via f64-lowering |
| 2 | `madaros_imported_capacity_gate.sh` | via f64-lowering |
| 2 | `madaros_imported_deref_f64_array_gate.sh` | via f64-lowering / full |

A compile that exits 0 and writes a file named `-o` is non-empty;
`[[ -s ]]` would have passed that trap. Magic is the check that refuses
it. Do not patch 91 files. Adopt `require_elf` on the 19 first.

## Pattern 3 — never record which engine compiled

Of 110 compile-invoking gates, **67** never mention `--version`,
`compile_engine`, `lean_single`, or `Madaros v`. **12** of those are
workflow-reachable (11 of them also fail pattern 1).

`--version` is not enough. `bin/souc --version` prints Madaros while
`souc src.sio -o dest` routes to lean_single. Classify the **compile
log** (`classify_compile_log`). `souc_banner` in
`scripts/lib/souc_invoke.sh` is for choosing argv, not for recording
what a compile actually did.

## Helper (not 98 patches)

Added to `scripts/lib/gate_assert.sh` — the 2026-08-04 emptiness
library, same failure shape (empty or wrong artefact read as success):

| function | refuses |
|---|---|
| `require_elf path` | missing, empty, magic ≠ `7f454c46` |
| `classify_compile_log log` | prints `madaros` / `lean_single` / `unknown` from the log |
| `require_compile_engine log expected` | log names the wrong engine |
| `gate_capture_rc dest -- cmd…` | writes the child's rc to a file |
| `require_rc_file dest [expected]` | missing/empty/non-numeric rc file |

Selftest: `bash scripts/ci/gate_assert_instrument_selftest.sh` breaks
all three lies on purpose (text file as ELF, lean_single log claimed as
Madaros, pipe extraction empty) and must go red on each.

First consumer: `scripts/ci/handle_table_ceiling_gate.sh`.

The sentence for the next gate is the comment at the top of
`gate_assert.sh`: before you assert anything, the file exists and
starts with `\\x7fELF`, the compile log names the engine, and the rc
is in a file that process wrote.

## Ratchet

`gate_vacuity_gate.sh` remains the emptiness ratchet. Do not fold the
98 into that baseline in this change — different pattern, and 98 cannot
be cleaned in one sweep. New gates that compile to a native ELF and
omit `require_elf` are a leak; catch them in review until a compile-ELF
ratchet exists.

## AI disclosure

Census and helper by AI agent (grok-cli1) under human direction,
2026-08-18. GAIDeT-ICMJE 2025.
