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

**Answer in one line.** Pattern 2 (rc-through-pipe) is empty. Patterns 1
and 3 are large. That is a helper, not 98 patches. The helper is
`scripts/lib/gate_assert.sh` (`require_elf`, `classify_compile_log`,
`gate_capture_rc`). Its own gate is
`scripts/ci/gate_assert_instrument_selftest.sh`.

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

Compile-invoking gates: **120**.

| | n |
|---|---:|
| compile, never check `\x7fELF` | **98** |
| of those, no `[[ -s ]]` / `-x` either | **56** |
| compile and *do* check magic | 22 |

98 is design. A compile that exits 0 and writes a file named `-o` is
non-empty; `[[ -s ]]` would have passed the E230 trap. Magic is the
check that would have refused it.

Do not patch 98 files in this PR. New compile gates call `require_elf`.
Existing copies of the four-line `od -An -tx1 -N4` / `7f454c46` test
(native_v2_*, exact_bitwise_*, madaros_full_gate) can switch when
touched.

## Pattern 3 — never record which engine compiled

Of the 120 compile invokers, **72** never mention `--version`,
`compile_engine`, `lean_single`, or `Madaros v`.

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

## Ratchet

`gate_vacuity_gate.sh` remains the emptiness ratchet. Do not fold the
98 into that baseline in this change — different pattern, and 98 cannot
be cleaned in one sweep. New gates that compile to a native ELF and
omit `require_elf` are a leak; catch them in review until a compile-ELF
ratchet exists.

## AI disclosure

Census and helper by AI agent (grok-cli1) under human direction,
2026-08-18. GAIDeT-ICMJE 2025.
