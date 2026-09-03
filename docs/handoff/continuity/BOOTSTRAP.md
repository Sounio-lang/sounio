<!-- docs:meta
topic_id: repo.docs.handoff.continuity.bootstrap
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.bootstrap
-->

# BOOTSTRAP — paste this into a fresh Opus or Haiku session

> You are executing one work packet of the fable5 continuity campaign (Madaros generic-`<F>` phase 2 + EISA default-lane parity) in the Sounio repo at `/workspace/sounio`.
>
> 1. Read `docs/handoff/continuity/SCOREBOARD.md`. Pick the FIRST work packet with status TODO whose `Model` column matches you and whose `Deps` are all DONE.
> 2. Post a CLAIM entry in `artifacts/omega/agent_handoff.log.md` (append; agent/lane/time_utc/files/intent/status fields, matching the log's existing format). Check first that no other agent holds a live CLAIM on the same files.
> 3. Read your `docs/handoff/continuity/WP-<id>.md` brief and execute it TO THE LETTER. The brief contains the diagnosis, exact file anchors, witnesses, and gates — do not re-derive, do not expand scope.
> 4. Any NEW defect you find: add a row to the scoreboard's new-gap ledger and move on. Never chase it inside your WP.
> 5. When done (or blocked): update your scoreboard row (status + evidence + commit sha), append a RELEASE entry to the handoff log, commit both files together with your work.

## Global guardrails (they cost the previous sessions hours — respect them)

- **FALSE GREEN**: the compiler's exit code lies (exit 0 on failure; silent miscompiles with clean rc). ALWAYS run the produced ELF and verify its actual stdout/exit code against the expected values in your brief.
- **No `println` of computed locals in Madaros witnesses** (`let y: i64 = x+11; println(y)` segfaults — pre-existing). Use `fn main() -> i64` + exit code, or `print_int`.
- **Serialized surfaces**: `self-hosted/compiler/lean_single.sio` and the `bin/souc*` binaries are Lane-4-token surfaces — do not edit them in any WP here. Never `git add -A`. Print `git branch --show-current` after every checkout.
- **LoRA/Contracts gate**: if you edit any file mirrored in `datasets/sounio-code-examples/train.jsonl`, resync that entry (completion must byte-equal the file) and run `bash scripts/ci/lora_assets_gate.sh`.
- **Builds**: Madaros modular build = `bash scripts/ci/build_modular_madaros.sh /tmp/<name>` (~2-4 min, raw ELF). Compile+run = `MADAROS_RAW_BIN=/tmp/<name> ./bin/madaros compile <t.sio> -o /tmp/out.elf && chmod +x /tmp/out.elf && /tmp/out.elf`. lean_single build = `./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/stage1` (interface `<src> <out>`), then self-compile to stage2/stage3 and `cmp` stage2 stage3.
- **CI parity** (what actually gates a PR): `SOUNIO_FORCE_SOURCE_BOOTSTRAP=1 bash scripts/ci/selfhost_host_gate.sh`, then `cp <gate-dir>/artifacts/souc-stage2 /tmp/final_boot4.elf && bash scripts/ci/souc_v2_gate.sh`, then `SOUC_NATIVE=<stage2> bash scripts/selfhost/selfhost_native_runtime_proof.sh`. The local `lean_single_fixed_point_gate.sh` is red even on clean main via the wrapper — it is NOT the CI gate; ignore it.
- **Umbrella** (Madaros regression matrix): `bash scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` (~100s-8m). Run BEFORE and AFTER your change; pre-existing reds are listed in the scoreboard; any NEW red is yours to fix before finishing.
- **4-cell bisect**: if two changes are in flight and a gate breaks, test all 4 combinations before assigning blame.
- If you spawn subagents and one stops "waiting for a build notification", resume it with: "poll your build result directly; do not wait for notifications".

## Commit/PR conventions
- No `Co-Authored-By` trailers. Commit messages follow the repo style (`feat(madaros): ...`, `fix(native): ...`).
- PR bodies list: what changed, witness table with ACTUAL outputs, gates before/after, known residuals.
- Squash-merge on green CI. If a Contracts/LoRA failure appears, see the LoRA guardrail above.
