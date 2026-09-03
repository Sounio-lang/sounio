<!-- docs:meta
topic_id: repo.docs.handoff.continuity.wp-a0
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.wp-a0
-->

# WP-A0 — Land the Madaros phase-1 PR [Haiku]

**Status at authoring:** PR #654 already OPEN (base `main`, head `coord/fable5-madaros-generic-f` @ `d15915f58`).

## Steps
1. `gh pr checks 654 --repo Sounio-lang/sounio` — wait for all checks. Expected all-green: the diff touches only Madaros-side files (`parser/items.sio`, `check/{specializer,compat}.sio`, `compiler/{main,module_frontend}.sio`, `ir/lower.sio`); the CI Full Test Suite runs on the lean_single stage2 and is unaffected; no LoRA-mirrored file is in the diff.
2. If the Contracts job fails on "LoRA dataset assets": run `bash scripts/ci/lora_assets_gate.sh` locally, resync the offending `datasets/sounio-code-examples/train.jsonl` entry (completion must byte-equal the source file), commit to the same branch, push.
3. If any other check fails: capture the failing job log (`gh run view --job <id> --log`), record it in the scoreboard as BLOCKED with the log excerpt, and STOP (do not improvise fixes to compiler code in this WP).
4. On green: `gh pr merge 654 --repo Sounio-lang/sounio --squash --subject "feat(madaros): generic <F> M-track phase 1 — trait/impl grammar, AST specializer, println i64 fix, #638 (#654)"`.
5. Update `SCOREBOARD.md`: A0 → DONE(merge sha). Append RELEASE to `artifacts/omega/agent_handoff.log.md`.

## Done criteria
PR #654 merged to main; scoreboard + handoff log updated.
