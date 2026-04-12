# CLAUDE_HANDOFF.md

Active workspace:
- `/workspace/sounio`

Recovered origin:
- recovered from VM `sounio-dev-01`
- older prompts and scripts may still mention `/home/demetrios/RustroverProjects/sounio`

Safe branch:
- `integration/sounio-dev-ready-base`

Active compiler/test entrypoints:
- compiler wrapper: `./bin/souc`
- canonical resolver: `scripts/lib/resolve_souc.sh`
- canonical suite harness: `scripts/run_sio_test_suite.sh`
- native compiler source: `self-hosted/compiler/lean_single.sio`
- promoted native artifact: `artifacts/self-hosted/souc-self-hosted-x86_64`

Current operating rules:
- treat `/workspace/sounio` as the only active development surface
- prefer executable scripts over stale prose when they disagree
- do not use destructive “align with main” workflows unless explicitly requested
- preserve unrelated dirty worktree state unless the task explicitly targets it

Current recovery context:
- the repo contains recovered history plus active self-hosted compiler work
- `bin/souc` delegates to the promoted native artifact
- runtime/compiler work should be validated through `scripts/run_sio_test_suite.sh`
