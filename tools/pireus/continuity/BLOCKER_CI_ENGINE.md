Blocker-ID: BLK-20260907-pireus-ci-engine-acceptance
Status: owned
Severity: B2
Class: compiler-integration
Evidence-Level: E3
Owner: codex-pireus
Worktree: /workspace/.wt/pireus-integration-20260906
Branch: codex/pireus-inkling-cycle-20260906
Observed: CI Full Test Suite fails Pireus cases using /tmp/souc-stage2; ci.yml explicitly identifies that artifact as raw lean_single. Current-source Madaros scoped gates are a separate evidence class.
Evidence: validation/ci-legacy-suite-observation.json; GitHub run 34067230235 job 101578538103
Acceptance-Gate: retained Pireus cases execute successfully on their supported compiler in CI, with baseline and engine comparisons; no skip/tag/expectation change used to manufacture green
Next-Action: compare exact base-branch failures and run the failing cases with rebuilt Madaros 859d37dd115c9f51a3182ff3fcb1a6c72efcd5b65547fe609d49fa2b97cb67a7 on Slurm before changing compiler routing or expectations
Fallback-Path: none
Legacy-Kept: yes
Automatic-Merge: false

The workflow-reference Contracts failure has a separate repair: retain frozen
workflow bytes and expand path-filter globs in check_workflow_script_refs.sh.
Its local positive/negative controls and 240-reference repository check pass.
Remote acceptance of the resulting commit remains to be observed.
