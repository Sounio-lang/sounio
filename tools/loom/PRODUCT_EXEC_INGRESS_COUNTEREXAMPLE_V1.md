# LOOM product Exec ingress counterexample v1

Status: current product attachment falsified; execution remains closed.

## Result

The preregistered counterexample gate executes the existing kernel-memory
custody selftest and inspects the exact issuer call path. The custody test's
positive issuer is not an externally authenticated hook event: the harness
constructs `PreToolUse` JSON itself and invokes the native OCaml `agent-hook`
binary. The kernel accepts it because the process has the expected UID,
executable, cwd, argv shape, token file, and harness ancestry.

This is a useful positive result for custody and a decisive negative result for
product ingress. The in-memory handle remains single-use, crash-revoked, and
outcome-bound, but the authority to mint it is still reproducible by a hostile
same-ancestry process.

## Exact boundary

The live checks establish:

- a same-UID peer outside the harness ancestry is refused;
- the same OCaml binary inside the ancestry can submit fabricated hook JSON;
- `EXEC_*` still reads the session bearer file;
- the Codex and Claude hook configs in this source worktree invoke the native
  OCaml runtime directly;
- the repository-wide Python compatibility launcher and runtime still exist.

The result does not claim that a hostile process exploited a human session. It
proves that the current acceptance predicate cannot distinguish the genuine
and fabricated issuers defined by the Garden. That is sufficient to falsify
product attachment.

## Reproduction

Command:

```text
bash scripts/ci/sounio_loom_product_exec_ingress_counterexample_selftest.sh
```

Result:

```text
sounio-loom-product-exec-ingress-counterexample-selftest: PASS semantic_authority=Sounio action=9030 operational_kernel=OCaml current_hook=forged-JSON-from-harness current_counterexample=accepted counterexample_falsifies_product_attachment=true shared_bearer_file=true same_uid_same_executable=true same_harness_ancestry=true outside_ancestry_control=refused missing_fact=non-bearer-inherited-ingress native_hook_config=codex+claude legacy_python_compatibility_bridge=present python_executed=false rust_executed=false product_exec_ingress_observed=false same_ancestry_forgery_refused=false non_bearer_product_ingress=false production_activation=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next=descriptor-bound-dark-ingress
```

## Frozen measurement

- Garden SHA-256:
  `067996ec5031fa77721664dc39c403bf20bea9cf979fcb7b841eed0f11f35c2b`
- Gate SHA-256:
  `8d89799a426ca01c67e0fe33e9a0661c6bf9984baafe21271ef0342ef8695dbf`
- Kernel source SHA-256:
  `be04045103bf41810017a70960d89d268a5c3d5031b325bb161592abac806177`
- Exec client source SHA-256:
  `60a597357088f48c5ffcc88c07c31dc6f4cec777e25f6a3ddb2649f80eb1ccdd`
- Codex hook config SHA-256:
  `2ee5eccde1cd334bdb5b130a27b779442251b6c3c392eb7603f666fc02f638ed`
- Claude hook config SHA-256:
  `f27f14b23e4359320148ad4f2dce11ea47489191a5c4596ffcb74ff9d64bfbd4`
- Kernel: `Linux 7.0.2-5-pve`
- CPU: `INTEL(R) XEON(R) GOLD 6526Y`
- OCaml: `4.14.1`
- Dune: `3.14.0`
- Bash: `5.2.21`

## Claim boundary

The counterexample may establish
`counterexample_falsifies_product_attachment=true`. It cannot establish a
product ingress, same-ancestry refusal, material execution, or production
activation. All attachment flags remain false.

The next experiment is the descriptor-bound dark ingress preregistered in
`GARDEN_PRODUCT_EXEC_INGRESS_V1.md`.

