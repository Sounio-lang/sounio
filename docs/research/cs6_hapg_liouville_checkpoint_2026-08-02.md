# CS6 V7-A.1 Liouville carrier-only checkpoint result

**Status:** the exact nine-attempt authoritative Slurm matrix completed, the
in-job retained audit passed, and a second audit from the clean execution
commit replayed all nine workers and all six positive verifiers. Both
alternative carriers emitted complete verifier-accepted Liouville checkpoints
on the previously masked cell. The frozen result is
`BOTH_ALTERNATIVES_PASS`.

This is a narrow carrier-only, two-return checkpoint witness. It does not
retroactively repair the parent V7-A full-pipeline receipt, execute C1 or C2,
validate the downstream section-resident crossing, select a V7-B winner, or
support a promotion, hyperbolicity, attractor, novelty, or open-problem claim.

## Frozen question and boundary

Parent V7-A job `8480` tested 40 cells under three C0 carrier families. On
parent ordinal 23, the baseline `C0HOTripletonSet` stopped with the declared
`rQ=[-nan,-nan]` exception, while each alternative carrier later stopped with
`one-step Newton crossing was not available`. Because the parent worker
serialized only after downstream work, neither alternative left a complete
Liouville receipt for that cell.

V7-A.1 prospectively froze the smallest differentiating test:

- parent ordinals 22, 23, and 24 in that order;
- masked ordinal 23 bracketed by two parent-positive controls;
- `C0HOTripletonSet`, `C0HORect2Set`, and `C0Rect2Set` per cell;
- exactly nine attempts, with no retry, substitution, fallback, or early stop;
- fresh ODE solver, section, map, and set state for every attempt;
- the same ODE, section, direction, order 8, two-return count, tile, challenge,
  compiler flags, and CAPD build across carriers;
- only the physical `x,y,w` flow plus auxiliary `ell`, followed immediately by
  checkpoint serialization and exit;
- no C1, C2, section-resident, or other downstream H-PG computation.

The frozen identities are:

| Field | Value |
| --- | --- |
| Contract SHA-256 | `3afc0475847ad8054234a2ddfa108b768cfd81991d0be71fc21c991f363631ce` |
| Coordinate SHA-256 | `527afc7c205fcf09b15a0bff91df6935f19ed2b7e7926895916ac5da33a992a7` |
| Root challenge | `ad536f25d02182c8b2add12ef1a7e8a8a18b4eb0d12e68535ea94ddb3eff0bdf` |
| Cell count | 3 |
| Carrier count | 3 |
| Maximum evaluations | 9 |

The oriented `Q0` determinant is the pointwise `xy` Jacobian density with
respect to the fixed global normalized source chart. It is not a finite
subtile-area determinant, and no factor proportional to the number of tiles is
inserted. The verifier checks the frozen Liouville identity

```text
det = exp(ell) * nu0 / nu2 * det(Q0)
```

where `nu0` and `nu2` are the independently recomputed positive normal
velocities at the source and returned section states.

## Authoritative execution

| Field | Authoritative value |
| --- | --- |
| Execution commit | `6a88920476c7be0305a4c368338782ec6eb99956` |
| Slurm job | `8496` |
| Node | `gpuorangefs-r770-proxmox` |
| UTC interval | `2026-08-02T18:14:53Z` to `2026-08-02T18:15:19Z` |
| Wall time | 26 s |
| Requested / allocated CPUs | 9 / 120 exclusive |
| Requested memory | 8 GiB |
| Peak batch RSS | 128760 KiB |
| Partition / account / QOS | `gpu-orangefs` / `lab` / `normal` |
| CAPD | 5.3.0, FILIB outward rounding, `O0` |
| Execution path | CPU prebuilt, Slurm, node-local scratch |
| Worker source SHA-256 | `9057b88190867fac590d1894cc1542acb82409e76ea83157589f02e27d3b8551` |
| Worker binary SHA-256 | `642e223cac6d3d0b0194fafd89a81d782fae6320f24a88773a9d6b2aa30b3c12` |
| Source archive SHA-256 | `7b98ab72fcb5a83f0c4e367279de669b8d9984fcb5d077451a96b1a2a930edb6` |
| Prebuilt archive SHA-256 | `865cc565ef7bead98c3766a33b21818a8a8c546657494827ced534bf0f2d1e88` |
| Job script SHA-256 | `af2dedb09159cccafb7d77d0a2450b641755af9c5843fa7a336e2dd809438510` |
| Config SHA-256 | `94f5e1c0427feba7ee1504deaccb444632f47018ebcef08d82aafd67146eac1c` |
| Result archive SHA-256 | `5c13428259e58d562a1ff7aa73ee152c988aa7af1791c4d1abd6e94cf895480f` |

The worker was compiled twice with GCC 13 and the two binaries compared equal.
The prebuilt tar and eight-file Git source archive were each generated twice
and compared equal. `git get-tar-commit-id` bound the source archive to the
execution commit. Local and remote hashes matched before submission.

Slurm recorded `Requeue=0`, `Restarts=0`, `OverSubscribe=NO`, 120 allocated
CPUs on the exclusive node, the exact submitted command, and the frozen time
and memory envelope. The job completed `0:0`; stderr was empty. The partition
also allocated one GPU as a node resource, but the probe used no GPU. No AMD
U250 or other FPGA was installed or used.

`EXECUTION_PROVENANCE_ATTESTED=false` remains part of the frozen contract. The
Git, digest, scheduler, script, config, executable, and replay bindings are
strong execution provenance, but they are not an independent external
attestation service.

## Transport correction before the result

The first authoritative submission, job `8492` from implementation commit
`8eafc674a4f9dedc6df4a31182b47d417c1c98ba`, produced no scientific output.
Slurm allocated the node, but `--export=NONE` implicitly requested login
environment reconstruction. The batch launch was delayed by 111 seconds; its
60-second JWT expired, `slurmd` rejected the launch, and the controller could
not requeue because `--no-requeue` was frozen. Accounting therefore records
`CANCELLED by 0`, not a worker or verifier result.

Two minimal jobs isolated the cause on the same node:

| Probe | Export mode | Result |
| --- | --- | --- |
| `8493` | `--export=NIL` | `COMPLETED 0:0` in 0 s; saw `SLURM_EXPORT_ENV=NIL` |
| `8494` | explicit `SLURM_EXPORT_ENV=NONE` | reproduced the same 1m53s launch-cancellation pattern |

Commit `6a88920476c7be0305a4c368338782ec6eb99956` changed only the wrapper,
runner, retained verifier, and synthetic gate from `NONE` to `NIL`. In Slurm,
`NIL` preserves scheduler variables without importing the caller environment
and does not invoke login-environment reconstruction. The authoritative
wrapper still replaces `PATH`, fixes locale/timezone, clears Python and shell
injection variables, and verifies the live control-plane allocation. The clean
full gate passed after this change. No scientific contract, coordinates,
worker source, or result criterion changed.

## Exact matrix result

| Cell role | Parent ordinal / node | Carrier | Exact status | Liouville determinant |
| --- | --- | --- | --- | --- |
| Positive control left | 22 / `U03-0000000005_S04-0000000010` | `C0HOTripletonSet` | declared `CAPD_SET_RQ_NAN` | not emitted |
| Positive control left | 22 / `U03-0000000005_S04-0000000010` | `C0HORect2Set` | verified; parent KAT pass | `[-0x1.f82a4374d4e82p-36,-0x1.ee914ac0de992p-36]` |
| Positive control left | 22 / `U03-0000000005_S04-0000000010` | `C0Rect2Set` | verified; parent KAT pass | `[-0x1.f843a7904b20cp-36,-0x1.ee78600a257f9p-36]` |
| Masked target | 23 / `U03-0000000006_S04-0000000010` | `C0HOTripletonSet` | declared `CAPD_SET_RQ_NAN` | not emitted |
| Masked target | 23 / `U03-0000000006_S04-0000000010` | `C0HORect2Set` | verified checkpoint | `[-0x1.f7f09dcc7884cp-36,-0x1.ee591a81230ap-36]` |
| Masked target | 23 / `U03-0000000006_S04-0000000010` | `C0Rect2Set` | verified checkpoint | `[-0x1.f80a25e656c66p-36,-0x1.ee3f98f4c3d36p-36]` |
| Positive control right | 24 / `U03-0000000007_S04-0000000008` | `C0HOTripletonSet` | declared `CAPD_SET_RQ_NAN` | not emitted |
| Positive control right | 24 / `U03-0000000007_S04-0000000008` | `C0HORect2Set` | verified; parent KAT pass | `[-0x1.fb4631998557bp-36,-0x1.f1a31bb580bc1p-36]` |
| Positive control right | 24 / `U03-0000000007_S04-0000000008` | `C0Rect2Set` | verified; parent KAT pass | `[-0x1.fb6122cabb36ep-36,-0x1.f188cfb07ccfdp-36]` |

The matrix-level results were:

```text
RUN_COMPLETE=true
RUN_VALID=true
ATTEMPTS_COMPLETED=9
VERIFIED_CHECKPOINTS=6
BOUND_RQ_NAN=3
BASELINE_PREREQUISITE_VALID=true
POSITIVE_CONTROL_KATS_VALID=true
MASKED_STATUS_VALID=true
INITIAL_HULL_INVARIANCE=true
OUTCOME=BOTH_ALTERNATIVES_PASS
MUTATION_TESTS=276
MUTATIONS_REJECTED=276
V7_B_WINNER=NONE
PROMOTION_ELIGIBLE=false
```

The three baseline attempts returned code 1, empty stdout, and the exact bound
tripleton `rQ`-NaN signature. All six alternative attempts returned complete
finite receipts. Each positive control reproduced its exact parent initial
hull and overlapping parent determinant interval. On every cell, the two
successful alternatives reconstructed the same frozen initial hull. Every
verified determinant has a strictly negative upper endpoint.

## Independent checks

For each positive receipt, the verifier independently reconstructed the
dyadic source tile and global source frame, reconstructed the exact-zero `w`
and `ell` initial hull, bounded `exp(ell)` with rational interval Taylor
arithmetic, recomputed the two positive normal velocities, checked section
containment, and checked the Liouville algebra and sign. It then ran exactly 46
predeclared grammar, binding, geometry, algebra, and sign mutations. All 276
mutations were rejected.

The in-job retained audit and the second clean-checkout audit both reported:

```text
AUDIT_PASS=true
ATTEMPTS_RECONSTRUCTED=9
WORKER_REPLAYS=9
VERIFIER_REPLAYS=6
BOUND_NEGATIVES=3
RUN_VALID=true
OUTCOME=BOTH_ALTERNATIVES_PASS
MUTATION_TESTS=276
MUTATIONS_REJECTED=276
```

The second audit used the retained executable and replayed every worker stdout,
stderr, and return code byte for byte. It also reran every positive verifier
and reconstructed the result table, summary, decisions, manifest, and content
index.

The independent-verification scope remains exactly the frozen one: geometry,
serialization, algebra, sign, and binding. It is **not** independent ODE
integration. Worker replay is reproducibility of the retained CAPD executable,
not a second integrator implementation.

## What the result supports

For this exact cell, carrier, and execution, each alternative completed a
two-return Liouville checkpoint that satisfied the frozen verifier. The
baseline reproduced its predeclared tripleton failure on the target and both
controls. The controls also reproduced the already available parent witness.

This differentiates the parent ordinal-23 ambiguity: a complete Liouville
checkpoint is available under both alternative C0 doubleton carriers when the
probe exits at that boundary. The later parent `one-step Newton crossing`
exception therefore did not demonstrate a missing or invalid alternative
Liouville checkpoint. It remains downstream of the boundary tested here.

That localization cannot be inserted retroactively into job `8480`. Parent
V7-A still has no serialized full-H-PG receipt for ordinal 23, and V7-A.1 did
not execute the computations needed to supply one. The result does not rank
the two alternative carriers or claim their propagated enclosures are
equivalent. Binary agreement on three cells is not a general carrier theorem.

## Residual boundaries

1. Only the observed `BOTH_ALTERNATIVES_PASS` branch was demonstrated. The
   frozen contract names three other valid outcomes, but the current declared
   failure classifier recognizes the exact tripleton `rQ`-NaN signature. The
   alternative doubleton types do not expose that same `rQ` component. A
   different alternative exception is conservatively `RUN_INVALID`; the
   nominal reachability of the other three outcome labels was not established.
   This cannot create a false `BOTH_ALTERNATIVES_PASS` in the observed run.
2. The result tar and its SHA-256 sidecar are each published atomically with a
   same-directory no-replace hard link. The exact target is CephFS, and a
   same-directory hard-link preflight passed. Publication of the pair is not a
   single transaction: a process death between the two links could leave a
   complete tar without its sidecar and require manual recovery. Job `8496`
   published and verified both files.
3. There was no fallback path. Legacy V7-A receipts and code remain unchanged.
   The public `souc`, Madaros, ontology wrappers, C1/C2 paths, and U250 path
   were not part of this execution.

## Blocker status and next experiment

The V7-A.1 execution-evidence requirement is satisfied at E4 by job `8496`,
its sidecar, both retained audits, and the clean full gate. No scientific or
execution-evidence blocker remains for this bounded result. Repository
integration is still gated by a disjoint documentation-registry ownership
conflict:

```text
Blocker-ID: BLK-20260802-cs6-v7a1-doc-registry-sync
Status: classified
Severity: B2
Class: ownership-conflict
Owner: codex-root
Lane: cs6-hapg-liouville-checkpoint-20260802
Worktree: /tmp/sounio-cs6-hapg-liouville-checkpoint-20260802
Branch: research/cs6-hapg-liouville-checkpoint-20260802
Files-Owned: .claude/llm_offload_log.md, docs/research/cs6_hapg_liouville_checkpoint_2026-08-02.md, scripts/ci/cs6_hapg_liouville_checkpoint_gate.sh, scripts/research/cs6_hapg_liouville_checkpoint_contract_v1.txt, scripts/research/cs6_hapg_liouville_checkpoint_coordinates_v1.tsv, scripts/research/cs6_hapg_liouville_checkpoint_probe.cpp, scripts/research/cs6_hapg_liouville_checkpoint_retained_verify.py, scripts/research/cs6_hapg_liouville_checkpoint_run.py, scripts/research/cs6_hapg_liouville_checkpoint_slurm_job.sh, scripts/research/cs6_hapg_liouville_checkpoint_verify.py, scripts/research/receipts/cs6_hapg_liouville_checkpoint_*/**
Files-Read-Only: docs/governance/topic-registry.v1.json, docs/governance/DOCS_AUTHORITY_MATRIX.md, docs/governance/DOCS_ACCEPTANCE_REPORT.md
Do-Not-Touch: the three generated governance files while claimed by codex/issue901-authority-current-20260802
Repro: node scripts/docs/check_docs_registry.mjs
Observed: registry and acceptance outputs are stale after adding this report; the required generated files have an active disjoint-lane claim
Expected: one owner regenerates and commits the complete governance output without overwriting either lane
Acceptance-Gate: node scripts/docs/sync_governance_metadata.mjs && bash scripts/dev/check_docs_registry.sh
Evidence-Level: E3
Evidence: scripts/research/receipts/cs6_hapg_liouville_checkpoint_job_8496_v1/governance-sync-blocker.txt
Fallback-Path: pre-commit no-verify evidence-commit exception; branch remains non-merge-eligible pending governance sync
Legacy-Kept: yes
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: the active governance owner transfers the three generated files or incorporates this report and runs the acceptance gate
```

V7-B remains evidence-blocked by a different, narrower missing witness:

```text
Blocker-ID: BLK-20260802-cs6-v7b-full-hpg-bridge
Status: classified
Severity: B3
Class: evidence-gap
Owner: codex-root
Lane: cs6-v7b-full-hpg-bridge
Worktree: /tmp/sounio-cs6-v7b-full-hpg-bridge-20260802 (prospective; not allocated)
Branch: research/cs6-v7b-full-hpg-bridge-20260802 (prospective; not created)
Files-Owned: none; prospective write set not allocated
Files-Read-Only: parent V7-A and V7-A.1 contracts, results, and receipts
Do-Not-Touch: frozen V7-A and V7-A.1 contracts and result artifacts
Repro: grep -E 'C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED|FULL_HPG_PIPELINE_EVALUATED|V7_B_ELIGIBILITY|V7_B_WINNER' scripts/research/receipts/cs6_hapg_liouville_checkpoint_job_8496_v1/result/summary.txt
Observed: all three fields are false; V7_B_WINNER=NONE
Expected: prospective verifier-backed C1, C2, and downstream compatibility evidence for the masked cell and controls
Acceptance-Gate: a separately frozen bridge matrix completes without unknown exceptions and passes in-job plus clean-checkout retained audits
Evidence-Level: E4
Evidence: Slurm job 8496 and scripts/research/receipts/cs6_hapg_liouville_checkpoint_job_8496_v1/
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: freeze a checkpoint ledger across C1, C2, and the section-resident crossing before any V7-B eligibility decision
```

The next smallest experiment is therefore not broader tomography. It is a
prospectively frozen bridge on these same three cells and carriers, with
serialized, independently checked boundaries after the Liouville checkpoint,
after C1, after C2, and immediately around the section-resident crossing. It
must keep V7-A and V7-A.1 immutable and define its own winner and invalidation
rules before execution.

## LLM-offload reviews

The result report received the mandatory M1 dual-provider review:

| Provider | Task | Target | Outcome |
| --- | --- | --- | --- |
| xAI / Grok 4.3 | focused mathematical and scope review | divergence, Poincare determinant, six intervals, counts, claim boundary | PASS on all five claims |
| Z.AI / GLM-5.2 | full report plus focused mathematical and scope review | report and the same five obligations | PASS on all five claims |

The first whole-report Grok leg returned `NO MATHEMATICAL CONTENT` and was not
counted. A first focused prompt then transcribed the executed `y'` incorrectly
as `x*y-(y-w+zs)/2`; Grok rejected the resulting divergence equality. Direct
comparison with the frozen worker showed the executed field is
`x*y-y*(w+zs)/2`. The corrected prompt received two independent PASS verdicts.
This was a review-prompt transcription error, not a change to the contract,
worker, evidence, or report formula. Raw reviews are temporarily available under
`/tmp/llm-offload-e8JC54/`, `/tmp/llm-offload-iwibOQ/`, and
`/tmp/llm-offload-krrtwy/`.

## Retention

Compact receipts are retained under
`scripts/research/receipts/cs6_hapg_liouville_checkpoint_job_8496_v1/`.
They contain all nine receipts and stderr records, all six verification
records, the frozen inputs and contracts, full result and archive hash indexes,
the runner and in-job audit transcripts, scheduler context, build and runtime
provenance, the clean-checkout audit, final accounting, and the raw transport
diagnosis for jobs `8492` through `8494`.

The 5,150,720-byte canonical archive remains at
`/orangefs/training/cs6-hapg-cover/55c331622b6076ac/v7a1-checkpoint-6a88920476c7-ad536f25d021/result.tar`.
Its committed sidecar binds SHA-256
`5c13428259e58d562a1ff7aa73ee152c988aa7af1791c4d1abd6e94cf895480f`.
The compact repository copy omits only the retained 4.4 MiB worker binary and
the nested 256,000-byte Git source archive; their hashes and the complete raw-file
index are retained. Full byte-for-byte worker replay therefore requires the
canonical archive, while all textual scientific and scheduler evidence is
committed.
