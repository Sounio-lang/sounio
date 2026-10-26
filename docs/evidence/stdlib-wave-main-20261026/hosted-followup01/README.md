<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-wave-main-20261026.hosted-followup01.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-wave-main-20261026.hosted-followup01.readme
-->

# Hosted follow-up: fixture dependency and preserved privacy failure

> **Status**: Beta | **Last validated**: unvalidated | **Source**: hosted logs and operational fake-compiler controls

Base PR3200 head e338f291d352e745691868cf0a640a5d9063896e retains two distinct Scientific Contracts failures. This follow-up fixes only the ChangedTests fixture dependency. Compiler/seed sources, `.github/workflows/ci.yml`, scientific algorithms, annotations and expected refusals remain untouched.

## Fixture dependency repaired

[ChangedTests job121526597741](https://github.com/Sounio-lang/sounio/actions/runs/40309427300/job/121526597741) checked the hosted merge0903f3503aa30c284f7ebdf5abd8a46538cd0f9c of base head e338 over mainc291. The isolated selftest copied current-main run_sio_test_suite_v2.sh but omitted its newly extracted dependency scripts/dev/sio_test_case.sh. The runner reported missing-file and run_test-not-found, produced zero results for one selected case and correctly refused as INCOMPLETE. This is fixture materialization failure, not a scientific result.

The only production change adds one copy of the unchanged main helper into the isolated fixture. Removing this line yields the byte-identical original selftest, preserving all26 control assertions and expected results. Source/helper identities and the exact diff are retained. An actual operational rerun returned0 and MADAROS_NATIVE_PHASE_OPERATIONAL_SELFTEST_PASS, with all26 original cases. It compiled only the operational C++ fake compiler and copied utility ELF; no real Sounio compiler or scientific computation ran. Complete local fixture receipts remain at the path printed in SELFTEST.stdout. No hosted rerun is claimed by this local control result.

## Canonical-seed privacy failure remains

[Private-field job121526597742](https://github.com/Sounio-lang/sounio/actions/runs/40309427300/job/121526597742), same run40309427300 and merge0903, passed the canonical self-compile and SeedReceipt provenance checks, then failed the first negative array fixture. Actual compile status0 emitted a594749-byte ELF where status1/E259 and no ELF were required. The negative ELF was never executed. RESULTS.tsv contains the original single failed-expectation row; gateSTATUS1 remains preserved. No passing privacy gate or whole-CI-green claim is made.

The seed SHA256 is06c90e5b408e3f42f7dbf5c0ef34817333381a770a0734dae4493e36a496c939; lean_single source136234a186fe360417c9e9aa8686839ac232250c53a86bda930d321c13cfab53; negative source7b145f9b4b749925cacf4ad478611501cbf781499f92c57c358c74e126bc2739; emitted ELF0e18c5ae5bb15bd557c113c65aa5bd8d647e4bae5e173a2c3a0ee4c61664611b. GitHub artifact12643867178, private-field-contract-receipts, archive143731B, carries the original compiler/source/output manifests. Its GitHub archive digest is2a31f6d16f5bbbdad2ac1b0cd05d4bd3c92053803d147626555425d3a2718fb7. Only selected text receipts and artifact metadata are committed here; the negative ELF stays outside Git custody. This observation applies to the seed actually selected by that hosted job, not automatically to raw Madaros a292.

An investigation request with exact hashes was sent through canonical coordination to devpass/madaros-v13-gen3-64g-fixedpoint: msg-1793011789-1933763-5523 and detail msg-1793011919-2272349-18358. Existing SIO181 received an adjacent-evidence note without changing Claude ownership, its14 consumers, or global status. No new duplicate card or gate relaxation.

## Other child observations

PR3201 head3f39a3def828825718ee63c97ce6136a9843a8f4 failed Impact run40309445089/job121526657400: stdlib changes had no newly ADDED valid changelog fragment. Contracts job121527329741 then failed because core and ontology were skipped. Those jobs did not reach README governance validation. Root owns the fresh-fragment repair. PR3202 head67d043b80e38e7c708805d6872518914ebb36c34 had no failures in the captured initial/refresh snapshots; Core/Ontology were still running, so no final-green claim.

HOSTED_LOG_EXCERPTS.txt records exact source/step/line context; HOSTED_LOG_IDENTITIES.json hashes complete logs retained in the temporary triage directory. PRIVATE_ARTIFACT_IDENTITY.json records GitHub provenance. FILES.sha256 covers this compact text-only follow-up. Source review, local operational result, hosted checks and compiler qualification remain separate.
