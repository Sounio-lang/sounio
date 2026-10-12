<!-- docs:meta
topic_id: repo.docs.handoff.root-history
authority: repo_only
audience: users
last_validated: 2026-10-12
validated_by: Codex
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.root-history
-->

# Root history and preserved evidence

Historical agent logs are retained unchanged in [artifacts/agent-logs](../../artifacts/agent-logs/). Their earlier location was `agent_logs/`; references outside preserved records now use the archive location. Historical commands inside the records describe their original checkout, not the current file tree.

The former root `TASK.md` is retained unchanged as [the lane 8c dossier handoff](../../artifacts/handoff/lane-8c-dossier.md). It records that lane's historical instructions and does not assign current ownership.

The previously tracked root JUnit report is retained unchanged at [legacy-root-junit.xml](../../artifacts/test-history/legacy-root-junit.xml). The test harness and CI still generate and upload `test-results.xml` at the root; that runtime output is ignored by Git.

Introductory demonstrations now live in [demos/intro](../../demos/intro/). The tour, talk and current setup guides use the new paths. Compiler, standard library, research manifests and source entrypoints are unchanged.
