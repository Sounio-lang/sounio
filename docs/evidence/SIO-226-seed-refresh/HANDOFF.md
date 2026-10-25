<!-- docs:meta
topic_id: repo.docs.evidence.sio-226-seed-refresh.handoff
authority: repo_only
audience: users
last_validated: 2026-10-25
validated_by: Slurm17539
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.sio-226-seed-refresh.handoff
-->

# SIO-226 source-bound seed repair handoff

Notion: https://app.notion.com/p/4043f37f937181eea142da2eadcfeed1
Existing PR: https://github.com/Sounio-lang/sounio/pull/3119
Worktree: /workspace/sounio-SIO-172-private-20261025
Branch: devpass/SIO-172-bootstrap-private-borrow

The original compiler source fix remains unchanged, SHA702b4bf3. The metadata
port is ebb906a9ce. Fresh Slurm17539 completed0:0/92s: g0old895 -> g1 -> g2 -> g3,
g1==g2==g3==retained13003, MD5358cfd5e. All four generation hashes are retained.
The old seed and receipt remain on the worker and in historical Git commits.

Canonical-before fails exactly as hosted CI. Canonical/provenance-after and final,
including the wrong-source control, pass; unchanged verify script passes M2/M3.
Private gate44 E259 cases plus original Phonon refusal pass; x86 positives run,
A64 positives are ELF-only and the existing A64 global E001 is disclosed.
No Python execution; active receipt metadata is Node. Six accepted legacy and14
specific refused metadata cases pass, old receipt ASCII roundtrip matches.
Source and Node/dynamic-library hashes verify before/after. Final staged source
catalog matches the own worktree after seed/receipt copyback.

Root must review the exact seed, receipt, metadata diff and evidence before push.
Do not merge main, enable auto-merge or enqueue. The root's later feature merge
must retain the matching compiler source and seed; source drift requires another
canonical qualification. This packet does not close full corpus or joint stdlib.

Node available for root joint jobs:
/scratch/sounio-stdlib-inference-scale-20261024/SIO226-seed01/tools/node
Worker and text evidence details are in README. Claims are in canonical state
/tmp/sounio-coord/2493585255 through /workspace/sounio/bin/sounio-coord.
