<!-- docs:meta
topic_id: repo.docs.architecture.kaxi-adapters
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.kaxi-adapters
-->

# K-AXI Adapter Flags

`flags` is a 32-bit packed field with this layout:

- `op_kind`: bits `[3:0]`
- `failure_inc`: bits `[15:8]`
- `success_inc`: bits `[23:16]`
- `alpha_q8`: bits `[31:24]`

Packing formula:

`flags = (op_kind & 0xF) | ((failure_inc & 0xFF) << 8) | ((success_inc & 0xFF) << 16) | ((alpha_q8 & 0xFF) << 24)`
