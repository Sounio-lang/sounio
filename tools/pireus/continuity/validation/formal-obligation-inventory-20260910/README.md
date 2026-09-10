# V13/V14 receipt inventory — 2026-09-10

This packet indexes the historical parity receipts present at source
342b3f4e36c78c48d70b3ca27b7ef69eb659001e. It preserves every key/value occurrence
in source order, including repeated fields, and the SHA256 of each receipt.
The inventory requires current receipt bytes to match that source revision.

Run from any directory:

    python3 -B tools/pireus/continuity/validation/formal-obligation-inventory-20260910/inventory.py --check

Use the opening receipts as historical context. Later specialized receipts
record narrower results and remaining obligations. In particular, inspect
these together when selecting the next V13 formal task:

- operator_orbit_class_reconstruction.formal-parity.v13
- operator_orbit_admission_reconstruction.formal-parity.v13
- concrete_quotient_action.formal-parity.v13
- streaming_minimum_correspondence.formal-parity.v13
- executed_streaming_probe.formal-parity.v13

The last receipt records a single frozen probe comparison and explicitly leaves
general_executed_sounio_streaming_equality=false. The streaming receipt records
formal_parity_scope=MODEL_ONLY_EXECUTED_SOUNIO_LINK_OPEN. These are transcribed
receipt statements, not new proof results from this inventory.

For V14, multiprobe_block_certification.parity-open.v14 records
actual_block_execution_complete=false and names
BLOCK_RECEIPT_PRODUCER_AND_MATERIALIZATION as its next concrete step.

This packet does not rerun Lean, validate theorem dependencies, or close any
formal, effect, lowering, material, performance, or pilot obligation. In
particular, comparing receipt bytes does not independently verify their claims.
It changes no frozen runtime, diagnostic protocol, source-CI binding, or
execution helper.
