CLEAR

The protocol's JSON parsing and field validation logic correctly enforces that the proposal cannot contain context-only fields (6-8), and the test suite confirms this through adversarial test cases. The key validation steps are:

1. **Field Whitelisting**: The `parse` function explicitly checks that context-only fields (6-8) are not present in the proposal by validating `p.seen == 65087`. This bitmask corresponds to keys 0-5 (0x3F) and 9-15 (0xFF800), but **not** keys 6-8. The prior claim about key 7 being incorrectly allowed was based on a miscalculation of the bitmask. The correct bitmask 65087 (0xFFBF) excludes bits 6 and 8-15, ensuring context-only fields are rejected.

2. **Adversarial Tests**: The `test_admission.py` suite includes tests like `result-injection` and `authority-injection` that attempt to inject unauthorized fields (e.g., `expected`, `claim_ready`). These tests confirm that such fields are rejected, validating the parser's strict field validation.

3. **Tensor Hash Integrity**: The tensor hash is computed solely by Sounio using the injective cd16-abk-offset1-v1 encoding. Since no tensor data is accepted from the proposal, there is no vector for hash forgery via adversarial input.

4. **Snapshot Binding**: The `context` field in the proposal is validated against the SHA256 hash of the context file, ensuring immutability and integrity of the trusted orchestrator input.

No concrete blocker exists. The protocol's design and implementation align with the stated security goals, and the test suite confirms robustness against adversarial input.
