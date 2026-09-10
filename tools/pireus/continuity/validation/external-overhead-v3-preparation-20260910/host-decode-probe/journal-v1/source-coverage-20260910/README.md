# Journal orchestration source coverage

This read-only audit compares all nine current journal orchestration files with
Git blobs in a full commit. It does not request CI evidence, change the protocol,
create an execution freeze or submit hardware work.

The old protocol source 342b3f4e36c78c48d70b3ca27b7ef69eb659001e fails:
ops/host_decode_journal_attempt.py is missing from that source. The CPU36
candidate e91a7a17ea47fc3df5fbd7e9eea0e27e85ffeda7 contains all nine expected
blobs. This content result does not qualify either source for inference.

A new protocol must bind the approved source after its required CI checks pass.
The old protocol and attempted CI evidence remain immutable. These receipts
are prospective tooling evidence, not part of the existing execution freeze.

Four unit tests use a real temporary Git repository, including a dirty worktree
that cannot repair the committed bytes, an uncommitted helper that cannot
satisfy source presence, invalid inventories and explicit false claim fields.
