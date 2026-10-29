# Frozen material-engine inputs

These five Markdown files are byte-exact historical Git blobs, retained solely as inputs to the existing frozen admission contract. Their prose is historical content, not a new validation record or current hardware claim. Current documents are unchanged.

FROZEN_PARENTS.json binds the full twelve-element argv order, each expected SHA-256, the original current path/hash, and the historical commit/blob for every extracted preimage. Seven parents already match and retain their original paths. The five replacements come from commit8e2f11899cd5b556f6abbf3a01d8c9ce431f3915.

The existing pmea_expected_parent_hash constants and all original assertions remain unchanged. A qualified raw8abe run must accept these twelve inputs and continue rejecting no arguments and the drifted current-document set.
