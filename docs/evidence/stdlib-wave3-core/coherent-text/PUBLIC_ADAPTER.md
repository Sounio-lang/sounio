# Public coherent corpus adapter

Slurm 19291 completed 0 in 3m38s, all 18 expected/actual phases agree.
This changes scripts/stdlib/wave3_core_gate.sh by 14 inserted and 2 removed lines:
an optional explicit WAVE3_COHERENT_KIND selects the previously qualified
runner/collector argument protocol. Default legacy invocation stays unchanged.

The actual original Sounio Unicode conformance harness was separately checked,
compiled, and executed without user arguments through both old and new gates.
Legacy ELF:
9259984e27b2faa877563515423e8705c7029408336eca014c2f6ec69926dbd9

The old public gate fails on the coherent ELF because it supplies no argument;
the new public gate executes kinds 1/2/3 and validates complete collection of
each declared private one-chunk manifest. These are focused adapter controls
using byte-identical first official chunks. They do not constitute complete
official corpus acceptance.

Wrong kind, mismatched family and unsupported phase return 64 before creating
output. Tampered kind, tampered argc and missing invocation records fail 1.
The public gate additionally pins ELF_INVOCATION.tsv in PART_RECEIPTS.sha256.
The coupled ELF remains 2bd52fdededc355138c796d6de92747d5a84c1e7764c9e10f6a69d9560048454.

Exact public gate SHA256:
284b3fe116e476a4bda286e2b16875994dec6e31a1870ae7cf770b32967bd869

Exact legacy compiled-source manifest SHA256:
61cea62ece76dd88e238af41b3173cd934bd0adf6094822589e30c7b8443f9ed

Raw compiler/runtime and coherent compiled-source manifest are those in README.
The worker primes both copied public wrappers before freezing the legacy source
manifest. The original 19289 FAILED1 receipt is retained: a copied wrapper
verification-cache fingerprint changed after the source manifest was frozen.
The source/ELF guard correctly rejected it. No checksum guard was removed.

Slurm 19321 completed 0 and durably archived qualified source, both gate
snapshots, the actual legacy ELF, all focused inputs/outputs and the original
failed 19289 attempt. The archive also preserves failed full-worker 19278,
which requested 000000.bin after the canonical wrong-level generator produced
wrong-level.bin; no broad corpus work had started. The corrected full replay
is independent of this public adapter receipt.

Durable directory:
/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/coherent-text01

public-adapter-source-receipts-19289-19291.tar.gz:
47bf87644fd5db239ee0e0b30937b1949bbad4f3fd0d3f362b0c36d475f2fe95

Existing ARCHIVES.sha256 is unchanged; this append-only artifact has its own
PUBLIC_ARCHIVES.sha256 and final verification. An extracted gate digest was
compared with the live qualified input. Raw receipts retain exact tool bytes.

This is not a new BIDI primary-oracle result. Pristine strict remains failed.
The complete coupled-ELF corpus replay, generic storage and unresolved nominal
alias/field/ref lowering remain separate gates.
