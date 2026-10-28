# Native pool / ADC custody receipt

Source implementation commit: d40ed0e285.
Initial pool/file implementation: 83778fc7b9.
Own-branch runtime adoptions 54be2fc9ea and ffe693e88c duplicate root's
e6a78eafba and c1b92bf7d6; skip those adoptions when integrating into root.

Full component execution: Slurm 18781, COMPLETED 0:0, 3m42.
Same-raw original failing source: Slurm 18788, COMPLETED 0:0 as an
expected-failure supervisor control; actual supervised child status143 after
the fatal managed-handle exhaustion marker. The original 600s budget and
million-row bytes were retained. The wrong-caller audit18785 is preserved
and is not treated as million-row evidence.

Durable export and supplemental validation: Slurm 18804,
COMPLETED 0:0, 34s, marker NATIVE_POOL_DURABLE_EXPORT_AND_SUPPLEMENT_OK.

Archive:
  /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/native-pool-closed-export01/native-pool-qualified-source-and-receipts.tar.gz

SHA256:
  f80fa5205d49dc5e6c252089e1c767b1550d08129e07ffc5d3a210c1f554fd56

The archive was listed, synced and read back by SHA256. PACK_INPUTS verification
has zero FAILED entries. It contains final source and artifacts, the exact
original million-row caller/library and inputs, the EOS failing-before file,
failed-attempt logs, raw21b3 and rawc23 compiler bytes, selected compiler source,
seed and original build receipts. The historical compiler source manifest also
verified0 against its recorded inputs; its original outer worker failure is
retained, not relabeled as compiler-wide success.

The EOS supplement reuses exactly the invalid bytes accepted in18780.
The corrected judge returns1 with explicit Arrow EOS missing. Both old/new
valid257-row input and Arrow output compare byte-for-byte equal.

The tracked harness has one metadata-only tightening after18781: closure
refusal matches an anchored error line. Its SHA256 is
88cb9754cd9198912fbb7b2ae8832769b6f567d40f57d7ff3289ea24f82a2dc6.
The supplement verifies the real #1542 error and rejects a filename-only spoof.
No native Sounio/runtime implementation or scientific input changed after the
full execution receipt.

Observed final native RSS: small3192KiB, file3152KiB, TCP3948KiB.
Both million-row routes produced62500 batches and exact sum
378114110446473770. General region/managed-allocation callbacks, callback
admission proof, final coherent compiler integration, full stdlib and official
CI remain separate open gates; see README.md.
