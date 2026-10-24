<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.functional-final-6ffc.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.functional-final-6ffc.readme
-->

# Canonical functional gate17020 at source6ffc

This is the canonical final-source functional receipt for Slurm17020 and source
`6ffc1a5bd8eaebeef2d15ccced830bfbc832a648`, frozen at
`/scratch/sounio-stdlib-inference-scale-20261024/combined_final_cache04/source`.
The run completed STATUS0 and scheduler COMPLETED0:0, elapsed00:21:07. Its actual
RESULTS.tsv contains the preparation, negative controls, fixture/example checks,
compilation, native execution, stdout markers, parser/render/numerical C++
concordance and rejection controls. All recorded outcomes are0. Negative and
mutant outcomes are normalized to0 when their expected rejection was observed;
that does not mean invalid source or incorrect native output was accepted.

This selected integration gate is not a claim that every repository test ran,
that every numerical input is supported, or that any calibration/performance goal
has been met. Runtime and actual stdout belong to their named programs; no
isolated-kernel speedup is inferred. In particular, aggregate cancellation now
has the conservative unavailability policy recorded in ../aggregate-cancellation.

## Immutable receipt identities

- Compiler d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6,
  separately guarded/selfchecked in ../compiler-effectbridge.
- Canonical initial input manifest
  c15be33cd683af0e625888d5dc284d0170935bd5646e934d776866e260454fb9.
- Actual RESULTS.tsv
  d058e284ae30666e1d51d52fd8d6bd579954c001523cbac61c1ffc69e96efbe2.

The initial and final input manifests, input-content verification, input-catalog
verification, compiler verification and root's independent current-input manifest
are copied. RESULTS includes actual prepare0, input-hashes0, input-catalog0 and
compiler-hash0 rows. The original seven-field FUNCTIONAL_GATE_RECEIPT is retained
without rewriting its cluster paths. ROOT_READINESS and the final03 packet's
actual readiness stdout both identify source6ffc, job17020 and compiler d74.
The packet's submission status0 and exact verify-ready source are retained as
provenance; copying this evidence does not initiate or accept a campaign.

Those receipt paths remain cluster-bound by design. Replay requires the exact
source/input inventory, compiler bytes and original manifests/RESULTS at the
recorded locations, or a separately validated relocation with new provenance;
editing a copied receipt's paths/hashes is not equivalent to this acceptance.
No replacement green receipt was created while assembling this folder.

## Full text and deterministic archive

index/ exposes original RESULTS/STATUS and input/compiler manifests/verification.
receipt-readiness/ preserves actual receipt, source head, job/submission, root
checks and final03 readiness controls. FULL_TEXT_RECEIPTS.tar.gz contains the
complete selected executed text artifacts and Slurm logs: native stdout/stderr,
resources, negative diagnostics, compile/check logs, identities, actual C++ parser,
render and numerical concordance, and their rejection-control receipts. Generated
Sounio/C++ case sources and executables are excluded, listed in EXCLUDED.tsv.
No ELF, cache, synthetic prebuilt payload or full source archive is included.

The archive uses a sorted NUL file list, fixed uid/gid/mtime, fixed pax header
paths, no pax atime/ctime, and gzip without timestamps. A second independent pack
had the same hash; extraction verified every member against its original-byte
manifest. Archive SHA256:
`96428c5e5bf315e5610fea6bf7b5bb7c935cdeb82c9e9fb427cb4c4fab2ea550`.
FULL_TEXT_ORIGINAL and FULL_TEXT_RESTORE preserve that member verification.
Large loose text is losslessly gzipped. SOURCE_MAP/ORIGINAL_FILES bind copied
metadata to original scratch paths; archive members have their own full byte
manifest. FILES.sha256 is the final copied-file manifest for this folder.

## Separate rejected historical receipt

historical-17009/ is explicitly NON-CANONICAL. That earlier source956c mathematical
integration job completed0, but its initially catalogued source inventory did not
include later materialized prebuilt/stamp paths. Root's current-manifest comparison
therefore differed. Preserved Slurm17016 readiness recheck exited65, before any
campaign/submission, with the original guard/receipt input hashes verified.
This rejection is not erased or turned into a numerical failure. The earlier
successful mathematical rows do not authorize qualification or substitute for
17020's preparation-first, unchanged-catalog final receipt.

All full scratch artifacts remain preserved. This custody operation copied and
hashed existing evidence only; it ran no numerical/native work, touched no frozen
stdlib/test/script source, made no CI or submission change, and does not include
live pilot/campaign outputs owned by the separate qualification monitor.
