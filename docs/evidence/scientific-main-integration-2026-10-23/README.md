<!-- docs:meta
topic_id: repo.docs.evidence.scientific-main-integration-2026-10-23.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.scientific-main-integration-2026-10-23.readme
-->

# Current-main scientific stdlib integration — 2026-10-23

Source19c684a99bb5728c15b424a13fd190742efad718 reconciles the later scientific
branch with main3ef3dbaa6e. Fresh Slurm15992: COMPLETED0:0,23 typechecks and five
actual native demonstrations. Slurm15990 is retained as a failed transport
integrity guard, not a failed compiler result. No test suite was run.

RESULTS.tsv records actual command statuses. BUILD_RESOURCE and native resource
receipts distinguish compiler build, whole job and native program measurements.
PACKET_MANIFEST.sha256.gz binds the selected source packet, including its explicit
packet-only dispatch replacement. Scientific/compiler Git source remains19c684a99b.
The retry uses an exact validated snapshot; Forge's default */data exclusion would
otherwise remove stdlib source. Canonical Forge itself was not modified.

`native/` contains byte-extracted SVG/HTML/report files, compressed raw stdout,
compile logs and ELF/stdout identities. Extraction recipe is preserved; no numeric
oracle or pixel comparison is claimed. Actual full-aspect PNG previews are retained
in the local delivery directory, while vector originals are versioned here.

Compiler/program ELFs and complete source snapshots remain in private t560 custody:
/home/devsounio/sounio-artifact-archive/scientific-integration-20261023/final
The complete immutable source tar also has a private copy and an identity here;
this is host-local custody, not independent physical backup. No historical files
were removed or shared quota increased. Large execution outputs stayed off quota.

See the execution receipt for exact resolution decisions and remaining acceptance
boundaries. Source review, typecheck, execution, figure inspection, numerical
concordance, complete corpus, CI and merge are separate claims.
