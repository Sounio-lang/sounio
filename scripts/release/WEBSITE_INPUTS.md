# Website input export contract

The website consumes normative documentation, recorded gate evidence and generated
render assets from a particular core commit. This export preserves those inputs
without requiring a compiler checkout. It does not certify the truth or freshness
of historical claims contained in the documents or gate receipts.

Create an export from a committed source tree:

```sh
python3 scripts/release/export_website_inputs.py create \
  --ref HEAD --output website-inputs.tar.gz > receipt.json
```

The receipt identifies the resolved 40-character source commit, archive SHA-256
and number of files. The archive contains a schema-1 inventory recording the
size and SHA-256 of every file. Timestamps, ownership and file modes are normalized
so the same source inputs produce byte-identical archives. Missing selected roots and unsupported members are errors; a failed export preserves an existing
output. Dirty working-tree files are not exported.

Twelve historical proof links under docs/papers/main/epistemic-types/proofs use
absolute paths from an old machine. The exporter recognizes their exact recorded
link text through an explicit allowlist and materializes the corresponding regular
formal/lean4 blobs from the same commit. Their source_path is recorded in the
inventory. It never reads those absolute paths on the host. All twelve proof entries are mandatory and must retain exactly their mapped
source paths. Regular files replacing those links, other source links and changed
link destinations are rejected. File roots must be exact regular files; directory
roots must contain selected descendants. Proof contents are copied without
modification; exporting them does not mean their obligations were rechecked.

Before consuming an archive, verify its digest against a separately pinned release
receipt, then verify every inventory entry:

```sh
python3 scripts/release/export_website_inputs.py verify website-inputs.tar.gz \
  --sha256 <pinned-archive-sha256>
```

The verifier rejects additional, missing, changed and duplicate files, invalid
paths, links, executable modes and unknown schemas/selections. The inventory is
integrity metadata, not a signature; an independently trusted digest is required.
It verifies an archive and does not extract it or execute any member.

`ROOTS` in the exporter is the versioned selection. Schema 1 includes `docs/`,
the documentation registry helper, README/changelog, selected status artifacts,
render examples and generated assets. `bin/souc` and `bootstrap/stage0.c` are
non-executable **source data** used by the website's status synchronizer, not a
compiler distribution. All members have mode 0644. A site consumer can materialize
this input layout beside its website directory; a future dedicated input-root
adapter must preserve the same dependency checks.

The workflow is triggered by exporter/contract changes and every selected input
class, including mapped proof targets. It tests integrity failures and compares
two real exports byte for byte.
Its uploaded artifact is a CI artifact, not a permanent release. Release publication
must preserve the receipt and digest; no release or standalone website deployment
is implied by an export passing CI. The current isolated website proof uses
pre-rendered assets. Regeneration still requires the matching compiler and must
be validated separately before declaring the site migration complete.
