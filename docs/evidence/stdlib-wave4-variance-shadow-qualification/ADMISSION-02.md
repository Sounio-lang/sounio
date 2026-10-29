Historical revision02: symlink policy below is superseded by ADMISSION-03.md.

# Admission revision 02: metadata only

The first commit df75a6a5a1 and its19790 C++23 receipt remain unchanged history.
This revision executes no Sounio compiler and does not qualify the proposed
variance-shadow patch.

Manual root authorization is still required. A JSON envelope must provide:

- authorized_by: root; patch_sha256; exact candidate lower_sha256.
- raw, source and sdk: absolute normalized paths to the admitted immutable inputs.
- raw_sha256; raw_ready and raw_ready_sha256. RAW_READY must contain exactly the
  sha256sum form RAW_SHA, two spaces, exact raw path, optional final newline.
- source_manifest and sdk_manifest: objects {file,sha256}, identifying external
  full-tree JSON manifests with schema sounio.sealed-tree.v1 and a nonempty files
  list of {file,bytes,sha256}, lexicographically sorted with no duplicates.
- root_proof: {file,sha256}, a manually supplied terminal compiled-graph proof.

The terminal proof schema is sounio.compiled-graph.root-proof.v1. It requires
finalized:true, exact raw/raw_sha256 and source_manifest_sha256/sdk_manifest_sha256.
It includes pinned {file,sha256} entries for build_status, ended_utc and a nonempty
final_guards list. BUILD_STATUS must be literal0; ENDED_UTC must be an ISO UTC
terminal timestamp; every final guard must be literal0. These are root's captured
terminal inputs, not a permission or proof manufactured by the harness.

Full-tree manifests must live outside the sealed source/SDK trees. Every regular
file in each tree must be listed, byte/hash verified and recounted both before and
after execution, including failure paths. Symlinks and special nodes are refused;
supply a reviewed, materialized immutable SDK snapshot if its original layout has
symlinks. This conservative admission is intentional. All root proof inputs,
manifests, RAW_READY and before/after admission receipts are copied into results.

Successful check/compile additionally requires no explicit error/fatal/E-code
diagnostic in stdout or stderr. Exit0 alone is insufficient. Before native, a
regular compiled file must have ELF magic. Its bytes/hash are recorded before and
after native execution, including ordinary native failure.

Canonical original Sounio tests retain their own runtime assertions and optional
markers. Their independent C++ oracle sources are pinned but are NOT executed by
this draft; no oracle agreement is implied. Only closed_integer.sio is compared
to the new independently executed C++23 numerical reference.

19792 ran30 synthetic metadata controls under Slurm: complete admission accepted;
malformed/wrong RAW_READY, source/SDK drift or extra/missing inputs, graph/build/
terminal/guard failures, symlinks, diagnostics with exit0 and ELF identity controls
were refused as expected. No compiler or ELF was executed. Before/after source
guards agree. The initial19791 failed before script/log launch because the new
task directory had root ownership; it was corrected only within that task dir and
resubmitted as19792. No scientific result is attached to either job.
