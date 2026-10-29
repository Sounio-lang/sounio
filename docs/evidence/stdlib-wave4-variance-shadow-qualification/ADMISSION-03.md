# Admission revision03: exact tree graph including dangling symlinks

This supersedes the blanket symlink refusal in revision02. It changes no source,
compiler, SDK, mathematical input or numerical oracle.

sounio.sealed-tree.v2 manifests contain root_mode and a strictly ordered nodes
array. All paths in the source/SDK tree are included: regular files have type,
mode, bytes and SHA-256; directories have type and mode, including empty dirs;
symlinks have type, lstat mode, literal target byte count/SHA-256 and target_base64.
Targets are read with readlink as bytes and NEVER followed. Dangling or external
targets remain literal Git graph entries and are admitted unchanged. No target
materialization is requested. Special nodes remain explicitly unsupported.

Before/after admission re-censuses every node and compares path/type/mode and
file bytes/hash or exact symlink target bytes. Added/removed empty dirs or links,
changed link targets and chmod drift are failures. treeManifest(root), exported
from admission.mjs, constructs this format without following symlinks. Store its
JSON outside the sealed tree to avoid self-reference.

The manually supplied root proof now also pins an adapter file and original_inputs
with distinct BUILD_STATUS_SOURCE, FINAL_SOURCE_VERIFY, FINAL_SDK_VERIFY and
ENDED_SOURCE roles. These original receipts are copied alongside the derived
BUILD_STATUS0/ISO/final-guard inputs, before and after execution.

The derived zero-status files and normalized UTC timestamp are an ADAPTER contract,
not a claim about the raw build-output syntax. The actual root adapter must first
recheck the original compiler/source/SDK final textsha receipts against their
bound tree and compiler, preserve their bytes and failures, then produce the
derived status/ISO inputs. Root manually authorizes the resulting graph binding.
The reader verifies identity and structure; it does not invent a successful build
or turn unparsed historical log text into proof. Merely repackaging arbitrary text
is not an authorized adapter.

The synthetic controls never execute the adapter, compiler or any ELF. They test
reader admission/refusal only, including valid dangling links, changed/added/
removed links, modes/directories, original-input identity and adapter identity.
19790 numerical oracle remains unchanged and is not rerun.
