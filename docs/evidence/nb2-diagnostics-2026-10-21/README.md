<!-- docs:meta
topic_id: repo.docs.evidence.nb2-diagnostics-2026-10-21.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.nb2-diagnostics-2026-10-21.readme
-->

# NB2 diagnostic execution evidence

Frozen executable source: `4a65e767af7ce955cfde85f25484ea7f8a5f44f2`.
Incorporated remote main: `085da1da15f3c05946b66e4ddf7b60a0e29d5d40`.

- `build`: Slurm15783 current-source compiler rebuild, four explicit typechecks,
  native compilation and example execution, COMPLETED0:0.
- `profile`: Slurm15785 execution of the exact identified native ELF under GNU
  time, COMPLETED0:0. One eight-observation whole-program resource observation.
- `compiler`: new compiler build identity and readback of the matching existing
  t560 ext4 custody. Large compiler ELF not copied into this Git evidence set.
- `figure`: original stdout, compressed native ELF, exit/compilation identities,
  extracted report/HTML/SVG and full1380x620 PNG rendered by macOS sips.
- Submission scripts/dispatchers: exact package-only Forge transport definitions.
- `FILES.sha256`: relative integrity manifest for these retained evidence files.

Raw logs and original input manifests are compressed losslessly with gzip -n.
The original uncompressed SHA256 values are retained beside them. Decompress with
`gzip -dc build/check-run.log.gz` or `gzip -dc build/INPUT_MANIFEST.sha256.gz`.
Input-manifest paths refer to the original packet; FILES.sha256 refers to this
retained directory. Native profile stdout is also compressed, with its original
hash retained. Byte identity is not a scientific numerical comparison.

The native stdout contains report text followed by HTML. Standalone HTML is the
complete document suffix; SVG interior bytes are unchanged, with a trailing
newline in the standalone file. The complete sips PNG was inspected after a
QuickLook thumbnail clipped the wide layout. No Canvas pixel capture/comparison.

No tests were added or run. Four typechecks and one real native example do not
establish a corpus run, independent numerical agreement, boundary coverage,
inferential calibration, a comparative benchmark, broader CI or merge.
See the [execution receipt](../../plans/NB2_DIAGNOSTICS_EXECUTION_2026_10_21.md).
