# Canonical BIDI runner and independent archive readback

The canonical runner completed its selected BIDI program in Slurm 19686 with exit 0 (37m59 allocation elapsed). Its unchanged schema-v1 result reports one pass, zero failures and zero skips, in runtime-fixture mode. This is one selected program and its official fixture corpus, not a whole-stdlib result.

The runtime fixture executed all 211 official chunks / 861948 cases using the retained program.elf. All 211 result rows have status 0, no signal and a positive conformance marker. Both controls use that same ELF and exit 1 as required: missing input and a deliberately wrong expected level. RESULTS.json retains the original per-fixture bytes, output_bytes and observed_output_bytes without flattening or renaming. For example, the first fixture is 199928 bytes while its stored and observed output is 46 bytes.

The canonical retained ELF is 1602416 bytes with SHA-256 c41ba3767011413208075bd4f3ddf9aff3fb46ff2c3d370f3f68da7e01419667. The temporary compile output and retained/executed inventory carry that same identity. Native start binds helper 8b35a59e4b63c2b0df0797ec5c57aea7f01c24e6067a23b1334a31ec2e1b4627 and golden b428aba82a8ed4da0b3e21f93a5228761c945d18769900acd153d6e961408eb5. Check, compile and native have separate successful phase receipts; final source/caller and argv identity flags are true.

The actual source scope is preserved verbatim in canonical/SOURCE_SCOPE: base f0e25b1fecbcb99bb9d621b170667c1b1bf0ff79, declared overlay ccc137cf642353502f39a1457b2e28cf5af147ea, runner projection pin fc271161afcd. This receipt worktree starts at root 12be30dd0bd91c366d24e8bea5550c99ba580347; that does not relabel the executed snapshot as the receipt commit. The raw compiler is the recorded experimental raw03 e0279ff62e97e861415cd8b105b854ae645aa61865b33b1e2b91184d2c28bf54.

## Direct and canonical executions

Direct scientific qualification 19618 completed 0 in 36m30 using the original resolver across the official corpus. Canonical qualification 19686 is the later end-to-end runner/runtime-fixture execution with retained ELF, full artifact manifests, native binding and diagnostics finalization. These are separate receipts. The new readback runs neither the Sounio resolver nor the C++ Unicode producer and makes no additional numerical acceptance claim. ICU pristine policy, data, expected levels, source and production scripts were not edited by this lane.

## Independent readback

Export 19708 completed 0 in 4m59. Its durable archive is:

- /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/bidi-canonical-export07/bidi-complete-sources-data-elfs-tooling-attempts.tar.gz
- SHA-256 1f0d5d5f1b8e1a72b5bb083e554cf29e0765d614a84d47e76e53fb10b28cd14d
- 1547845754 bytes.

Independent Slurm 19728 completed 0 in 1m13 on DL380, where OrangeFS was already mounted. It verified the archive hash before and after reading, matched all 137858 tar entries to the export inventory, and extracted only 183 MiB of results/fixtures/metadata. It independently checked the complete 702-file / 433-directory artifact manifest, the 214 official corpus files and their golden hashes, all 211 result rows and both negative controls, original/final input equality, retained ELF binding, and separate stored/observed output sizes. All checks passed. PHASES.tsv is a compact table derived from the original phase JSON; it is not an execve or sbatch-spool trace.

The whole timed canonical command reports wall 37:45.16, user 2237.60 seconds, system 25.50 seconds and maximum RSS 2875164 KiB. This scope includes the runner, compilation and fixture production/execution; it is not a resolver-only memory measurement.

## Preserved failures and recipe provenance

| Job | Preserved outcome |
| --- | --- |
| 19616 | Direct helper refused an unlisted fixture before complete corpus execution. |
| 19654 | Canonical attempt failed because g++ was absent; its later empty-path hash errors remain in the log. |
| 19668 | The Node result-emission control failed a hash assertion; earlier recorded control phases passed. |
| 19670 | Protocol controls received the C++ producer as original-source input: actual SHA 92ea6b4e… versus expected original Sounio b44635d5…. |
| 19672 | Private C++ linking failed because -lpthread was unavailable. |
| 19727 | First independent readback verified hash/census but misplaced extraction due to tar option ordering; missing RESULT.json was refused. |
| 19730 | Compact receipt collection omitted archived log members from its selection and refused the missing member; no scientific execution was involved. |

The archived recipes under recipes/session-copies are identified session copies, not recovered original sbatch spool scripts. SubmitLine records attest submitted commands/paths separately; they do not prove that a later session copy has the exact original spool bytes. The independent readback/collection recipes and source are included with their own tool/content hashes.

Large JSON/manifests are losslessly gzip-compressed where useful. COMPACT_FILES.json records each source path/provenance, original bytes/SHA, storage compression and stored bytes/SHA; the collector verified decompressed identity against every source. The original 211 rows, two controls, input/ELF/artifact manifests, phase start/result/complete records, source scope, resource report and final guards remain available in this compact receipt and the complete durable archive.
