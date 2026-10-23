# Candidate attribution of observed Madaros pathname records

`attribute_madaros_trace.cpp` is a C++23 text-analysis tool for the private trace
format emitted by `observe_madaros_inputs.sh`: numeric task ID prefixes, no
timestamps, `-f -q -yy -xx`, ordinary argv truncated by `-s 0`.

Build on the remote analysis host with:

```sh
g++ -std=c++23 -O2 -Wall -Wextra -Wpedantic \
  scripts/dev/attribute_madaros_trace.cpp -o /private/path/analyzer
umask 077
/private/path/analyzer TRACE NEW_OUTPUT_DIRECTORY OBSERVED_SOURCE_ROOT \
  EXACT_BOOTSTRAP_EXEC_REQUEST_PATH EXACT_GENERATED_SEED_EXEC_REQUEST_PATH
```

This standalone analyzer does not build or execute Sounio. Compiler builds remain
Slurm workloads. No dependencies are downloaded, no tests are implicit, and no
cache configuration is changed. OUTPUT must not already exist; its directory is
restricted to its owner. Keep the parent private and set umask077 for files.

## Contract

- Reconstruct unfinished/resumed records using task ID and syscall name; retain
  first/last source line references. Replaced, mismatched, undecoded and pending
  records have reasoned entries in `unresolved.tsv`.
- A successful exec request starts an epoch, even if its pathname is undecodable.
  Failed exec requests do not change the previous image hint. Exits clear it.
  Epoch counters increase across observed task-ID reuse, but there is no parent
  tree reconstruction or proof of task generation completeness.
- `compiler_image` means an exact successful exec-request pathname supplied by
  the operator. `hash_utility_image` means `/usr/bin/sha256sum`. These are hints
  based on request paths, not verified ELF identity or exclusive process purpose.
  Script execs can involve interpreter handling. Other paths are `other_image`;
  tasks without their own known successful exec remain `unknown`. No inheritance
  is guessed, including children that appear before a clone return.
- A fixed syscall/operand whitelist identifies requested pathname operands.
  General syscall records and their argument contexts remain in `events.tsv`.
  Unsupported pathname interpretations are not enumerated in `unresolved.tsv`;
  zero reconstruction/decode failures does not imply complete syscall semantics.
- Paths and raw record text are encoded as hexadecimal bytes. This avoids TSV
  delimiter injection and assumes no UTF-8 encoding. Strict quoted decoding
  accepts complete `\\xHH` tokens only. No shell evaluation is used.
- `absolute_source_candidates.tsv` includes only decoded pathname operands with
  a literal `SOURCE_ROOT/` prefix; it strips that prefix without normalization.
  It includes successes, failures, metadata and mutations, not just opens.
  Relative paths and dirfd/cwd annotations are preserved as context, not resolved.
  Symlinks, `..`, mount namespaces, CLONE_FS and inherited descriptors are not
  resolved. A candidate is not proof of consumed file contents.
- The `failed` column counts the observed `-1 errno` form. Other results, including
  `exit_group = ?`, are not called successes merely because failed=false.
- Lines are bounded while reading to1MiB, physical records to2million, and current
  image/pending maps to100000 entries at the per-record guard. Candidate and epoch
  maps are bounded indirectly by the record limit. This is a bounded format tool,
  not a constant-memory or hostile-input service. Bound/I/O failure exits70 and
  leaves partial diagnostic output. Usage failure exits64. An analyzer exit0
  permits unresolved records; inspect SUMMARY before making any statement.

## Output and custody

Private: `events.tsv`, `exec.tsv`, `path_operands.tsv`,
`absolute_source_candidates.tsv`, `unresolved.tsv`, and the full `SUMMARY.txt`.
The full summary includes exec-request paths in hex and is also private.
Signals/exits remain in the raw trace; their records count toward complete_records
but do not create pathname-event rows. Therefore role totals are syscall rows,
not all trace records. Preserve the original trace alongside these derivatives.

Public: a redacted summary containing counters and role totals, source/binary/
trace identities, resource receipt, and identities of compressed private tables.
Use deterministic gzip and compare decompressed content with the original table
before declaring custody complete. No public per-path or per-task table is needed.

`candidate_only=true`, `complete_input_closure=false`, `cache_authority=false`.
The broad key and guards remain in force. This tool neither narrows a key nor
proves content closure, numeric concordance or uninstrumented compiler speed.
See the [executed receipt](../../docs/plans/COMPILER_PATH_ATTRIBUTION_EXECUTION_2026_10_22.md).
