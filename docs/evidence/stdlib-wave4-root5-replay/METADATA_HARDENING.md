# Metadata-only hardening controls

Slurm19795 completed0 in1s,1CPU/512MiB. No Sounio or C++ compiler, scientific
test, replay or generated native program ran. Only shell/Perl file-metadata
controls were executed in a fresh synthetic directory.

The057162e8db baseline is preserved. Its extracted inventory helper emits
literal backslash separators instead of NUL records. Its verify_binding can
return success after a SHA/CD failure when called through if!, because Bash
suppresses errexit in that context. The old inline RAW_READY check is projected
verbatim into a fixture wrapper for its format controls.

The revision makes these changes only:

- single-backslash find printf escapes, producing real tabs and NUL separators;
- explicit failure returns for every verify_binding subshell/pipeline, including
  explicit cd failures, independent of errexit;
- canonical RAW_READY byte comparison, rejecting extra lines, blank lines,
  missing final newline, embedded NUL and extra fields.

All16controls pass after. The baseline has5passes and11false acceptances/failures.
The same-path byte mutation retains the exact inventory bytes/mode/link records
and is rejected specifically under if! verify_binding. The output preserves the
failed SHA evidence, demonstrating that a later successful inventory comparison
cannot mask it.

These are metadata-helper qualifications only. The96-phase C++ census and replay
remain prepared and unexecuted. ROOT5 main19789 failed admission and has no
RAW_READY; its emitted ELF is not an authorized replay target.
