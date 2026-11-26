The committed lean_single seed now builds Madaros (`self-hosted/compiler/main.sio`)
with no `error:` line; on main at `fd59aa3ecb` it printed 32, all in
`self-hosted/check/check.sio`, exited 0 and wrote the ELF anyway. Three sites:

- `ir/serialize.sio`'s private byte helpers `read_i64`/`write_i64` are renamed
  `soir_read_i64`/`soir_write_i64`. The seed resolves function names across the
  whole bundle, so the raw-memory probe `checker_current_fn_name_roundtrip`
  called them instead of the `read_i64`/`write_i64` builtins (27 errors);
- the loop-break state is saved in a `LoopValueSaved` struct or in separate
  locals, not in an `(i64, bool, TypeEntry)` tuple, whose `.2` the seed refused
  and loaded from the wrong slot (4 errors);
- the by-value `Checker::check_expr` match gains the `ExprPerform` and
  `ExprResume` arms it lacked (`match must be exhaustive`; their type checks stay
  on the in-place path).

On 334 `tests/run-pass` and `tests/compile-fail` programs, stdout and exit codes
of a Madaros built from this source equal those of one built from main. The new
`seed_builds_madaros_without_errors` gate fails on any `error` line.
