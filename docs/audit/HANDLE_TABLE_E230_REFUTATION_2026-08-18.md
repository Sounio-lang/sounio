<!-- docs:meta
topic_id: repo.docs.audit.handle-table-e230-refutation-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.handle-table-e230-refutation-2026-08-18
-->

# E230 handle-ceiling diagnostic — closed by refutation (2026-08-18)

**Status:** CLOSED. The proposed target did not exist. Do not reopen E230
to add `count=` / `4194304` / `(2^22)` to the refusal so a gate has
something to grep.
**Verdict:** the handle table is already fail-closed. Capacity is
4 194 304 (2²²). Overflow prints `madaros: handles full` and
`emit_exit(182)`. There is no E230, no count, no capacity in the
message.
**Not this closeout:** raising the ceiling, reclamation, or a later
optional dispatch that adds `N of M` to the existing line. That would
be a new ticket, not E230.

`E230` in this document means the *proposed handle-table diagnostic*.
It is not the existing raw-pointer error E230 in
`docs/internal/compiler/RAW_FIELD_TERMINAL_CONTRACT.md`. Do not merge
the two.

## Recommendation: B, not A

**A** would add the missing number to the runtime diagnostic, then
write a gate that greps it. **B** records that the wall is already
named and fail-closed, and keeps a one-witness gate that checks the
string that exists.

B is the close. Adding a number so the instrument has a target is
building the target to justify the instrument. The number has modest
operational value (did I really burn 4 M slots, or is the table
corrupt at 100?). That value does not keep E230 open. It can be a
separate, tiny dispatch later: print `N of M` on the line that already
exists, with the 3-slot must-run selftest in front, and without a 90 %
warning or a `runtime_context_size` bump. The v3 patch that tried to
do all of that at once SIGSEGV'd every 3-slot aggregate (grok-cli5,
14-cell matrix). Re-entering that site under the E230 name is how this
lane spent three rounds measuring nothing.

## What was measured (unpatched Madaros, 2026-08-18)

Compiler: default Madaros, `runtime_context_size() = 248`, no
`e230_90` field. 3-slot substrate (`struct { x,y,z: i64 }`, N=1) runs
`rc=0`. Same shape with `Alloc`, N=1000: `rc=0`, stdout `done\n`.

| N | rc | stdout | stderr |
|---:|---:|---|---|
| 1 | 0 | empty | empty |
| 1 000 | 0 | `done\n` | empty |
| **4 194 320** (capacity+16) | **182** | empty | `madaros: handles full\n` (22 bytes) |

STAGE: `/orangefs/training/handle-ceiling-unpatched-20260818T180747Z`.
ELF 12 744 B, `\x7fELF`, compiled with `souc compile`.

The wall has a name. It does not have a number. The E230 gate was
written to grep a diagnostic current Madaros does not emit.

## Why the instrument looked infinitely broken

Three verification rounds, each closing a real instrument defect:

1. Compile without checking an ELF existed (`souc src.sio -o dest`
   wrote a file named `-o`, rc=0).
2. Wrong engine (that bare form is lean_single `SRC OUT`).
3. `run_rc` read through a pipe on empty ELF stdout (`set -o pipefail`
   aborted the rest).
4. Witnesses were 3-slot structs on a compiler that could not
   construct them (the E230 *patch*, not Madaros). Same crash, dressed
   as a measurement.

Those defects were real. They were also what you find when the signal
behind the instrument does not exist. No amount of gate repair makes
`warning[E230]` appear on a binary that never prints it.

## What a gate may claim

`scripts/ci/handle_table_ceiling_gate.sh` now asserts only what was
measured:

- 3-slot aggregate must **run** (`rc=0`) or the gate **refuses to
  measure** (do not emit handle-ceiling numbers from a compiler that
  cannot build structs).
- A loop of capacity+16 handle-consuming allocs must exit **182** and
  print `madaros: handles full`.

It must not grep `E230`, `count=`, or `4194304`. Those strings are not
on the current path.

## Do not reopen as E230

If a later lane wants `count=N of M` on the existing line:

- New dispatch, new name. Not E230.
- Do not bump `runtime_context_size` for a once-per-process 90 % flag.
- Do not emit a 90 % warning unless someone files that as its own
  target, with a substrate selftest in front.
- The 3-slot must-run cell is mandatory. A patch that cannot construct
  `{x,y,z: i64}` must not ship.

## AI disclosure

Measurement, refutation, and closeout by AI agent (grok-cli1) under
human direction, 2026-08-18. GAIDeT-ICMJE 2025.
