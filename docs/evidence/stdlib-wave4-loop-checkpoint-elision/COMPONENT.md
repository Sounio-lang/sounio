# Loop-checkpoint component receipts

Implemented module and component-qualified synthetic IR only. No frontend hook
is applied, no raw bootstrap was run by this lane, and no performance claim is
made. See [design](../../../docs/design/loop-checkpoint-elision.md).

- Final published gate: Slurm19635 COMPLETED0, 7 seconds, 1 CPU/8GiB allocation.
- Modern check raw8abe SHA8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932.
- Legacy positional producer seed9cf SHA9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a.
- check/compile/native0; 98 exact output lines, 97 controls plus final marker.
- ELF SHA63c062399774bb9a44d9870ee2aa781c0663333572e681f20cdd4a0a7163e1f7
  in job19633; published gate19635 records its own ELF hash under component-final.
- All complete source, raw and seed hash readbacks passed.
- Original durable job directory:
  /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/loop-checkpoint09
- Exact worker, stdout/stderr, statuses, source/ELF hashes, sacct and final
  verification readbacks are included in component-final. ELF remains in scratch.
- Earlier component milestones:19590 22cases;19593 extended attacks;19605 cloned
  alias/pool and float/ABI controls;19631 final primitive/descriptor guards;
  19633 module wrapper; all completed0 and are not replacements for real target IR.
- Original modern compile139 is preserved in loop-checkpoint02/results (19584).
  Public frontend capture failed before IrModule in loop-arena01/results (19595).
  Its worker0 is not a target success.

Review-only frontend patch SHA7a452c5e47ee92c8e7928fd8615a375c53d4e826a593d772befd5a29af43b626:
git apply --check against root frontend9d33ff... passed. No compile/native
qualification of that hook is asserted. Source trace Schema2 commits8aee6eaa3f
and7414a51ed2 were delivered separately and root applied the latter in b2a567a1ee.
