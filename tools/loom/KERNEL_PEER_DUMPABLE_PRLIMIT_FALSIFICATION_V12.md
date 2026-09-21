# LOOM V12 Dumpable/Prlimit Material Falsification

Status: `PREREGISTERED_AGAINST_FROZEN_SOUNIO_V12`

## Frozen hypothesis

The Sounio V12 authority plan assigns operation 9, `prlimit64`, to the refused
side of `DUMPABLE_ONLY_CONTROL`. Its frozen totals require four completed and
six refused observations for that vertex. The V12 Garden is less specific: it
states that memory rows may refuse while signal rows complete.

This experiment does not repair that discrepancy. It tests the frozen Sounio
hypothesis as written.

## Material vertex

The KVM microhost creates two different processes with:

- the same real, effective, saved, and filesystem kernel UID `61234`;
- the same real, effective, saved, and filesystem GID `61234`;
- the same minimal `CAP_SYS_NICE` permitted/effective capability;
- different PIDs, pidfds, start ticks, and cgroups;
- the same user namespace;
- attacker `Seccomp: 0`;
- no loaded LOOM BPF policy;
- target `PR_SET_DUMPABLE=0`.

The attacker invokes `prlimit64` on the target, changes `RLIMIT_NOFILE`, reads
the changed value, and restores the original value. A successful syscall alone
is insufficient. The material completion witness requires all three steps plus
a final target snapshot equal to the baseline.

## Decision

```text
frozen_expected = REFUSED_BEFORE_EFFECT
material_observed = EFFECT_COMPLETED | REFUSED_BEFORE_EFFECT
```

If the material observation is `EFFECT_COMPLETED`, V12 is falsified. The result
must remain `DENY451`; the frozen V12 Sounio file and manifest must not be
rewritten. A new Garden and Sounio V13 executable must define the corrected
control before any broader control bytes are produced.

If the material observation is `REFUSED_BEFORE_EFFECT`, the exact errno and
unchanged target snapshot are recorded and V12 remains open for its full
five-vertex matrix.

## Boundaries

- This is one preregistered counterexample probe, not the full control matrix.
- C++20 is transitory `MATERIAL_BOOTSTRAP`, never semantic authority.
- The semantic authority remains the already frozen Sounio V12 artifact.
- Python and Rust are forbidden.
- No result may set `material_peer_matrix`, `same_uid_peer_isolation`,
  `material_coverage`, `complete_effects`, or `claim_ready`.
