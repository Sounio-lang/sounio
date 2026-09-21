# LOOM V13 Kernel Peer Control Matrix

Status: `MATERIAL_BOOTSTRAP_AUTHORIZED_BY_FROZEN_SOUNIO_V13`

## Authority

The sole expected-result authority is:

```text
tools/loom/kernel_peer_authority_plan_v13.freeze.v1
sha256=b3557d850ce0dc13c900f8dbb10c33f824ac25e908cb4a48dd2ef913267194c2
```

C, BPF, and C++20 only realize the frozen matrix. They may falsify it, but may
not revise it. The V12 dumpable/prlimit counterexample remains an immutable
parent.

## Experiment

One diskless, networkless KVM guest boots the named host kernel with only
`lockdown,capability,bpf,ima,evm`. For each of ten frozen operations it emits:

- one active-mediator same-UID treatment;
- one same-process mediator-removal completion;
- one distinct-UID credential control;
- one same-UID caller-seccomp unavailable control;
- one same-UID `dumpable=0` partial control.

Every process pair has distinct PIDs, pidfds, start ticks, and cgroups. The
guardian observes all four UID slots, user namespaces, seccomp mode, target
state, and final extinction. Every observation carries invariant, delta,
attempt, target, and extinction hashes.

## Seccomp control

The seccomp vertex installs an irreversible classic-BPF filter in the attacker
after `pidfd_open`. It returns `EPERM` at the caller boundary for all ten attack
surfaces. That result is `EXPERIMENT_UNAVAILABLE`, never receiver isolation.
The target snapshot must remain unchanged.

## Corrected dumpable control

With the mediator absent and target `dumpable=0`, operations 1-4 and 9 must
complete their typed effects. Operations 5-8 and 10 must refuse before effect.
Any other signature falsifies V13 and preserves `DENY451`.

## Sabotage twins

The five frozen sabotage comparisons are emitted as separate receipts:

1. active mediator to removed mediator, same process epoch, ten crossings;
2. removed mediator to active mediator, same process epoch, ten crossings;
3. distinct UID to same UID, fresh required epoch, ten credential refusals disappear;
4. seccomp-filtered to open caller, fresh required epoch, ten unavailable rows disappear;
5. dumpable zero to one, fresh required epoch, five partial refusals disappear.

Fresh epochs are required for credential and seccomp changes. Their receipts
must name the exact delta and may not claim byte-identical process identity.

## Promotion boundary

A passing host matrix may set `controls_executed=true` and
`material_peer_matrix=true`. It must keep `same_uid_peer_isolation=false` and
action `9025=DENY451` until a new Sounio judgment consumes the certificate.
All product and hook surfaces remain closed. Python and Rust are forbidden.
