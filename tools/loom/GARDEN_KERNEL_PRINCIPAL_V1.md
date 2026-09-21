# Garden: Kernel Principal Extrusion V1

Status: `PREREGISTERED`

## Question

Can LOOM turn mutually hostile lanes that begin under one Unix account into
kernel-distinct execution principals, without treating a bearer secret,
ancestry, cgroup membership, or an LLM assertion as proof of isolation?

## Authority Order

This experiment follows the repository language-authority contract:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

Sounio action `9026` is the first executable definition of the certificate and
its expected decisions. OCaml may later transport and verify material facts.
Linux, subordinate-ID helpers, namespaces, cgroup v2, and a native attack probe
may realize or measure those facts. None may define the expected decision.
Python and Rust are forbidden. External LLMs are review-only.

## Hypothesis

A lane is isolated from a hostile process that started with the same outer UID
only if it is extruded into a disjoint kernel principal and every execution
grant is bound to that principal. A valid certificate requires all of:

1. a non-overlapping subordinate UID and GID allocation for exactly one lane;
2. exact kernel `uid_map` and `gid_map` receipts, with `setgroups` denied before
   the GID map is installed;
3. the lane's effective kernel UID and GID differ from the outer account and
   from every sibling lane principal;
4. the execution kernel observes that principal through `SO_PEERCRED` and also
   binds `pidfd`, process start tick, boot identity, PID namespace, exact
   harness ancestry, worktree, and cgroup-v2 domain;
5. the lane and Guardian have no ambient or bounding `CAP_SYS_PTRACE`, are
   non-dumpable, and cannot regain privilege through `execve`;
6. a hostile process under the original outer UID is denied `ptrace`,
   `/proc/<pid>/mem`, grant consume, and grant burn;
7. a sibling lane principal is denied consume and burn even when it knows the
   handle and socket path;
8. a process with the right subordinate UID but outside the exact harness
   ancestry is denied without burning the grant;
9. grants exist only in kernel memory, transition atomically
   `ISSUED -> CONSUMING -> BURNED`, and cannot be replayed;
10. kernel or Guardian loss revokes every unconsumed grant and materializes
    every consumed-but-uncommitted outcome as incomplete;
11. all positive and negative observations are hash-bound to the frozen Sounio
    source, semantics, allocation, namespace maps, toolchain, hardware,
    command, and result;
12. independent single-rule sabotages prove that the Sounio rule under test is
    what refuses each unsafe certificate.

The kernel principal is the security identity. Lane names, cgroup paths,
ancestry, socket paths, and handles are bindings to that identity, never
substitutes for it.

## Certificate Frame

The action `9026` frame is decimal and fixed-width. It contains:

- action and stage;
- parent freeze, Linux, and x86_64 flags;
- subordinate allocation, UID-map, GID-map, setgroups, and disjointness flags;
- outer/principal/sibling UID and GID values;
- user, PID, mount, and cgroup-v2 domain flags;
- `SO_PEERCRED`, `pidfd`, start-tick, boot, executable, ancestry, worktree, and
  cgroup binding flags;
- non-dumpable, no-new-privileges, capability-drop, ptrace denial, proc-mem
  denial, and cross-principal signal denial flags;
- in-memory, atomic, one-shot, non-burning-refusal, and crash-revocation flags;
- sabotage count and required sabotage count;
- nonzero canonical digests for the `9025` parent, allocation, UID map, GID
  map, peer measurement, attack observations, Sounio source, frozen semantics,
  toolchain, hardware, command, and result.

## Decision Order

The Sounio authority evaluates rules in this order:

1. malformed frame or non-boolean flag -> `DENY424`;
2. wrong action/stage or missing frozen `9025` parent -> `DENY405`;
3. unsupported platform or namespace/cgroup substrate -> `DENY454`;
4. missing or overlapping subordinate allocation -> `DENY455`;
5. invalid UID/GID maps or `setgroups` ordering -> `DENY456`;
6. outer, lane, or sibling kernel principals are not pairwise distinct ->
   `DENY457`;
7. peer identity or harness/cgroup binding is incomplete -> `DENY458`;
8. dumpability, privilege, ptrace, proc-mem, or signal isolation is incomplete
   -> `DENY459`;
9. grant custody, atomic burn, non-burning refusal, or crash revocation is
   incomplete -> `DENY460`;
10. sabotage evidence is incomplete -> `DENY461`;
11. provenance is incomplete -> `DENY462`;
12. otherwise -> `ALLOW`.

The order is load-bearing. A platform without material subordinate-ID mapping
must stop at `454` or `455`; later ancestry or grant facts cannot launder it.

## Causal Sabotage Controls

The gate must run an unchanged unsafe witness against a derived source in which
exactly one refusal rule is removed:

1. remove only principal distinctness; equal outer and lane UIDs become
   `ALLOW`;
2. remove only ptrace/proc-mem isolation; an injectable lane becomes `ALLOW`;
3. remove only non-burning refusal; an outsider able to destroy another
   lane's grant becomes `ALLOW`;
4. remove only crash revocation; a pre-crash handle becomes `ALLOW`;
5. remove only sabotage completeness; an unsabotaged certificate becomes
   `ALLOW`.

Each control is valid only when the unchanged frozen source returns the named
DENY and the one-line derived source returns ALLOW. Text matching, mock return
codes, or changing the witness are not causal controls.

## Current Material Probe

On the preregistration host:

- outer identity is UID/GID `1000`;
- unprivileged user namespaces are enabled;
- `/etc/subuid` and `/etc/subgid` allocate `100000:65536`;
- cgroup v2 and Yama `ptrace_scope=1` are present;
- `unshare` and Bubblewrap are present;
- `newuidmap` and `newgidmap` are absent;
- a plain `unshare --map-root-user` maps namespace UID/GID zero back to host
  UID/GID `1000`, so it does not create a distinct kernel principal.

Therefore the current host is expected to DENY material admission until a
reviewed subordinate-ID realization exists. Installing packages or privileged
helpers is not implicit authorization to claim success.

## Acceptance Gates

The semantic gate must prove:

- a fully specified synthetic certificate is `ALLOW`;
- the current material-shaped certificate is `DENY455`;
- equal principal, missing maps, injectable peer, incomplete grant custody,
  incomplete sabotage, missing provenance, and unsupported platform reach their
  exact named decisions;
- all five causal sabotage controls admit their unchanged unsafe witness;
- deterministic source-built execution uses the pinned Sounio toolchain;
- no Python, Rust, shell, OCaml, or external LLM computes an expected result.

A later material gate must additionally prove, with real processes and kernel
receipts, that the outer-UID attacker, sibling principal, and right-UID/wrong-
ancestry attacker all fail before grant state changes. That later gate may set
`same_uid_peer_isolation=1` in action `9025`; this Garden alone may not.

## Nonclaims

- This Garden does not claim that the current host is isolated.
- User namespaces that map back to the original host UID do not satisfy it.
- Cgroup membership, PID namespaces, ancestry, non-dumpability, or Yama alone
  do not satisfy it.
- Root, `CAP_SYS_PTRACE` in the target namespace, kernel compromise, and
  physical-host compromise remain outside the hostile-lane threat model.
- Availability is not confidentiality or integrity: an outer same-UID process
  may still be able to cause denial of service until a service-manager boundary
  is materialized.
- No general Exec/Bash, commit, or CI attachment is authorized.

Claims-Forbidden: same-UID bearer possession proves isolation; ancestry alone proves principal identity; a user namespace mapped to the outer UID creates a new kernel principal; cgroup membership prevents ptrace
Assumptions: Linux exposes subordinate-ID mapping and a kernel-enforced peer identity to the outer LOOM kernel
Measurements: exact uid/gid maps, SO_PEERCRED, pidfd/start tick, namespaces, cgroup, capabilities, dumpability, ptrace/proc-mem/signal attacks, grant state, crash recovery, receipts
Budget: one Garden, one Sounio action, one deterministic semantic gate, then a separately reviewed material realization
Stop-Rule: any overlap, missing kernel receipt, privilege regain, successful injection, burn by an unauthorized peer, stale grant after crash, or noncausal sabotage keeps admission denied
Negative-Witness: outer-equals-principal, overlapping sibling, map omission, ptrace success, proc-mem success, outsider burns grant, crash preserves grant, missing receipt hash
Owner: codex-1/loom-subprocess-membrane-20260828
