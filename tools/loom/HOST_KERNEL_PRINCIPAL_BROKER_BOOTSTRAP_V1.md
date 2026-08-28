# LOOM Host Kernel-Principal Broker Bootstrap V1

Status: `TRANSITORY_BOOTSTRAP_MATERIAL_REFUSED`

## Purpose

This bootstrap creates the narrow host boundary required by frozen Sounio
action `9027`. It does not replace the resident OCaml LOOM kernel, the in-memory
`ExecGrant` protocol, or action `9026` kernel-principal evidence. Its only
intended production responsibilities are:

1. prove that the broker was activated by the host service manager outside the
   lane's privilege boundary;
2. maintain a root-owned, hash-chained principal-lease journal;
3. collect material receipts and submit the fixed action `9027` frame to the
   frozen Sounio executable before releasing a lane harness;
4. force uncertain leases to `QUARANTINED` after broker loss;
5. refuse range reuse until Sounio admits the affirmative extinction triple.

The bootstrap is C++20 because current Sounio cannot yet implement every Linux
socket, namespace, cgroup, pidfd, and durable-file primitive needed here. C++
is `MATERIAL_PARITY`, never `SEMANTIC_AUTHORITY`. The destination remains a
broker compiled in Sounio when the required system surface exists.

## Frozen Authority

The broker must pin and verify:

- `kernel_principal_lease_authority.freeze.v1` by SHA-256;
- action `9027`, stage `SEMANTICS_FROZEN`, producer `Sounio`, role
  `SEMANTIC_AUTHORITY`;
- the exact Sounio authority executable hash recorded by that manifest;
- parent action `9026` and its frozen manifest binding.

Manifest error, authority-executable drift, malformed Sounio output, timeout,
or any non-`ALLOW` decision fails closed. The broker never contains an expected
`ALLOW` result for a material operation and cannot promote a C++ receipt into a
semantic decision.

## Unforgeable Activation Boundary

Production `--serve` accepts only systemd socket activation. It refuses before
opening the lease journal unless all of the following hold:

- real and effective UID/GID are root;
- PID 1 is `systemd` and the broker's parent is PID 1;
- `LISTEN_PID` names the broker and `LISTEN_FDS=1`;
- inherited descriptor 3 is an `AF_UNIX` listening socket;
- the socket path is exactly the configured `/run/sounio` path;
- the socket inode and parent directory are root-owned and not group/world
  writable;
- the broker was not entered through `sudo`, `doas`, or another interactive
  privilege-regain environment;
- the frozen manifest and authority executable are regular, non-symlink,
  root-owned files with no group/world write permission; the installation gate
  applies the same ownership rule to the systemd environment file.

The broker never binds its own production socket. A lane cannot manufacture
the service-manager receipt by invoking the binary directly, even if the outer
account currently has passwordless `sudo`.

Accepted peers are rechecked with `SO_PEERCRED`. Bootstrap V1 accepts only a
root host controller. Lane processes never connect directly and never receive
broker authority.

## Lease Journal

The append-only journal is opened with `O_NOFOLLOW`, `O_CLOEXEC`, mode `0600`,
an exclusive `flock`, and regular-file/link-count/ownership checks. Each record
contains:

- schema and strictly increasing sequence;
- broker epoch and lease generation;
- lease identifier and numeric lifecycle state;
- allocated UID/GID range;
- prior-record digest;
- SHA-256 of the complete canonical body.

Every append is followed by `fsync`. Replay verifies the full digest chain,
monotonic counters, fixed-width field count, decimal bounds, known states, and
the frozen transition subset. Truncation, duplicate sequence, unexpected
transition, or digest mismatch refuses startup.

The C++ transition subset is defense in depth. A transition can be appended in
production only after the frozen Sounio authority admits its action frame.
Sounio remains the source of expected decisions.

On recovery, any last state other than `FREE` or `QUARANTINED` is appended as
`QUARANTINED` under a new broker epoch before the service can answer `READY`.
Recovery never appends `FREE`.

## Bootstrap Protocol

The root-only socket protocol is newline-delimited and capped at 4096 bytes:

- `STATUS` returns frozen hashes, broker epoch, and counts by state;
- `LAUNCH ...` is refused in Bootstrap V1;
- `RECYCLE ...` is refused in Bootstrap V1.

The last two refusals are intentional. Namespace creation, cgroup delegation,
irreversible privilege drop, inter-principal attacks, and extinction receipts
must be implemented and run on the host before material operations are opened.
An unimplemented operation is not a degraded fallback.

## Diagnostic Modes

`--diagnose` is read-only. It measures the current activation boundary, builds
the current material-shaped `9027` frame, executes the hash-verified Sounio
authority directly without a shell, and prints the decision and a SHA-256-bound
receipt. On the current pod it must return `DENY463`.

`--selftest-journal PATH` is a non-root, test-only constructor. It requires a
new path, writes a complete deterministic lifecycle ending in `FREE`, replays
it, and exits. It cannot open an existing production journal. `--verify-journal`
is read-only and is used to prove that a one-byte mutation is refused.

## Negative Controls

The bootstrap gate must prove:

1. current `--diagnose` reaches Sounio `DENY463`;
2. direct non-root `--serve` refuses before journal access;
3. `sudo -n --serve` also refuses because service-manager activation is absent;
4. environment variables claiming root/systemd authority do not bypass checks;
5. one-byte manifest drift refuses before Sounio execution;
6. a fake or modified authority executable refuses by hash;
7. one-byte journal drift refuses replay;
8. `LAUNCH` and `RECYCLE` remain closed in the bootstrap protocol;
9. two source-fresh C++ builds are identical and have no Python or Rust runtime
   dependency.

## Material Opening Gate

Bootstrap V1 may be promoted to material operations only on a host where all of
these are executed, not simulated:

- systemd socket activation and root-only controller peer;
- disjoint UID/GID mapping and cgroup creation;
- irreversible drop with no outer privilege regain;
- outer, sibling, and wrong-ancestry attacks denied before grant lookup;
- broker kill at every lifecycle edge;
- restart quarantine and affirmative process/namespace/authority extinction;
- action `9026` `ALLOW` followed by action `9027` `ALLOW`;
- causal sabotage proving each material rule is load-bearing.

Until that gate passes:

`material_broker=false`, `same_uid_peer_isolation=false`,
`exec_attached=false`, `commit_attached=false`, `ci_attached=false`.

## Nonclaims

- This bootstrap does not create a kernel principal in the current pod.
- A valid journal does not prove a valid namespace or cgroup.
- Root execution without service-manager activation is refused.
- Passwordless `sudo` is a blocker, not a broker installation mechanism.
- The service units are installation inputs, not evidence that systemd is
  present or that the service ran.
- General Exec/Bash, commit, and CI attachment remain refused.
