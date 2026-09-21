# Loom Execution Capability V1

The execution capability broker is the native OCaml operational realization of
the frozen Sounio frame `9021`. Sounio decides whether a measured request may
cross the pre-execution boundary. OCaml measures, transports, enforces, and
records that decision; it does not define the language roles or expected
result.

## Authority Chain

An execution request follows this order:

1. the native hook parses the bounded JSON event and extracts exactly one
   command field;
2. the broker lexes a static command, resolves the executable through `PATH`,
   canonicalizes symlinks, hashes the executable and command, classifies the
   producing language, and measures the execution closure;
3. the broker revalidates the pinned execution-authority manifest, Sounio
   source, entrypoint, semantic bundle, and Sounio runtime;
4. the broker submits the complete frame `9021` to Sounio;
5. only an exact Sounio `ALLOW` creates a random 256-bit, mode-`0600`, expiring
   capability outside Git;
6. the hook removes the original command from `updatedInput` and returns only
   the absolute broker path and opaque token. Both Codex and Claude consume
   this through the strict `hookSpecificOutput` PreToolUse envelope with an
   explicit `permissionDecision=allow`;
7. consumption atomically renames the capability before validation, so every
   attempt burns the token;
8. the consumer revalidates uid, root, cwd, expiry, environment, hardware,
   command, argv, executable, broker, authority chain, frame, and decision;
9. Sounio decides the unchanged frame a second time;
10. the broker unlinks and directory-`fsync`s the capability before `execve`.

Policy absence, timeout, malformed or dynamic commands, executable drift,
environment drift, replay, expiry, record tampering, runtime drift, and any
Sounio denial fail closed before the measured executable runs.

## Capability Receipt

The private capability binds:

- token, issuance and expiry times, uid, repository root, and cwd;
- original command and parsed argv;
- canonical executable path and executable hash;
- broker hash;
- frozen manifest, Sounio source, and semantics hashes;
- producing language and derived language role;
- hardware record and hash;
- versioned environment record and hash;
- exact Sounio frame and exact Sounio decision;
- a digest over the complete record.

Mode `0600` isolates other Unix identities, not peer lanes running under the
same uid. The trailing SHA-256 is an integrity checksum, not a keyed
authenticator: a hostile same-uid peer that can enumerate the directory can
read a token, rewrite the record, and recompute that checksum. V1 therefore
does not claim cross-lane capability custody.

The append-only decision journal records `ISSUE` and `CONSUME`, `ALLOW` or
`DENY`, the reason, and the same authority chain. It logs only the environment
hash, not environment values. The pre-execution journal deliberately says
`execution_result=pending`: an `ALLOW` receipt is not an execution outcome.

## Environment Boundary

The environment record is `loom-execution-environment-v1`. It binds stable
process inputs such as `PATH`, identity, locale, shell, temp and timezone
settings, along with all `SOUNIO_`, `SOUC_`, `LOOM_`, `OCAML`, `CAML_`, `LC_`,
`LD_`, `DYLD_`, and `XDG_` variables. Fixed security-sensitive variables are
represented explicitly even when absent.

The broker refuses nonempty shell-startup, locale-loader, or dynamic-loader
injection controls, including `BASH_ENV`, `ENV`, `GCONV_PATH`, `LOCPATH`,
`NLSPATH`, `ZDOTDIR`, `LD_*`, and `DYLD_*`. This lets a normal shell bridge
preserve the measured environment without accepting a startup script or
loader injection as an invisible child execution. The Sounio authority runtime
and the authorized leaf receive a newly constructed environment containing
only these bound entries; unmeasured harness variables are not inherited across
the execution boundary.

## Initial Closure

V1 attests only four root-owned, non-group-writable, non-world-writable ELF
leaves resolved under `/usr/bin`:

- `true`
- `false`
- `printf`
- `pwd`

Every other native executable is `closure_attested=0` and Sounio refuses it
with code `227`. Shell interpreters and commands with metacharacters are refused
with code `226`. Python and Rust are refused with code `210`, including symlink
aliases, direct shebang interpreters, and targets forwarded through
`/usr/bin/env`. General forwarding wrappers such as `timeout`, `xargs`, `nohup`,
`nice`, `setsid`, and `stdbuf` are permanently measured as dynamic and refused
with code `226`; adding one to a leaf filename list cannot silently attest its
child closure. Git commit is not yet attached and its current unattested closure
is refused.

This intentionally small closure proves the one-time enforcement mechanism. It
is not yet an ergonomic general command runner.

## Required Custody Upgrade

The attachment successor moves grants into the replaceable Loom kernel rather
than moving the same bearer file behind an existing socket. The minimum
protocol is:

1. `EXEC_ISSUE` stores a short-lived grant only in kernel memory, bound to
   agent, lane, session, instance, harness pid/start tick, kernel generation,
   boot id, pid namespace, root, cwd, environment, hardware, command,
   executable, and Sounio authority digests;
2. the returned handle is a lookup key, not authority by possession;
3. the Unix listener captures Linux `SO_PEERCRED`, pins pid identity, and walks
   `/proc` ancestry to the exact Loom-owned harness pid and start tick;
4. only an authenticated broker descendant can atomically move `Issued` to
   `Consuming`; wrong-ancestry probes cannot burn another lane's grant;
5. the broker remeasures, obtains the second Sounio decision, deletes the
   in-memory grant, and executes with the reconstructed environment;
6. kernel death revokes every outstanding grant. Grants are never recovered
   across a generation change.

`SO_PEERCRED` plus ancestry assumes peers cannot ptrace or inject into one
another. A claim against actively hostile same-uid principals additionally
requires per-lane uid or user-namespace isolation and an LSM/cgroup policy;
ancestry alone is not that security boundary.

## Gate

`scripts/ci/sounio_loom_execution_capability_selftest.sh` proves:

- hook-to-shell-to-broker-to-`execve` round trip for an audited leaf;
- exact Codex `cmd/workdir` and Claude `command` PreToolUse output envelopes;
- original-command removal and private capability permissions;
- complete authority, hardware, environment, command, and executable binding;
- replay, record tamper, expiry, cwd drift, broker drift, and environment drift
  refusal with token consumption;
- missing/tampered policy and runtime refusal;
- unreadable-policy refusal with an explicit execution-journal `DENY`;
- shell-startup and dynamic-loader injection refusal;
- Python and Rust refusal through direct names, symlink aliases, shebangs, and
  `/usr/bin/env` forwarding;
- structural refusal of general command-forwarding wrappers;
- dynamic shell, missing executable, non-leaf native tool, and Git commit
  refusal;
- executable sentinels proving Python and Rust did not run;
- durable ISSUE/CONSUME ALLOW/DENY decisions;
- explicit `execution_result=pending`.

The Sounio execution-authority gate separately runs 32 expected-result cases
and a causal sabotage control: removing only the Python prohibition changes the
unchanged Python frame from DENY to ALLOW.

## Attachment State

```text
exec_attached=false
child_exec_attached=false
commit_attached=false
ci_attached=false
post_execution_receipt=false
shell_bridge_direct_exec=false
same_uid_peer_isolation=false
capability_custody=file
record_authentication=unkeyed-sha256
parity_open=false
claim_ready=false
```

The broker code recognizes execution events, but the checked-in Codex and
Claude hook matchers remain structured-write-only. Exec/Bash attachment waits
for an in-memory persistent custodian that atomically owns tokens and verifies
the consuming peer's process ancestry/generation, a broader attested closure
strategy, and a durable post-execution outcome receipt. Child execution,
commit, and CI require their own adapters and negative fixtures; none inherit
authority from this proof.
