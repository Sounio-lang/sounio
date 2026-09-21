# Loom Execution Custody V2

Status: `CUSTODY_AND_OUTCOME_PROOF_ACCEPTED_ATTACHMENT_REFUSED`

Loom Execution Custody V2 removes the execution capability from the shared
filesystem. The Sounio execution authority decides the command before issuance
and again immediately before supervised execution. A second frozen Sounio
authority decides whether the observed outcome is complete enough to commit.
OCaml realizes transport, process identity, one-shot custody, supervision, and
crash revocation. Neither OCaml nor the kernel invents semantic results.

## Authority order

The execution authority remains frozen under:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

The pinned execution-authority manifest is
`d07823382125e668eb0d7afe5d52092de3c58ec5bef9655fa2f1f56a9c84d8c0`.
This successor manifest changes only the frozen selftest receipt hash; the
Sounio source, entrypoint, executable, and semantics hashes remain unchanged.
Python and Rust remain forbidden as Guardian implementations and semantic
oracles. Shell and `jq` appear only in the adversarial test harness as
mechanical transport and structured-data parsing.

## Custody protocol

1. The native hook measures the command, environment, toolchain, hardware, and
   frozen Sounio authority chain.
2. Sounio action `9021` must return `ALLOW` before any grant exists.
3. The hook sends the capability body to `EXEC_ISSUE` over the lane's Loom Unix
   socket. Production does not write a `.cap` file.
4. The single-threaded kernel stores the body only in an in-memory hash table.
   It returns a random 256-bit handle bound to the current kernel generation,
   instance, worktree cwd, and expiry.
5. The replacement command carries only instance, generation, and handle.
   The handle is a lookup key, not sufficient authority.
6. `EXEC_CONSUME` authenticates the peer before looking up or burning the
   handle. An unauthorized probe therefore cannot consume or invalidate it.
7. The kernel removes the grant, creates a generation-bound pending outcome
   obligation, and returns the capability body. A crash after removal can no
   longer be mistaken for success.
8. The broker remeasures the command and environment, reloads the frozen
   policy, and invokes Sounio `9021` again.
9. The broker forks the measured leaf with inherited standard streams, observes
   exit or signal, writes the observation, and asks frozen Sounio action `9022`
   whether the outcome is complete.
10. Only an allowed 9022 decision can produce an outcome receipt and close the
    kernel obligation through `EXEC_OUTCOME`. Duplicate closure is refused.

The previous file-backed capability path remains available only when all three
test controls are explicit: `SOUNIO_LOOM_HOOK_TEST_MODE=1`,
`SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR`, and the agent-hook argument
`--test-file-capability-fixture`. Production hook configuration does not carry
that argument. Leaked test environment variables alone therefore remain on the
kernel-memory route. This is a V1 sabotage fixture, not a production fallback.

## Peer identity

The kernel obtains `pid`, `uid`, and `gid` from Linux `SO_PEERCRED`; the client
cannot supply them. It opens a close-on-exec `pidfd` and snapshots the peer
start tick and PID namespace. At each execution request it revalidates:

- effective UID and GID;
- live `pidfd`;
- unchanged `/proc/<pid>/stat` start tick;
- boot identity and PID namespace;
- exact `/proc/<pid>/exe` path and SHA-256 of the running Loom binary;
- command-line role (`agent-hook` or `exec-capability --handle ...`);
- ancestry reaching the exact harness PID and harness start tick;
- worktree containment, plus exact issued cwd at consume time.

Recovery preserves the user-visible Loom instance and Guardian-owned harness,
but creates a fresh random kernel generation and an empty grant table. A handle
issued before kernel death is therefore unusable after recovery.

The existing session token remains a compatibility credential for the general
Loom protocol. It is not sufficient for `EXEC_ISSUE` or `EXEC_CONSUME` because
the peer checks above are mandatory before grant state changes.

Inherited agentd endpoint variables are also not treated as identity. Loom uses
an agentd endpoint only when its agent, lane, raw session ID, and physical
worktree all exactly match the current hook event. A foreign tuple is ignored;
the matching Loom endpoint remains available as the immediate-delivery path.
All `SOUNIO_AGENTD_*` variables are removed from the environment of mechanical
`sounio-coord` subprocesses; endpoint identity is passed only through validated,
explicit arguments. This prevents an unrelated live agentd from recursively
contending on presence state.

## Durable evidence

The kernel journal records only event kinds and digests for grant issue,
consume, outcome commit, explicit incomplete recovery, refusal, expiry, and
kernel generation. It does not store the command, capability body, or execution
environment. The execution decision log binds the outcome and receipt digests;
the self-contained receipt lives under the shared Git common directory.

The adversarial gate proves:

- a legitimate descendant executes one audited native leaf exactly once;
- a same-UID process outside the exact harness ancestry sees the handle but
  cannot consume or burn it;
- cwd drift refuses without burning the grant;
- replay and expiry refuse;
- no production `.cap` file exists;
- leaked V1 test variables do not enable the file issuer without its explicit
  CLI fixture flag;
- a foreign inherited agentd tuple is ignored rather than registered for the
  current lane;
- kernel crash plus recovery revokes every pending grant;
- normal exit, nonzero exit, and POSIX signal outcomes retain their exact
  observed status;
- outcome replay is refused;
- a crash after receipt creation but before `EXEC_OUTCOME` becomes an explicit
  `INCOMPLETE` event during recovery;
- the digest-only journal remains verifiable across recovery.

Run:

```sh
bash scripts/ci/sounio_loom_execution_custody_selftest.sh
```

## Security boundary

This is lane-process custody, not a claim that one Unix UID separates mutually
hostile principals. A malicious process already inside the same trusted harness
ancestry is in that harness's trust domain. The proof also assumes no successful
`ptrace`, `CAP_SYS_PTRACE`, process injection, or kernel compromise. Strong
isolation between actively malicious same-UID lanes still requires per-lane UID
or user namespace isolation plus LSM/cgroup policy and a non-dumpable Guardian.
Ancestry or cgroup membership alone is insufficient.

## Attachment remains refused

The repository must not yet enable global Exec/Bash interception. Custody and a
durable post-execution result receipt are now proved for the audited leaf
surface. General shell composition remains open: pipelines, substitutions,
redirections, shell functions, and dynamically selected interpreters cannot be
treated as one measured executable without a stronger closure protocol.
Attaching the current hook to arbitrary Bash/Exec would overstate that proof.
The exact outcome contract is in `tools/loom/EXECUTION_OUTCOME_V1.md`.
