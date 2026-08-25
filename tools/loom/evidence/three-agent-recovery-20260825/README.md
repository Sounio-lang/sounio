# Loom Three-Agent Recovery Canary

Run ID: `20260825T221959Z-685009`

This retained canary tested three real agent CLI processes under independent
Loom Guardians: Codex, Grok, and MiniMax through OpenCode. The hypothesis,
control, and acceptance criteria were written to `prereg.json` before any kernel
was destroyed. All three native processes were launched through the same
`loom-provider-abi-v1` `provider-start` surface with explicit context isolation
and unsafe-auto policy.

## Result

- PASS
- disposable kernels destroyed: 3
- replacement kernel PIDs observed: 3
- Guardian, CLI, and Loom instance identities preserved: 3 of 3
- physical tool receipts: 3
- tokens recovered from durable replay: 3
- sequential recovery interval: 548 ms

The physical receipt is the control against a model merely printing a success
claim. The canonical `sounio_loom_selftest.sh` separately mutates one byte of an
exited session's `output.bin` without changing its length and requires the
specific `guardian-output:digest-mismatch` refusal.

## Verify

```sh
cd tools/loom/evidence/three-agent-recovery-20260825
sha256sum -c sha256.txt
```

`pre-crash-status.txt`, `kernel-absent-status.txt`, and
`post-recovery-status.txt` preserve the process and generation identities.
`snapshots/` preserves the exact replay witness for each CLI in deterministic
gzip form; use `gzip -dc` to inspect one without rewriting it.

## Boundary

This is a bounded, single-Unix-host recovery result with the Guardians,
processes, and durable files still alive. It is not evidence of recovery from
Guardian loss, host loss, storage loss, credential loss, provider outage, or a
network partition. It does not establish exactly-once tool-effect execution or
prove a broader PL/CS novelty claim. Replaying an older but internally consistent
`output.bin` and Guardian journal pair is a separate monotonic-freshness problem;
this canary does not claim to solve it.
