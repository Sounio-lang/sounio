# One-shot CPU36 qualification dispatch

Source e91a7a17ea was published before dispatch. Group 3 is restricted to this
repository and ci.yml at that exact full SHA. Runner 6140 was observed online and
idle in that group. Dispatch 34491511493 binds the CPU36 profile and frozen PR
comparison base; no automatic retry is permitted.

The Job has 4 CPUs, 36 GiB memory, no mounted service-account token, no host
namespaces or host mounts, the tested network policy, and an immutable bootstrap
ConfigMap. The runner registration is JIT, with its payload supplied only through
an immutable Kubernetes Secret. No credential payload is in this packet.

This is a launch receipt. Source CI, the full compiler job and Inkling remain
unqualified until their own terminal evidence is collected. Keep this packet as
the launch snapshot and append separate terminal evidence.

After the attempt, preserve logs and results before removing only this attempt's
runner/Secret/Job and group/routing configuration. Do not delete shared resources
or retry a failed job under the same identity.
