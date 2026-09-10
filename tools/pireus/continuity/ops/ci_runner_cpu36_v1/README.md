# Ephemeral Current-Source CPU profile, version 1

Prospective resources: 4 CPU cores and 36 GiB memory, requested and limited
equally; 24 GiB ephemeral disk requested, 28 GiB limit, 20 GiB work directory.
The preflight deadline is 20 minutes. The runner lifetime ceiling is 180 minutes;
a future workflow must set its own smaller bounded job timeout. No retries.

Ubuntu 24.04 amd64 image is pinned by manifest digest. The GitHub runner 2.337.0
archive is pinned by SHA-256 from the official release API. System dependencies
are installed by the verified runner's own dependency script. Installed package
versions are recorded; apt repository state is not frozen.

A dedicated namespace, pod-selected ingress denial and public IPv4 HTTP/HTTPS
egress rule isolate the runner. Private, Tailnet, metadata, loopback and multicast
ranges are excluded. DNS is allowed only to kube-system kube-dns pods.
No service-account token, host namespace, host path, Docker socket or PVC is
mounted. Bootstrap installs packages as root with narrowly listed capabilities;
the runner then executes as UID/GID 1000 with an empty bounding set and
no-new-privileges. Work storage is ephemeral. This configuration is not yet
evidence of network-policy enforcement: that must be checked before registration.

Default mode is preflight. It installs and verifies the runner archive, records
package versions, requires the actual cgroup CPU/memory limits, checks absence of
a mounted service-account token, and invokes only Runner.Listener --version.
It never registers a GitHub runner and needs no GitHub credential.

Execute mode requires a separately supplied JIT secret and uses --jitconfig.
No secret is generated or embedded by this tool. Registration, source-specific
routing, network enforcement checks and terminal log custody remain required
before execute mode can be used. Never reuse a terminated Job or JIT identity.

The profile must eventually execute every existing Current-Source job step and
preserve the ratchet. Preflight success is not compiler, CI or Inkling acceptance.

Official runner references:
https://github.com/actions/runner/releases/tag/v2.337.0
https://github.com/actions/runner/blob/main/src/Runner.Listener/Runner.cs
https://docs.github.com/en/actions/reference/security/secure-use

## Explicit workflow selection

The CI workflow accepts current_source_runner=pireus-cpu36-v1 on manual dispatch.
PIREUS_CPU36_RUNNER_LABEL must identify the prepared one-shot runner, and
current_source_base_sha must pin the PR comparison base. The declared job budget
is 150 minutes within the runner's 180-minute lifetime ceiling.

The guard rejects absent routing configuration, a hosted fallback, the wrong
CPU/memory cgroup, UID 0, a mounted service-account token, non-dispatch invocation
or a missing/invalid comparison base. The changed-tests step uses the frozen PR
base and actual dispatched head, preserving PR test selection instead of the
manual dispatch's ordinary one-fixture fallback. Existing gate order and commands
are unchanged. The default profile remains github-hosted.

Registration and dispatch still require a restricted runner identity and raw
receipt custody. Workflow routing code is not evidence that the job ran.
