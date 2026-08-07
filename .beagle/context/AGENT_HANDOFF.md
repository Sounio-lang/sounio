# Beagle Context Handoff

This folder contains useful workspace metadata, but it is not branch authority.

Current operational notes:

- Active workspace: `/workspace/sounio`.
- Safe branch verified by Git during orchestration setup:
  `integration/sounio-dev-ready-base`.
- Some Beagle context files still report branch `main`; agents must verify with
  `git branch --show-current` before acting.
- Beagle status may be `local-only` or `degraded`; use repo files and Git as
  source of truth.

Shared startup packet:

1. `CLAUDE_HANDOFF.md`
2. `AGENTS.md`
3. `.agent-orchestration/HANDOFF.md`
4. `.beagle/context/workspace-subagents.json`

Current subagent metadata says the live role is `sounio-core` with role tag
`compiler-runtime`. Treat this as workspace orientation, not edit permission.

---

## Operational field notes — 2026-08-07 (Kimi session, SAN-FPGA line)

Hard-won infra facts from the 2026-08-06/07 cluster session; each cost real
time to diagnose. Read before touching the Slurm cluster or pushing.

### GitHub auth from agent shells

- In Kimi CLI sessions, `HOME` is `/workspace/.home/openvscode-server/.agents/kimi-cli2`,
  so `gh` looks for its config in the wrong place and `git push` fails with
  "could not read Username for 'https://github.com': terminal prompts disabled".
- Fix: prefix git/gh commands with
  `GH_CONFIG_DIR=/workspace/.home/openvscode-server/.config/gh`
  (that is where the working oauth token lives).
- 2026-08-07 also saw a transient full-DNS outage on the workspace host
  (getent returned nothing for all hosts; recovered after ~10 min). If
  `git push` fails with "Could not resolve host", check DNS first
  (`getent hosts github.com`) before touching credentials.

### Slurm cluster (slurm-pilot)

- Access: `kubectl -n slurm-pilot exec slurm-pilot-login-slinky-<pod> -- ...`.
  The login pod is a submission host only — do not run torch/training on it.
- To inspect files on a busy compute node: `srun --overlap --jobid=<JID> ...`
  works and is the only route when the node is fully allocated.
- `cpuops-t560-proxmox`: its `/orangefs/training` is a **local ZFS mount that
  diverges from the real OrangeFS** (contents differ), and the node has **no
  outbound internet** (pip/curl fail). Do not schedule work that needs the
  shared FS or downloads there.
- `gpuorangefs-r770-proxmox` exposes only 128533 MB to Slurm: a job asking
  `--mem=128G` (131072 MB) **can never schedule there** and will sit PENDING
  with `Reason=Resources` while the node looks idle. Use ≤ 64G to keep all
  three GPU nodes eligible.
- QOS `burst` allows ~3 concurrent GPUs per user; extra jobs queue with
  `Reason=Priority`. Short `--time` (e.g. 3h) helps backfill.
- The SAN harness exits nonzero when the contract verdict is L_RED, so
  `sacct` shows FAILED for fully completed runs. Check the log tail for
  `SUFFERING_AWARE_LARGE_VERDICT` before assuming a crash; all ledgers are
  written before the verdict line.
- Long-running cluster jobs from agent sessions: arm a background watcher
  polling `sacct` every 5–10 min rather than blocking the session.

### Reproduction anchors for this line

- Worktree: `/tmp/sounio-san-fpga-blockers-20260804`, branch
  `research/san-fpga-san-v3-20260805` (PR #1659).
- Cluster runs: `/orangefs/training/sounio/kimi-runs/san-large-gpu/<RUN_ID>/`.
- Payload staged at `/orangefs/training/sounio/san-large-source/`
  (`suffering_aware_large_architecture_v2.py` + job scripts).
