# Pireus Heartbeat Runner Setup

This document describes how to configure a GitHub Actions self-hosted runner on an NVIDIA DGX Spark (GB10/sm121, ARM64) to run the Pireus Heartbeat workflow.

## Overview

The Pireus Heartbeat workflow (`.github/workflows/pireus-heartbeat.yml`) is a read-only monitoring system that collects cluster state on an hourly schedule. It captures:

- Slurm queue state (all jobs, filtering for Pireus/Inkling/SGLang)
- Recent job history (26-hour window via `sacct`)
- GPU status (`nvidia-smi` queries)
- Evidence directory state and checksums

The workflow **never submits, modifies, or terminates jobs**. It produces a JSON + markdown receipt uploaded as an artifact with 90-day retention and written to the GitHub Actions job summary.

## Verdict Rules

The heartbeat script determines one of five verdicts based on **freshness** (26-hour threshold):

| Verdict | Meaning | Exit Code |
|---------|---------|-----------|
| `RUNNING` | Pireus job active/recent (< 26h) **AND** evidence fresh (< 26h) | 0 |
| `IDLE` | Evidence exists but stale (≥ 26h) or no recent Pireus jobs | 0 |
| `FAILED_JOBS` | Pireus-related jobs with failure states in last 26h | 0 |
| `NO_EVIDENCE` | No Pireus jobs and no evidence directories found | 0 |
| `NO_SLURM` | Slurm commands unavailable | 1 |

**Freshness rule**: A `RUNNING` verdict requires **both**:
1. A Pireus/Inkling/SGLang job that is currently active OR ended within the last 26 hours, **AND**
2. At least one evidence directory with files modified within the last 26 hours.

This ensures the verdict reflects actual cluster activity, not just stale artifacts.

## Prerequisites

### Spark Nodes

- **Host identifiers**: 3c59 and 8e54 (both ARM64 DGX Spark GB10/sm121)
- **Operating system**: Linux ARM64 with Slurm and Apptainer
- **Slurm access**: An unprivileged user with read access to `squeue`, `sacct`, `sinfo`
- **GPU access**: Read-only access to `nvidia-smi`
- **Network**: Outbound HTTPS to `github.com` and `*.actions.githubusercontent.com`

### Runner User

Create a dedicated unprivileged user for the runner (e.g. `gh-runner`):

```bash
sudo useradd -m -s /bin/bash gh-runner
```

Ensure the user has:

- Read access to Slurm commands (`squeue`, `sacct`, `sinfo`)
- Read access to `nvidia-smi`
- Read access to the repository checkout (the runner will clone it)
- No write access to Slurm job submission, job control, or cluster configuration

## Runner Registration

### Step 1: Navigate to Repository Settings

1. Go to the Sounio repository on GitHub: `https://github.com/Sounio-lang/sounio`
2. Click **Settings** > **Actions** > **Runners**
3. Click **New self-hosted runner**

### Step 2: Select Platform

- **Operating System**: Linux
- **Architecture**: ARM64

GitHub will display commands to download and configure the runner.

### Step 3: Download and Extract

On the Spark node (as `gh-runner`):

```bash
cd ~
mkdir actions-runner && cd actions-runner

# Download the latest ARM64 runner (example - use the URL from GitHub's instructions)
curl -o actions-runner-linux-arm64-2.319.1.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-arm64-2.319.1.tar.gz

# Extract
tar xzf ./actions-runner-linux-arm64-2.319.1.tar.gz
```

### Step 4: Configure the Runner

Run the configuration script with the **exact labels** required by the workflow:

```bash
./config.sh \
  --url https://github.com/Sounio-lang/sounio \
  --token <REGISTRATION_TOKEN> \
  --name "spark-$(hostname)-pireus-heartbeat" \
  --labels "self-hosted,linux,ARM64,spark,pireus-heartbeat" \
  --work _work \
  --unattended \
  --replace
```

**Critical**: The runner **must** include all five labels:

1. `self-hosted` (automatic)
2. `linux` (automatic)
3. `ARM64` (architecture)
4. `spark` (reused from `gb10-wmma-receipt.yml`)
5. `pireus-heartbeat` (workflow-specific label)

The workflow's `runs-on` line references all five:

```yaml
runs-on: [self-hosted, linux, ARM64, spark, pireus-heartbeat]
```

### Step 5: Install as a Service

Install the runner as a systemd service so it starts automatically:

```bash
sudo ./svc.sh install gh-runner
sudo ./svc.sh start
```

Check the service status:

```bash
sudo ./svc.sh status
```

The runner should appear as "Idle" in the GitHub Actions Runners page.

## Repeat for Second Spark

Repeat the entire registration process on the second Spark node (3c59 or 8e54). Each node should have its own runner instance with a unique name but **identical labels**.

## Security Considerations

### Public Repository Risk

**CRITICAL**: The Sounio repository is public, so any GitHub user can open a pull request. By default, pull request workflows from forks do **not** run on self-hosted runners, but repository administrators must enforce this policy.

### Workflow Restrictions

1. **Restrict fork PR workflows**:
   - Go to **Settings** > **Actions** > **General**
   - Under "Fork pull request workflows from outside collaborators", select:
     - **Require approval for first-time contributors**
   - This prevents untrusted PR workflows from running on self-hosted runners without approval.

2. **Do not run PR workflows on this runner**:
   - The Pireus Heartbeat workflow is **not** triggered by `pull_request` or `pull_request_target` events.
   - If adding PR triggers in the future, add explicit runner restrictions or use GitHub-hosted runners for PR builds.

3. **Workflow isolation**:
   - The workflow runs in the repository checkout, not in user home directories.
   - The script is strictly read-only: it never submits jobs, writes to Slurm state, or modifies the cluster.

### Network Restrictions

If the Spark nodes have firewall rules, allow outbound HTTPS to:

- `github.com` (runner registration, workflow download)
- `*.actions.githubusercontent.com` (job dispatch)
- `*.githubusercontent.com` (artifact upload)

## Verifying the First Run

### Manual Trigger

1. Go to **Actions** > **Pireus Heartbeat** workflow
2. Click **Run workflow** > **Run workflow**
3. Watch the job progress in real-time

### Expected Output

The workflow will:

1. Checkout the repository
2. Run `scripts/ci/pireus_heartbeat.sh`
3. Upload the receipt as an artifact
4. Display a markdown summary with:
   - Verdict (`RUNNING`, `IDLE`, `FAILED_JOBS`, `NO_EVIDENCE`, or `NO_SLURM`)
   - System information (hostname, timestamp)
   - Evidence directory table (path, exists, file count)

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Runner not picking up jobs | Missing or incorrect labels | Re-run `config.sh` with correct labels |
| `NO_SLURM` verdict | Slurm commands not in PATH | Ensure `squeue`, `sacct`, `sinfo` are accessible to `gh-runner` |
| `nvidia-smi: command not found` | NVIDIA driver not in PATH | Add `/usr/bin` or `/usr/local/bin` to `gh-runner`'s PATH |
| Workflow times out | Slurm query hangs | Check Slurm controller connectivity; reduce timeout in workflow |
| Artifact upload fails | Network or disk space issue | Check outbound HTTPS and `/tmp` space |

## Local Dry-Run Testing

Before registering the runner, test the heartbeat script locally:

```bash
cd /path/to/sounio
PIREUS_HEARTBEAT_DRY_RUN=true \
PIREUS_HEARTBEAT_RECEIPT_DIR=/tmp/test-heartbeat \
bash scripts/ci/pireus_heartbeat.sh
```

This simulates the collection without querying Slurm and writes output to `/tmp/test-heartbeat/`. It will still check evidence directories if they exist.

## Scheduled Runs

The workflow runs hourly via cron:

```yaml
schedule:
  - cron: '17 * * * *'
```

Artifacts are retained for 90 days. To view historical runs:

1. Go to **Actions** > **Pireus Heartbeat**
2. Select a past run
3. Download the artifact or view the summary tab

## Unregistering a Runner

To remove a runner (e.g. when decommissioning a Spark):

```bash
# Stop the service
sudo ./svc.sh stop
sudo ./svc.sh uninstall

# Unregister from GitHub
./config.sh remove --token <REMOVAL_TOKEN>
```

The runner will disappear from the GitHub Runners page.

## Evidence Directory Configuration

By default, the workflow monitors **cluster-side paths** outside the runner checkout:

```bash
PIREUS_EVIDENCE_DIRS=/workspace/.wt/pireus-integration-20260906/tools/pireus/continuity/validation
```

This points to the Pireus worktree on the cluster, which contains:
- `validation/` directory with actual execution receipts
- `real-inkling-cycle-20260907/` subdirectory with live Inkling cycle artifacts

**Automatic WorkDir discovery**: The script also automatically discovers and monitors Slurm job working directories by parsing `sacct --format=WorkDir` output for any jobs matching `pireus`, `inkling`, or `sglang` in the last 26 hours.

**Freshness calculation**: For each directory, the script:
- Finds the newest file (by mtime, recursively)
- Reports its timestamp and age in seconds
- Marks it as "fresh" if age < 26 hours

To add more directories, update the `PIREUS_EVIDENCE_DIRS` environment variable in the workflow file or pass it as a repository variable:

1. Go to **Settings** > **Secrets and variables** > **Actions** > **Variables**
2. Create a new repository variable: `PIREUS_EVIDENCE_DIRS`
3. Set the value to a colon-separated list of **absolute cluster paths**

The workflow will prefer the repository variable if set.

**Important**: Evidence directories must be **cluster-side paths** (e.g. `/workspace/.wt/...`, `/scratch/...`, `/data/...`), not repository-tracked paths. Repository paths like `tools/pireus/evidence` prove nothing about actual cluster execution.

## Contact

For runner issues or workflow failures:

- Check the workflow run logs in the GitHub Actions UI
- Review the uploaded artifact (`pireus-heartbeat-<run_id>-<attempt>.zip`)
- Open an issue in the Sounio repository with the run URL and artifact
