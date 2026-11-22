<!-- docs:meta
topic_id: repo.docs.ops.macos-ci-runner
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ops.macos-ci-runner
-->

# Self-hosted macOS arm64 runner for the merge queue

The `Native Self-Host (macOS arm64)` job is the slowest job in a merge-queue group: GitHub-hosted `macos-15` runners are scarce, and every group waits for one. Setting the repository variable `SOUNIO_CI_MACOS_RUNNER` sends that job to a self-hosted Mac **only** for `merge_group` runs and for `main` (push, schedule, dispatch). Pull-request runs stay on GitHub-hosted macOS, so unreviewed code never reaches the machine. Deleting the variable falls back to `macos-15` immediately, without a commit.

This mirrors the Linux `SOUNIO_CI_RUNNER` boundary in `.github/workflows/ci.yml`.

## 1. A dedicated macOS user

Run the runner as its own account, never as your login user. That account must not be able to read your home folder, keychain, SSH keys or tokens.

1. System Settings → Users & Groups → Add User: a **Standard** account named `sounio-ci`.
2. Do not sign it into an Apple ID or a browser, and do not give it admin rights.
3. Toolchain. The job calls `/usr/bin/clang++ -std=c++23` (Apple clang 21 accepts it) and probes `$(brew --prefix)/opt/coreutils` and `.../opt/boost` on the filesystem, running `brew install` only if one is missing. It does not call `brew list`, because Homebrew refuses every command, even read-only ones, for a user that does not own the prefix. Homebrew in `/opt/homebrew` belongs to your login user, so a Standard account cannot install into it. **Pre-install from your own account**: `brew install bash coreutils boost`. Then make sure `sounio-ci` sees `/opt/homebrew/bin` first in its `PATH`, for example via `~/actions-runner/.path`, so `bash` resolves to Homebrew's bash 5 and not to `/bin/bash` 3.2.

## 2. Register the runner

As `sounio-ci`:

1. On GitHub: repository → Settings → Actions → Runners → New self-hosted runner → macOS / ARM64. Copy the download and `config.sh` commands.
2. Run them in `~/actions-runner`. When `config.sh` asks for labels, add `sounio-macos` (keep the defaults `self-hosted`, `macOS`, `ARM64`).
3. Install it as a service so it survives logout and reboot: `./svc.sh install && ./svc.sh start`.
4. Check the runner shows **Idle** on the Runners page.

## 3. Point the job at it

Repository → Settings → Secrets and variables → Actions → Variables → New repository variable:

- Name: `SOUNIO_CI_MACOS_RUNNER`
- Value: `sounio-macos`

The next merge-queue group runs the macOS job on the Mac. To undo, delete the variable.

## 4. Operating notes

- The machine must stay awake and online while the queue has entries; a sleeping runner leaves the job queued until `check_response_timeout_minutes` (180) expires.
- Each job checks out into the runner's work directory; clear `~/actions-runner/_work` if disk runs low.
- Keep the runner updated; GitHub refuses runners that are too old.
- `/tmp` is not reset between jobs on a self-hosted machine. The job's first step deletes the entries in `/private/tmp` owned by the runner user, and nothing else, because the gates refuse to overwrite existing outputs.
