# CI resource-envelope discovery

Live GitHub API queries returned zero repository runners, zero organization
self-hosted runners, and zero organization larger hosted runners. Counts and
observation time are preserved in the adjacent raw responses and summary.

The current job uses ubuntu-24.04 and a 75-minute whole-job budget. Its gen2
measurement on DL380 took 85 minutes under the independent 36 GiB allocation.
A runner label alone would therefore create a queue dependency without an
available runner. The existing K-AXI runner manifest is a template with an
unresolved image and is not evidence of a live runner.

Prospective qualification: an isolated ephemeral x86-64 CPU runner with at least
36 GiB usable memory, a bounded job budget allowing the measured gen2 duration
plus the existing build and gates, and the complete existing Current-Source job.
The final memory/time allocation must be recorded before execution. The 36 GiB
allocation is a measured gen2 starting point, not qualification of the full job.

Preserve checkout identity, all existing gates, expected rung and progress
thresholds. Rebuild the compiler from the checked-out source. Preserve raw logs,
compiler identity, resource observations, generated artifact hashes and terminal
GitHub job/run/CI Decision. A gen2-only Slurm success cannot stand in for these.

Provisioning remains open. No runner was registered, no workflow label or timeout
was changed, and no model or CI retry was submitted by this discovery packet.
