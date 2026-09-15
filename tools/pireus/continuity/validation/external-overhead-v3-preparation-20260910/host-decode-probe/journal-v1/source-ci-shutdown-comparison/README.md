# Source CI shutdown comparison
The successful Current-Source job 102752324070 (run 34439721975, source 2416ffb47c) and interrupted job 102785095676 (run 34443820184 attempt 2, source 342b3f4e36) used identical Git objects for self-hosted, stdlib, scripts/ci, scripts/lib and the CI workflow. This compares source objects, not emitted compiler binary bytes or runner resources.

Both logs contain arena_reset_totals ok=0 skip=124 sites_reclaimed=0 and Merged IR: 13269 functions. These markers alone do not discriminate the failure. The successful run records gen2 command_rc=0, then the expected rung run; gen3 returns 139, which is an existing next-rung limitation. The interrupted run records no gen2 completion, then exit 143 and a runner shutdown signal.

The observer emits log samples every 30 seconds; extracted first-observation times are not exact compiler event times or benchmark measurements. The signal sender, host memory pressure, OOM and shutdown cause remain unestablished. No deadline, ratchet or runtime profile is changed by this comparison.

Reproduce comparison.json by running python3 compare.py from this directory (or any working directory in this checkout). The failed log and job identity are referenced from the adjacent source-ci-attempt-2 archive. No compiler or model is reexecuted.
