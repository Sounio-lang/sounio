# Frozen-source CI interruption diagnostic

Source e17959ac1238d3e0ceb5fbd9f94bd117e8ee8446 is not CI-qualified.
Attempt1: runner GitHub Actions1000078926,00:25:42–01:29:12UTC.
Attempt2: runner GitHub Actions1000078989,01:30:42–02:34:46UTC.
Both exited143 following an explicit runner shutdown signal during gen2.
Gen1 check passed with rc0/errors0 in both. Lean passed in attempt2.

The workflow declares75 minutes; observed job durations were63m30s and64m04s.
These observations do not establish a75-minute timeout or a compiler defect.
The long compile writes to gen2.log, so its progress was absent from the CI
console for roughly48 minutes before each shutdown. Whether silence contributed
to termination is an unproven hypothesis.

The new diagnostic observer reports elapsed seconds, log bytes and the last
two log lines every30 seconds. Compiler argv, raw log bytes, exit status,
rung expectation and progress threshold remain unchanged. The observer adds
no timeout or retry. Local controls cover live output, byte-preserving logs,
statuses0/7/78/143, observer cleanup and pre-execution refusal of an invalid
interval. Passing them does not establish the root cause or qualify inference.

No third unchanged-source rerun is configured. A diagnostic CI run has a new
source identity; any subsequent feedback execution must bind an explicit
new freeze to that source. Original token requests and evidence remain preserved.
