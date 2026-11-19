# Journal diagnostic: second source CI attempt stopped
The exact-source CI run 34443820184, attempt 2, failed on source 342b3f4e36c78c48d70b3ca27b7ef69eb659001e. Current-Source job 102785095676 completed at 2026-09-10T08:36:21Z; CI Decision failed.

The job log reports exit 143 and a runner shutdown signal. The final compiler observations include completed lowering and an IR containing 13269 functions. These observations do not establish compiler correctness, an OOM, a timeout, or the root cause of runner shutdown. The earlier attempt's 75-minute timeout remains a separate result.

The recovery waiter recorded STOP_SOURCE_CI at 08:37:29Z and exited. Neither the driver invocation marker nor the diagnostic attempt marker exists. No Inkling job was submitted, and no third CI recovery was requested.

Next: diagnose runner termination before defining a new source qualification. Preserve the frozen journal diagnostic as unexecuted; retain all 33/32 GiB memory boundaries and pilot acceptance limits.
