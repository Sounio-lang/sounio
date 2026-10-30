GPU lane receipt for Plan J serving (NVIDIA L4, job 20457, ExitCode 0:0): params unit test SERVJPARAMS PASS, parity demo greedy parity=1 SERVEDONE PASS, two-client overlap OVERLAP-OK (non-stream 200 with usage, stream 200 with SSE frames plus data: [DONE]).
Overlapping requests isolate on sess=0 and sess=1 with per-session KV, TTFB 6.9 s and 8.2 s on GPU.
Fixes: IO effect on fail helpers, SSE DONE length 14 in test and demo, stream send-error check < 0 on nw_send_all byte count.
