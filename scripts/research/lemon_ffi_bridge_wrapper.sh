#!/bin/sh
# scripts/research/lemon_ffi_bridge_wrapper.sh
#
# Short-command wrapper for examples/cayley_dickson_lemon_g2_ffi.sio's
# run_bridge(). Needed because the fixed `system()` FFI stub added in
# self-hosted/compiler/lean_single.sio (docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md,
# Track B) segfaults/hangs on long command strings (~100+ chars observed
# broken, ~94 chars observed working) -- a separate, narrower bug from the
# no-op this wrapper works around. Keeping the actual invocation short by
# routing through this checked-in script avoids it entirely.
python3 /workspace/sounio/scripts/research/lemon_ffi_bridge.py 15 15 /workspace/.home/openvscode-server/.agents/claude-2/.claude/jobs/f7926023/tmp/lemon_g2_bridge.csv
