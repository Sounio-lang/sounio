# Reject mixed skipped receipts

reliability_status.mjs now refuses success whenever E2E totals include skipped cases. This applies to current receipts and legacy v1 receipts without status_summary. Existing checks for executed passes, producer errors, duplicate cases and missing receipts remain active.

Slurm19131 completed0. The15 operational controls include both mixed pass/skip formats, all-skipped, empty selection, failed producer and the full shell flow after a failed hyper stage. This is an orchestration result; it makes no numerical or whole-corpus acceptance claim. The corpus19073 remains455pass/54fail/221skip.
