# Empty extern namespaces and arena discriminator

The final capability validator previously required executable parameter registers in empty native/extern ABI stubs. The real imported thread and loader fixtures reached error 204 after the diagnostic `SOUNIO_INTO_ACC_NO_RESET=1` removed their earlier identity loss/hang. Their actual IR contains empty dlopen/dlsym/region/assert stubs with param_count > 0 and reg_count = 0; these are valid emitter-owned ABI entries, not executable IR frames.

The validator now skips executable-register initialization for empty ordinary stubs. An empty function carrying any source opaque parameter/return contract still fails closed with error 229. Existing call-site capability checks and fresh source signatures remain mandatory.

Slurm 19431 preserved a paired native component: before exits 1 with EXTERN_NAMESPACE_FLOW_REFUSED 204; after exits 0 with WORKER_EXTERN_NAMESPACE_OK ordinary_empty=1 opaque_empty_refused=1. Both modern checks passed. The existing 15 IR mutations, nine CFG corruptions, and three positive CFG controls also passed. Source/input/modern-compiler guards are recorded in worker-extern-components. The legacy seed identity was collected after execution only; this run does not provide a before/after seed guard.

This closes the component namespace defect only. Real BFS is still blocked by dependency arena lifetime: the default raw11332 thread fails identity10/cap201 and an imported fixture times out; NoReset is only a discriminator. No Worker API or native pool acceptance is claimed. The current production compiler entry remains closed.
