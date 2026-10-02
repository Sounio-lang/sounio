# PR #2737: Default Region Integration Routing

Blocker-ID: BLK-20261002-pr2737-default-region-integration
Status: review-ready
Severity: B2
Class: evidence-gap
Evidence-Level: E3
Owner: codex
Lane: pr2737-default-region-thinlink
Worktree: pr2737-region-reclaim-real
Branch: codex/pr2737-region-reclaim
Source baseline: b7905db7e8467ca19e44e297a2c354c20baa40b2

## Verified Current Route

`main.sio` imports `compile_multimodule_native_advanced` from
`module_native_driver.sio`, not `module_loader.sio`. The driver delegates imported
sources to the full modular IR path; profile/cache options explicitly report
that they are not supported yet. The tuple-array gate's existing header names
the old thin-link unit builder, but the actual execution log identifies
`module_native_driver: using full IR path` and `imported_compile`.

All four integration gates passed with the opt-out unset on the current-source
compiler built by CI run 37030399012. Its compiler sources match b7905db7;
b7905db7 changes only the scalar-escape witness. No native source change is
required to restore these gates to the default mode. Each gate now explicitly
unsets the A/B opt-out, including when the caller exported it.

## Legacy Investigation (Not A Reproduced Default-Route Failure)

Region entry points are reserved in the lowering summary before body lowering.
`thin_module_summary_from_ir_module` therefore includes them in
`local_fn_count`, not just in the appended builtin slots. The thin-link local
emission loop prefixes non-builtin names with `thin_mN_`, whereas native codegen
recognizes region entry points only by their exact `__rgn_*` names. Merely adding
them to the builtin predicate does not deduplicate the local slots across units.

Before reactivating this legacy path, test exact runtime-name preservation and
local-slot deduplication, as well as cache separation by region mode. A candidate
patch was discarded from this change because these current gates cannot test
that dormant route. The legacy code remains unchanged. Source inspection alone
does not establish a failure in the shipped default route.

## Acceptance Gate

Executed in the isolated lab with `SOUNIO_NO_REGION_RECLAIM` unset, including
their refusal controls:

- `madaros_pub_in_specialized_gate.sh`
- `madaros_duplicate_main_collapse_gate.sh`
- `madaros_module_qualified_type_import_gate.sh`
- `madaros_thinlink_tuple_arr_import_gate.sh`

All returned exit status 0. The visibility refusal emitted E175 and no ELF;
the duplicate-main refusal retained its type-mismatch diagnostic and no ELF.

Compiler SHA256:
`f3661cebf54530c1ca01afe1d4291c414c88bdbba5c6b49dd807b2306129fca9`

Evidence root:
`/workspace/upstream-lab/pr2737-thinlink-default-20261002/receipts/baseline/`

Repro: `ulimit -s 524288; MADAROS_RAW_BIN=<current-source ELF> bash scripts/ci/<gate>.sh`
Observed: all four default-mode gates pass on x86-64 Linux.
Expected: pass without disabling region reclamation.
Acceptance-Gate: same four gates in exact-tip CI, followed by final PR review.
Fallback-Path: none; default modular full IR route.
Legacy-Kept: yes; legacy thin-link/cache code is untouched.
LLM-Offload: not-required; local Sounio scaffold attempt failed verification and
was not used in the final change (no new Sounio source is committed).

Next-Action: push the default-mode gates and verify exact-tip CI. This dispatch
does not clear unrelated activation, streaming or escape-path coverage findings.
