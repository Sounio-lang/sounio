# Nested tuple-alias lowering refusal: resolution receipt

**BLK-20261026-F64-NESTED-TUPLE-ALIAS** is resolved by the isolated source
change qualified below. Integration into main is a separate CI/merge claim.
The earlier #3227 receipt remains historical evidence of the refusal.

## Root cause

`lower_block_tail_variance_ref` reconstructs only bindings whose initializers
are safe to evaluate twice. A tuple constructor is conservatively outside that
set. The walk skipped `let t = (7, 2.0)` and marked the remaining tail unknown,
but still tried to lower a subsequent `let u = t`. Since the skipped binding
did not exist in the replay scope, the value-lowering path reported unresolved
`t` and refused a program that had passed checking.

Gate subsequent `StmtLet` replay with `!skip_later_expr`, just as subsequent
expression statements were already gated. The original value path is unchanged;
the skipped replay still returns the existing unknown-variance sentinel `-1`.
No initializer is newly executed, no variance formula is changed, and the
normal scope-depth/count restoration remains on the same exit path.

## Executed before/after

Source base: `8e6ecf90e01ec84ba63eea05d87852fb55e10ddf`.
Both compilers were rebuilt through `scripts/ci/build_modular_madaros.sh` on
`cpuops-t560-proxmox`, inside disposable allocated Slurm worker trees.

| Artifact | SHA-256 |
|---|---|
| Baseline compiler, Slurm 19209 | `9884dc1b76045e0ebb5ab68c9780559a863a9a8f8032dbf5ddd9cbb4280351b7` |
| Baseline lower.sio | `ff05736e3fd23060a819d771d610a080caf6c742e3870ce0327ec11951eff139` |
| Repaired compiler, Slurm 19218 | `62df9ba942979b343b281741b92d0b1723e84771e4a94df172418fea6ba548c5` |
| Repaired lower.sio | `1cfe5cf6be641a3913006ac7d5d769e7a1a60b9ca704a37e95dba67d0b6ac97b` |
| Preserved original reproduction | `7255b01f0858871a40b23f191f209146e76835fe7f443349f226ddba0e73b622` |
| Expanded native regression | `e75d82a8b1d80100db1e1f8095299c382c472a123704a34a06919a385486926a` |

The preserved original source passed `bin/souc check` with the rebuilt
baseline, then compilation exited **1**, with two unresolved `t` diagnostics
at reported site 15349. With the repaired compiler the same source passed
checking, compilation and native execution (**0**), printing
`NESTED_TUPLE_ALIAS_OK`.

The expanded `nested_tuple_alias_block.sio` checks both condition values,
different f64 branch results, multi-link aliases, integer tuple controls, and
an effectful tuple-returning helper that must execute exactly once. It passed
check/compile/run and printed `NESTED_TUPLE_ALIAS_BLOCK_OK`.

The canonical harness selected `engine=madaros`. Each of the three filters
`nested_tuple_alias_block`, `f64_block_metadata_audit` (#3227), and
`f64_expression_class_audit` (#3133) executed one nonempty selection and reported
**Pass 1 / Fail 0 / Skip 0**. Initial corpus enumeration is not whole-corpus
execution.

Six neighboring native tests compiled and exited 0:
`gum_fo_if_let_abs_2966`, `fo_product_rule_call_evaluated_once`,
`gum_fo_runtime_if_1706`, `fo_inline_nested_if_call_args`,
`madaros_tuple_f64_array_from_if_and_match`, and `madaros_f64_deref_operand`.
The job ended with `NESTED_TUPLE_ALIAS_REPAIR_PASS`.

## Reproduction

Run in a disposable allocated Linux worker, not the shared development pod.
Install the rebuilt ELF only in that private worker tree:

```sh
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
bash scripts/ci/build_modular_madaros.sh /tmp/madaros-nested-tuple
mkdir -p artifacts/self-hosted
cp /tmp/madaros-nested-tuple artifacts/self-hosted/madaros
export MADAROS_RAW_BIN="$PWD/artifacts/self-hosted/madaros"
bin/souc check self-hosted/ir/lower.sio
SOUNIO_MADAROS_AVAILABLE=1 SOUNIO_TEST_JOBS=1 \
  bash scripts/run_sio_test_suite.sh nested_tuple_alias_block --verbose
```

For the original discriminating shape, compile and run:

```sounio
fn main() -> i32 with IO, Mut, Div {
    let flag = true
    let result = if flag {
        let t = (7, 2.0)
        let u = t
        u.1
    } else {
        let t = (9, 2.0)
        let u = t
        u.1
    }
    if result / 16.0 != 0.125 { return 1 }
    println("NESTED_TUPLE_ALIAS_OK")
    0
}
```

## Custody, review and limits

Durable receipts are under
`/orangefs/training/sounio-foundry/runs/nested-tuple-alias-baseline-20261028`
and `nested-tuple-alias-fixed-20261028`. Compiler/log custody was retrieved into
`artifacts/nested-tuple-alias-20261028/{baseline,fixed}/` in the isolated branch
`fix/nested-tuple-alias-20261028`.

Advisory source reviews: Anthropic `claude-sonnet-5-5` via the cluster LLM
Gateway and Alibaba Qwen3-Coder-Next via Darwin's local `coder-next`. Neither
found a supported new control-flow/effect/scope defect in the guard. The
Anthropic suggestion of a new precision loss was checked against the old tail
guard: tails after an unsafe let were already unknown, so that suggestion is
not a new regression. No reviewer execution or formal proof is claimed.
Other review attempts were unavailable: xAI returned no usable response,
ZAI's quota was exhausted, and BEAGLE returned HTTP 502. Outcomes are recorded
in `.claude/llm_offload_log.md` without claiming their approval.

This closes the observed valid-program refusal. It does not establish complete
tuple uncertainty propagation, whole-corpus correctness, self-compilation fixed
point, wide-float parity or mathematical soundness of every unknown-variance
consumer. No scientific model, data, default executable or legacy compiler
path was changed. No independent Python/Rust numerical oracle was used.
