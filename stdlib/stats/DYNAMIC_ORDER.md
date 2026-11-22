# Budgeted exact dynamic selection

`stats::dynamic_order::numeric_select(a,n,k,partition_budget)` selects a
zero-based order statistic from a caller-owned NumericBuffer prefix. It is
deterministic comparison-based selection of an existing finite value, not a
statistical approximation or floating-point average. The prefix can be permuted.
Inactive cells are never read or modified.

The existing numeric_heapsort implementation is retained unchanged. The new
primitive passed focused central checks, status and independent generated rank
references at Slurm job16127. The normalization consumer is measured separately
in genomics/DYNAMIC_NORMALIZATION.md; selection alone does not imply a universal
speedup or full corpus/CI acceptance.

## Contract and statuses

```sounio
numeric_select(a:&!NumericBuffer,n:i64,k:i64,partition_budget:i64)
  ->NumericSelection with Mut,Div,Panic
```

NumericSelection exposes status,value,partition_work,partitions,fallback.

- Status0: value and the prefix element at k are the selected order statistic.
- Status1: unsuccessful owner, n outside1..length, k outside0..n-1 or negative
  partition budget.
- Status2: a nonfinite active input element.

All range/owner checks and the entire active finite scan occur BEFORE any
mutation. On those failures, value/counters are0, fallback=false and the
buffer remains unchanged. A valid buffer is caller-owned mutable workspace;
success does not restore its initial order. No Alloc effect, fixed dimension
cap, hidden sample copy or random generator is used.

Signed zeros compare equal; a particular tie order or selected zero sign is
not promised. The returned value is copied from an actual element rather than
recomputed or averaged. Comparisons of finite binary64 values introduce no
rank tolerance.

## Partition and fallback

A scalar median of the first, middle and last active elements supplies the
pivot. A three-way Dutch partition separates values below, equal to and above
that pivot. It handles all-equal inputs in one traversal rather than repeatedly
peeling off individual ties. Only the side containing k is revisited.

partition_budget counts visited partition items. Each partition traversal
visits exactly its active width; the three pivot reads and heapsort comparisons
are not charged as partition items. Before starting a traversal, the full width
is compared with budget-partition_work. This avoids counter overflow and
prevents an unbounded quadratic partition path.

If the next traversal would exceed the remaining budget, the existing checked
heapsort sorts the ENTIRE original active prefix, and the value at k is returned.
fallback=true records this path. partition_work/partitions retain only completed
partition traversals; they do not pretend to count heapsort work. Budget0 or1
therefore uses heapsort immediately for n>1. A single element needs no partition
or fallback, including budget0.

For budget B, the work is O(n+B+n*log(n)) in the fallback case, including
preflight and sorting. A caller choosing B proportional to n obtains bounded
partition effort and the existing O(n*log(n)) worst-case escape. This is not a
worst-case-linear selection certificate, timing deadline or benchmark result.

The chosen value partitions the prefix: elements before k are <=value and
elements after k are >=value. A fallback additionally sorts the whole prefix.
There is no stable ordering guarantee among equal values.

## Prepared independent checks

`stdlib_deep_numeric_select_status.sio` checks transactional invalid ranges,
negative budget, active NaN, excluded poison, direct selection, full-prefix
budget0 fallback, n1 and all-equal one-partition behavior.

`tests/reference/deep_numeric_select_reference.cpp` computes independent
expected ranks with C++23 std::sort. It emits all k for selected small lengths
1,2,3,4,5,7,8,9,16,25,31 across sorted, reversed, many-tie, organ-pipe, all-equal,
finite extreme/subnormal and deterministic scrambled data. Each rank runs with
budgets0,1 and12*n. Additional20000-element cases exercise the two ends and both
middle order statistics under the same budgets.

Native checks compare exact values, kth position, partition ordering, full
sortedness after fallback, work-budget metadata and untouched inactive NaN
padding. Minimum-subnormal expected values are built by an exact power-of-two
division, avoiding a subnormal-literal parser limitation. There is no
unit-sized absolute tolerance that could hide a wrong tiny selected value.

Build the C++23 emitter on the authorized worker with -O2 -Wall -Wextra
-pedantic and generate
`tests/run-pass/stdlib_deep_numeric_select_reference.sio`.
Checks/compilation/execution are coordinated through central Slurm; no local
execution was performed by the implementation agent.
