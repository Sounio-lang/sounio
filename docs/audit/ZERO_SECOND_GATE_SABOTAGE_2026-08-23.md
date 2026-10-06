<!-- docs:meta
topic_id: repo.docs.audit.zero-second-gate-sabotage-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.zero-second-gate-sabotage-2026-08-23
-->

# The forty-five instant-green gates, arbitrated by sabotage

**Date:** 2026-08-23
**Method:** every gate below was RUN, then SABOTAGED in the specific thing it
claims to check, then RUN AGAIN, then restored. A verdict of REAL means the
sabotage turned it red. Nothing here is a verdict from reading the script.

## Why these forty-five

A full sweep of 465 CI gates against a freshly built Madaros (build rc=0, 773 s,
on dl380) returned:

| | |
|---|---:|
| PASS | 108 |
| FAIL | 276 |
| ENV (environment missing, not judged) | 46 |
| TIMEOUT | 35 |

Forty-five of the 108 passes completed in a reported **0 s**, which looked like
the signature of a gate that cannot fail.

**That premise was partly wrong, and correcting it is the first result.** The
sweep's timer rounds to whole seconds. Most gates that genuinely invoke the
compiler finished in 0.2–0.7 s and were reported as 0 s. A "0 s" row is not
evidence of an empty gate; it is evidence of a gate faster than the timer's
resolution. The gates that really do no work are the ones with a named
early-exit guard, and they had to be identified individually.

## Verdicts

45 gates, all four verdicts assigned by measurement.

| verdict | count |
|---|---:|
| REAL — sabotage turned it red | 30 |
| SKIPPED — a named guard stopped it before its own logic | 12 |
| VACUOUS — sabotage did not turn it red | 2 |
| not a gate | 1 |

### The two vacuous ones

**`fo_method_xfer_fragment_gate.sh`.** Its certificate compares `compile_css()`
against `golden = compile_css()` — the same function called twice, with no
independent oracle. That is a tautology, and one of its three checks is a
hardcoded `True`. Gutting `compile_css()` to return `[(99,99)]` still printed
`CERT_OK 5/5`. No change to the function under test can ever fail this gate.
*Smallest fix:* replace the golden with an independently written literal RPN
list, as every sibling certificate already does, and replace the hardcoded
`True` with a real comparison.

**`kaxi_ptx_capture.sh`.** Not an assertion at all — it writes golden files that
do not yet exist and skips everything once they do (318/318 kept here, hence the
instant green). Replacing `bin/kretikos` with a script that fails every time and
deleting a golden pair still exited 0; failures become `.unsupported` marker
files that nothing counts. The comparison against golden lives in a different
script.
*Smallest fix:* stop calling it a gate, or fail on a non-zero `UNSUPPORTED`
count and a hash mismatch against a fixed reference set.

### Not a gate

**`scripts/dev/madaros_v2_enir_gate_scope.sh`** is a sourced bash function with
no top-level call. The sweep counted a library file. The function itself is real
and has 14 callers.

## Defects found in gates that are otherwise REAL

These passed their sabotage but carry a hole worth closing.

**A typo that has become a contract.** `scripts/ci/souc-native-wrapper.sh:100`
tests `grep -q "Madares v"`. The binary prints `Madaros v0.80.0`. That branch
can never match. The misspelling is in **31 files, 13 of them gates**, and eight
`proof_carrying_*` gates now write `grep -Eq "Madares|Madaros"` — somebody
noticed the divergence and accepted both spellings instead of fixing it.

**`no_false_float_axioms.sh` cannot see a multi-line axiom.** Its grep is
anchored to a single line. `formal/` already contains 72 axiom declarations
written across two lines, so a forbidden axiom in the file's own prevailing
style is invisible. Verified: the single-line form is caught, the two-line form
is not.
*Smallest fix:* normalise each statement onto one line before matching.

**`kretikos_kaxi_phase_w_gate.sh` fails where it means to skip.** Line 78 builds
`kaxi_ptx_runner.c` without `-lm`; on this toolchain that is an undefined
reference to `sqrt`, and it fires BEFORE the GPU-absence check. The gate reports
FAIL where it intends SKIP. One flag.

**`zd_deep_dive_gate.sh` asserts a string, not a number.** It checks only that
`ALL PASS` was printed. The file's own baseline comment says 92 zero-divisors;
the live output is 84. The gate reads neither.

**`madaros_launcher_exit_status_gate.sh` has a duplicated target.** Its string
occurs twice in `bin/madaros`; editing only the reachable occurrence still
passes.

**`madaros_imported_f64_mul_gate.sh` accepts either outcome.** A `FIXED` marker
or a `RESIDUAL` marker both count as a pass, and a `grep -q … || true` at line
53 asserts nothing.

**`seed_receipt_provenance_gate.sh` has never run its namesake check.**
`bin/souc-lean-single-x86_64.SeedReceipt.json` does not exist anywhere in the
tree or its history, so the `--check-against-tree` hash comparison — the thing
the gate is named for — has never executed. What runs is the synthetic mutant
control and the surface-touch classifier, both of which are real. This is
correct by the gate's own policy (main must stay green before a receipt exists),
so the gap is an artifact-production gap, not a script bug: generate and commit
the receipt once.

## The structural finding

The seven `fo_*` gates are individually REAL — six of seven failed their
sabotage correctly. But every one of them is a Python re-implementation checked
against itself. None reads `self-hosted/`, none reads a `.lean` file, none
touches a compiler artifact. And their formal half is behind
`FO_CSS_LEAN_BUILD=1`, which defaults to off, so in this sweep the Lean build
ran zero times.

A green from any of them means **"this hand-written Python model is internally
consistent"**, not "the compiler or the proof does what the docstring says". That
is a defensible thing for a gate to check. It is not what a reader of the gate
list would assume it checks.

## The twelve that skipped, and why that is not automatically a fault

Eight K-AXI/GPU gates exit early because `ptxas`, `nvidia-smi` or `libcuda` is
absent; two Sinkhorn Slurm gates exit because `SOUNIO_MOONSHOT_A_RUN_SLURM` is
unset; `semantic_orc_swow16_kaxi_slurm_gate.sh` because
`SOUNIO_SEMANTIC_ORC_RUN_SLURM` is unset; `check_feature_matrix.sh` and
`check_new_warnings.sh` because `SKIP_BUILD` defaults to 1 through
`SOUNIO_REPO_HARD_NO_RUST:-1`.

Each of those is an honest skip. The problem is upstream of the gates: **the
sweep reports a skip as a pass.** Two of them write a JSON artifact whose own
`status` field says `not_run_opt_in_required` while the shell exit code says 0,
and the sweep reads the exit code. A summary line of "108 passed" therefore
counts at least twelve gates that ran none of their substance.

`kretikos_kaxi_l4_launch_gate.sh` deserves a separate note: its skip depends on
`kubectl` reachability, and on this pod `kubectl` DOES reach a live Slurm login
pod. On a machine like this one it would take the live-submission branch rather
than skipping. What the real CI runner has was not established.

## Method notes, recorded because they cost time

**Grep the code, not the file.** Counting `emit_sar_rax_cl` across a source file
found two hits, both inside comments explaining why that instruction is wrong
there. Counting occurrences of a function name to find its callers counts the
definition line as a call.

**One agent forced an opt-in variable and submitted a real cluster job.** While
testing `semantic_orc_swow16_kaxi_slurm_gate.sh` a sub-agent set `RUN_SLURM=1`;
job 10649 ran and failed on its own in about a second. The instruction it had
said "do not run builds", which did not cover "do not submit jobs". A gate that
skips for want of an opt-in variable is a finding to report, never an obstacle
to remove.
