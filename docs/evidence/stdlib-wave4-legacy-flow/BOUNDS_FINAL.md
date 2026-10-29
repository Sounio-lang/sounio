# Frozen metadata bounds follow-up

This supersedes the private seed hashes in FINAL_CANDIDATE.md. The reachability,
nominal coverage and shadow-presence behavior remains as qualified there.

Root requested explicit physical bounds rather than relying on the scanners'
current caps. The final change is confined to x86 coverage helpers:

- EN_COUNT must be in0..256 and ST_COUNT in0..1024.
- TC must be in2..2097152, matching the physical token arrays.
- A defining token is checked with nt>=TC-1 after the TC guard, avoiding nt+1
  overflow. The token bounds apply to validator and both lookups.
- Cached-valid nominal entries must have variant count1..254 before the lookup
  returns an enum identity or indexes FLOW_EN_VARIANT_TOK_X86.

Empty ST is valid. Empty EN admits no entry. These are physical/type-state guards,
not an attempt to tolerate arbitrary corruption of compiler memory. The contract
continues to assume stable, valid Lexer TS/TE spans and source bytes. Root froze
this scope after adding cached-VALID+variant-count0/255/MAX controls.

## Exact final gates

Slurm19782 completed0 in81s on1CPU/8GiB:

| Arm | C++23 result |
| --- | --- |
| Baseline seed9cf, reconstructed byte-identically | 45 failures /104 |
| New private generation1 | 104/104 |
| New private generation2 | 104/104 |

Selfcompile0; corruption controls4/4/3;148 AArch64 source chunks unchanged.
The source/SDK/seed before-and-after manifests passed.

Slurm19783 completed0 in18s. Its independent C++23 oracle checks34 metadata
boundary cases, including MIN/MAX i64, zero, final physical counts, first
out-of-range counts, extreme token/argument indices, and forged cached-valid
entries with variant count0/255/MAX. The predecessor has22 failures and3 bounded
5s timeouts. The final helpers pass34/34 with no out-of-range read attempts.

The component extracts the real helper bodies and replaces metadata-array loads
with bounded diagnostic getters. A getter records an attempted out-of-range
load and returns0 rather than executing undefined memory access. The predecessor
therefore exposes EN257, ST1025 and tokenMAX attempts explicitly; each predecessor
case executes independently so MAX counters cannot hide later cases. The after
component runs the complete sequence. Invalid global counts cause zero array
reads. Instrumentation is private to this component; production source retains
ordinary array accesses protected by the new guards.

This is a finite component qualification, not a claim of arbitrary memory-
corruption resistance. Both its original source file and the104-case producer
gate pin the identical final source SHA below.

## Frozen source and private seed

Source:
16cab9c8700981608469f052a96733360e822db6598221432f438d404cd2913b

Generation1:
7c7f7083e36dc0a49e28701d403ae0b33272721dd0b178fc8473f5cd8696cec7

Generation2 (for root's single sealed modern build):
ccfaec48d1bae176091ef1f3401c2d863c1f05284390f3c50ca9d66e94c7114f

Private generation2 path:
 /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-bounds03/results/generation2-producer.elf

The34-case component is:
 /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-bounds-component02

The full producer gate is:
 /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-bounds03

19769 and19777 remain preserved predecessors. The intermediate31-case component
19775 and104-case gate19777 passed before the final per-entry variant-count guard.
The final104-case gate uses the unchanged104-case computational fixtures/oracle;
the final34-case component separately pins the updated bounds harness.

Root still owns the fresh modern bootstrap and replay of the eleven original
SIG139 inputs, PBPK, loader and current focused gates. No principal bootstrap,
shared binary replacement, lower.sio/AArch64 change, push or merge was performed
by this lane.
