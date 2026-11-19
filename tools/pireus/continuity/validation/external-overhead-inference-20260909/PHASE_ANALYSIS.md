# Post-hoc phase analysis of the closed 11976 / 11977 pair

This is diagnostic analysis of immutable failed-screen evidence. It does not
change the v2 acceptance criteria or qualify the external observer.

## Sampling-gap mechanism

The rank-0 maximum start-to-start gap was 610,521,254 ns. Its preceding sample
took 410,233,356 ns, including 408,327,089 ns in process_smaps_rollup. The remaining
200,287,898 ns includes the configured 200 ms post-sample sleep and unmeasured
loop/scheduling overhead. The frozen observer explicitly sleeps after completing
each sample, so expensive reads add to the configured interval.

Both ranks' largest gaps occur after lifecycle OBSERVER_START and before the
first EXTEND_ENTRY. The records identify a slow smaps read plus post-sample wait
as the immediate mechanism for rank 0's cadence violation. They do not identify
why the kernel read was slow, observer CPU consumption, or a cause of the later
guardian stop.

A prospective cadence repair can schedule against monotonic deadlines and
account for time already spent reading. It must avoid catch-up bursts, preserve
identity checks and report actual gaps. This would reduce the avoidable wait
after slow reads, but cannot guarantee 500 ms gaps when a read itself is slow.
Such a change is not implemented or qualified by this report. It changes sampling
frequency and requires a new profile and controls, including observer-process
resource accounting.

## Memory: comparable phase hooks

At 13 paired hooks per rank (DECODE_ENTRY 0..6 and REFERENCES_RELEASED 0..5),
recorded CUDA allocated bytes are identical between arms. At these hooks,
rank-1 target-process PSS differences range from -4,276,224 to +3,760,128 bytes,
while observed host available memory is 188,657,664 to 539,422,720 bytes lower.

Already at DECODE_ENTRY index 0, rank 1 has 351,436,800 fewer available host bytes
than baseline, although its PSS is 1,507,328 bytes lower and CUDA allocated bytes
match. This rules out treating the availability difference as directly measured
growth in those two counters. It does not rule out other allocations, unified
memory effects, observer resources, host processes or cache/accounting effects.

From the first DECODE_ENTRY to REFERENCES_RELEASED index 5, observed rank-1
available memory falls 562.918 MiB and PSS rises 37.945 MiB. At the same hooks in
baseline, available memory falls 494.227 MiB and PSS rises 40.547 MiB. These
phase-matched observations provide no evidence of a large additional main-process
PSS accumulation in the observed arm. They are not a proof that no leak exists.

The full JSON retains host fields, PSS components, CUDA counters and preceding
external cgroup anon/file/kernel snapshots, their age and sample duration.
External snapshots are not simultaneous with hooks. At the minimum external
rank-1 host-availability sample, available memory is 35,453,714,432 bytes; this
does not capture the guardian's lower 35,426,271,232-byte stop sample. No
interpolation or summation of overlapping PSS/CUDA/host/cgroup views is used.
No OOM counter changes were seen in the recorded task-cgroup samples; ancestor
cgroup limits and observer-process RSS were not collected.

## Token divergence

The frozen input bundle hashes match both launch receipts. Request index 3 uses
temperature=0.7, seed=3, max_new_tokens=4096 and stop token 200006. The frozen
runtime passes the request seed to SamplingParams and torch.manual_seed, and
rank zero broadcasts the selected token to both ranks.

Both outputs contain 121 tokens. At offset 76 they differ (19 versus 17); the
within-job rank outputs still match. This establishes cross-run token divergence
under the recorded seeded sampling configuration. It does not establish changed
inputs, a broken rank broadcast, decoded semantic impact or a numerical cause:
no logits are available. The original equality criterion remains failed.

## Next implementation boundary

1. Implement deadline-based external sampling as a new source/profile, retaining
   measured gap reporting and preventing catch-up bursts.
2. Add a local timing control with injected slow reads; test that time spent in
   reads reduces the subsequent wait and that overruns remain explicit.
3. Qualify observer resource accounting and cadence separately before any
   loaded-model comparison. Preserve guard 33 GiB, floor 32 GiB, frozen v2
   receipts and no-retry semantics.
4. Keep token divergence as a separate investigation. Do not relax the frozen
   equality gate or infer logits/semantic changes from token IDs alone.

Reproduce without hardware from the repository root:

    python3 tools/pireus/continuity/validation/external-overhead-inference-20260909/phase_diagnostic.py

The script first revalidates the pinned negative closure and input bundle, then
reproduces phase-diagnostic.json. A successful replay is diagnostic integrity,
not experiment acceptance. Source runtime files in the archive remain unchanged.
