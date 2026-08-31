1. [BLOCKER] Naming “SAN-ViT-large” is false: the model is d=384, 12 blocks, 6 heads (ViT-Tiny/Small class), not ViT-Large (d=1024, 24 blocks).
   <location: §3.1; Abstract; Table 2>
   <why it matters>
   Inflates contribution and misleads any reader comparing to ViT-L/16 literature FLOPs (61.55 GMAC ledger is then an apples-to-oranges constant glued onto a different net).
   <minimal fix>
   Rename everywhere to the actual scale (e.g. SAN-ViT-S/d384) and drop or re-derive the “real-scale ViT-L/16” ledger claim.

2. [BLOCKER] Primary GPU claims (32–52% metered-MAC savings; freeze-on-green “works”) rest on a deliberately trivial regime: 4k/1k CIFAR subset, τ ∈ {0.34, 0.251, 0.165}, acc@t* ∈ {0.39, 0.26, 0.17}, budget 8–10 epochs.
   <location: Abstract; §4.2 Table 2; §5.2>
   <why it matters>
   Freeze at epoch 4 of 8 on a target any random-ish net hits immediately is not evidence the learning rule scales; EarlyStop captures most of the training-MAC cut. Venue (TPDS/FPL/MLSys) will treat this as non-result with a story.
   <minimal fix>
   Either run full CIFAR-10 / ImageNet-scale with competitive τ and multi-seed stats, or demote the paper to a kernel/microbenchmark note and remove architecture-family “savings” from abstract/contributions.

3. [BLOCKER] §4.4 (CIFAR-100) and part of the contribution table are empty (“Results are pending”) while the draft targets arXiv and peer-reviewed systems venues.
   <location: §4.4; §1.2 row 4–5 framing>
   <why it matters>
   Incomplete empirical body; cannot support the cross-family generalization claim.
   <minimal fix>
   Remove §4.4 and all forward references, or wait for finished jobs and freeze the draft.

4. [BLOCKER] Reference list is factually wrong on the hardware/early-exit citations that justify novelty.
   <location: §7 References [5],[6],[11],[12]>
   <why it matters>
   [5] Deep Compression, [6] generic edge review, [11] IoT edge, [12] ShuffleNet are not prior early-exit FPGA/transformer accelerators. Hostile referee will mark related-work as incompetent or padded.
   <minimal fix>
   Replace with actual early-exit / EE-FPGA / EE-transformer papers (e.g. real BranchyNet follow-ons, EE hardware works) or delete the bogus anchors.

5. [MAJOR] Kernel is a bus-bound priority-encoder + LUT accumulate (no FP, no DSP, no trunk). Calling this “first measured deployment of SAN” equates SAN with a 4-way threshold scan.
   <location: Abstract; §1.2; §3.2; §4.1>
   <why it matters>
   Any CPU/GPU can sustain hundreds of Msamples/s on four Q0.15 compares; 511 Msamples/s at 95% of a self-defined peak is expected, not a systems breakthrough. No baseline (AVX/CUDA scan, or end-to-end trunk+PCIe+scan) is reported.
   <minimal fix>
   Add host CPU and GPU scan baselines, full pipeline latency (trunk → PCIe → kernel → host), and rewrite claims to “audit-path microbenchmark,” not “SAN deployment.”

6. [MAJOR] Table 2 verdict column hides contract failures: ViT L6 requires exit-frac > 0.07 but reports 0.03; ResNet L5 is a “tradeoff”; ablation §4.3 maxes at 7/8 and stays L_RED.
   <location: §4.2 Table 2; §4.3; Appendix B L5–L6>
   <why it matters>
   Companion contract is the paper’s own success criterion; burying fails under “PASS; L6 exit-frac 0.03” is reviewer-bait and undercuts “machinery operates correctly across families.”
   <minimal fix>
   Mark L6 FAIL explicitly, drop “across three families” green rhetoric, and align abstract percentages with failed clauses.

7. [MAJOR] Energy figure (~3.3 nJ/sample) is methodologically weak: 1 Hz `xrt-smi` samples, 30 s window, board sensor only; tiny-cohort “load < idle” admitted; no rack/PCIe/host power; no uncertainty.
   <location: §4.1 Energy paragraph; §5.2>
   <why it matters>
   Incremental 1.7 W on a 24 W idle card at 1 Hz cannot support two-significant-figure nJ claims for a duty-cycled scan; production relevance is unproven.
   <minimal fix>
   External power meter or higher-rate telemetry, confidence interval, idle-subtraction protocol, and explicit “kernel-only board ΔP, not system energy” in abstract (or remove nJ from abstract).

8. [MAJOR] No end-to-end SAN inference measurement: trunk stays on host/GPU; reported Msamples/s are scan-only stress streams; ImageNette wall time 41 Msamples/s already shows enqueue/PCIe dominance vs 511 peak.
   <location: §3.2; §4.1; §4.5; §5.1>
   <why it matters>
   Systems paper must show the deployed path that a production SAN actually runs. Audit-only peaks do not establish deployment value.
   <minimal fix>
   Measure closed-loop: host trunk batch → confidence pack → U250 → metered exit decision → reported latency/throughput/energy per image.

9. [MAJOR] Training rule optimizes accuracy ≥ τ then freezes; “patient suffering” (asymmetric C) is not the training constraint—only a post-hoc grid / L5–L7 check. Anti-Goodhart claim is therefore overstated.
   <location: §2.1–2.3; §3; Appendix B L3/L5/L7>
   <why it matters>
   Freezing on a low accuracy proxy is itself a Goodhart path relative to the stated patient channel; compassion grid cannot fix an infeasible patient optimum if τ is the only gate.
   <minimal fix>
   Either train with patient-harm constraints (or constrained-ERM) or rewrite §2 to say “accuracy-gated early stop + post-hoc harm audit,” not two-channel suffering optimization.

10. [MAJOR] Statistical reporting is absent: single seed, one data order, no error bars, no repeated trials; 1.08× latency called out as “within noise” yet still in abstract-adjacent claims.
    <location: §3.3; §4.2 Table 2; Abstract>
    <why it matters>
    40.7/32.0/52.2% and 1.08× are unpublishable as point estimates without variance; CUDA synchronize microbenchmarks on tiny forwards are noisy by the authors’ own admission.
    <minimal fix>
    ≥3 seeds, report mean±std, pre-specify primary metric, drop 1.08× from highlight positions unless significant.

11. [MAJOR] “Bit-exact on real photographs” only checks the integer scan against a Python golden on exported confidences—not that SAN-ResNet-18 exits or FLOP totals match a full trained early-exit policy under the paper’s own meter convention end-to-end.
    <location: §4.5; §3.2 T3>
    <why it matters>
    T3 is necessary but far from sufficient for “deployment soundness” of SAN; trunk, head calibration, and Δ policy are out of band.
    <minimal fix>
    Define and measure an end-to-end conformance test (host SAN forward vs card-metered path on the same images) with explicit tolerance on decision histogram and S_m.

12. [MAJOR] Partial MAC convention (no norms/softmax/residuals/pooling/dispatch/PCIe) is used to headline 32–52% “machine burden” savings while wall-time is ~1× for attention models.
    <location: §2.1; §4.2; §5.2>
    <why it matters>
    Relative savings under a favorable partial meter are not system savings; abstract does not carry the caveat at equal strength to the percentages.
    <minimal fix>
    Lead with wall-time and full-system energy; move partial-MAC % to appendix, or recompute with a fuller profiler (NvTX/Nsight).

13. [MAJOR] Stress cohort “ImageNet-completo-sized” (1.2 M) is synthetic confidences, not ImageNet; abstract/intro still lean on it next to “real-image” language.
    <location: Abstract; §3.2; §4.1; §5.2>
    <why it matters>
    Conflates throughput of random/synthetic packs with vision deployment evidence.
    <minimal fix>
    Label stress cohort as synthetic-only in every highlight; never pair it with ImageNet scale without “synthetic volume” qualifier.

14. [MINOR] Internal inconsistency on default Δ: Table 2 ResNet 40.7% / acc 0.390 vs Table 3 Δ=0.45 → 44.5% / 0.355; GPT/ViT defaults unspecified.
    <location: §4.2 vs §4.3>
    <why it matters>
    Reader cannot reproduce Table 2 or know which operating point the abstract cites.
    <minimal fix>
    State Δ, τ, seed, and job IDs for every Table 2 row; make Table 2 a designated row of Table 3.

15. [MINOR] Theoretical peak math underspecified: 4×Q0.15 = 64b, yet 512b beat claimed as 4 samples (128b/sample)—padding/fields not defined; 135.2 MHz single-build clock treated as characteristic.
    <location: §3.2; §4.1>
    <why it matters>
    95%-of-peak claim is only as good as the packing story; single Vitis build is anecdotal.
    <minimal fix>
    Document record layout bit-exact; report II/fmax from Vitis HLS/impl logs; avoid “95%” without that appendix.

16. [MINOR] Patient cost matrix is synthetic (truck hazard 5/2/1; GPT “negation tokens”) while language of “patient suffering” and “compassion” invites clinical over-read; §7 says “No clinical content” but §2 does not.
    <location: §2.1; §4.2; §7>
    <why it matters>
    Author is MD/Sounio PI; sloppy patient framing is reputational and safety-adjacent if forked into pharma tooling.
    <minimal fix>
    Rename to “task asymmetric loss,” put C in a table, and state non-clinical in §2.1 not only in AI disclosure.

17. [MINOR] Reproduction path requires U250 + XRT + private OrangeFS/Slurm jobs + bitstreams not shipped in the draft; gate script claims I_GREEN 8/8 while GPU L-clauses fail.
    <location: Appendix A; Appendix B; §1.2>
    <why it matters>
    Third party cannot reconstruct from the paper alone; dual contract (L vs I) greens are easy to confuse.
    <minimal fix>
    Publish xclbin hash, host binary hash, cohort digests, and a CPU-only golden path that yields numeric equality to every Table 1–3 cell.

18. [MINOR] Prior-work gap: no comparison to matched early-exit training (BranchyNet/SDN budgeted trains) under the same meter; Dense vs EarlyStop only.
    <location: §1.3; §3.1; §4.2>
    <why it matters>
    Cannot isolate freeze-on-green vs ordinary early-exit effects.
    <minimal fix>
    Add a standard early-exit baseline trained to same τ/budget with same heads.

19. [NIT] Status line claims prior hostile LLM review “addressed blockers”; present draft still has empty §4.4, wrong ViT name, and broken refs—so either review was weak or fixes were not applied.
    <location: header Status; §4.4; §3.1; References>
    <why it matters>
    Undermines process claims in §7.
    <minimal fix>
    Delete process marketing from the header; fix blockers before asserting they are closed.

20. [NIT] Orthography/style: mixed thin spaces in numbers (“5 000”), “ImageNet-completo-sized,” and “pre-silicon in spirit” for a chip that already ran on U250—confusing jargon.
    <location: §3.2; §4.1; §5.2>
    <why it matters>
    Sloppy systems writing invites desk friction.
    <minimal fix>
    Standard EN-US thousands separators; replace “pre-silicon in spirit” with “single bitstream, no DSE.”
