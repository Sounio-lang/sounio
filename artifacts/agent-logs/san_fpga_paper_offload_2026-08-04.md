=== Task: review | Provider(s): deepseek xai gemini ===
=== LLM Offload Fan-Out ===
Output dir: /tmp/llm-offload-RCqmzL
Providers: deepseek xai gemini

  -> Sending to DeepSeek deepseek-v4-pro (deepseek-v4-pro, max=8192, timeout=180s)...
  -> Sending to Grok grok-4.5 (grok-4.5, max=8192, timeout=180s)...
  -> Sending to Gemini 2.5 Pro (google/gemini-2.5-pro, max=2048, timeout=180s)...
  <- Gemini 2.5 Pro: ERROR (see /tmp/llm-offload-RCqmzL/gemini.json)
  <- DeepSeek deepseek-v4-pro: ERROR (see /tmp/llm-offload-RCqmzL/deepseek.json)
  <- Grok grok-4.5: DONE (11600 bytes)

=== Results ===

━━━ grok ━━━
```
1. [BLOCKER] Reported accuracies at freeze (ResNet 0.390, ViT 0.262, GPT 0.167) are far below any credible feasibility target τ for the claimed tasks, so “freeze-on-green” is met on a trivial/near-random bar and the 32–52% “suffering” savings are mostly early-stopping a failed run.
   <location: §4.2 Table 2; §2.1–2.3>
   <why it matters: A hostile POPL/MLSys/TPDS referee will treat the entire training study as invalid if τ is not stated and the achieved accuracy is not competitive; gratuitous-FLOP claims require that the model actually solved the task.>
   <minimal fix: State τ explicitly per family; show full learning curves vs Dense/EarlyStop through budget B; either raise τ to a non-trivial level or reframe claims as “early-stop under a weak constraint,” not task-feasible SAN.>

2. [BLOCKER] Sections 4.3 and 4.4 are empty placeholders (“To be populated…”) while the abstract, contributions table, and conclusion assert a complete measured study ready for arXiv/TPDS/FPL/MLSys.
   <location: §4.3; §4.4; Abstract; §1.2; §6>
   <why it matters: Draft status is declared, but venue-facing claims and “every empirical claim is backed” are incompatible with missing ablations and larger-proxy results; this is reviewer-bait for reject-as-incomplete.>
   <minimal fix: Remove or quarantine empty sections from any submission draft; strip abstract/contribution sentences that depend on them; or finish the runs before claiming completeness.>

3. [BLOCKER] Bit-exactness is shown only for the catastrophe-scan/FLOP-meter kernel on pre-exported confidences, not for an end-to-end SAN inference path (trunk + exits + decision) on the U250.
   <location: §3.2; §4.1; §4.5; Abstract “first measured deployment of a SAN”>
   <why it matters: The systems claim is a “complete pipeline” and “SAN deployment”; what was measured is a tiny integer priority-encoder + LUT accumulator. Calling that a SAN deployment overreaches and will be attacked immediately at FPL/MLSys.>
   <minimal fix: Rewrite claim to “measured deployment of the SAN exit-audit/FLOP-metering kernel, trunk on host/GPU”; or actually close the loop with on-card or tightly coupled trunk exits.>

4. [BLOCKER] Machine-suffering accounting explicitly leaves biases, norms, activations, softmax, residuals, and pooling unmetered, and equates training backward = 2× forward, while reporting precise percent savings and nJ/sample as if the meter were exact.
   <location: §2.1; §4.1–4.2; Abstract>
   <why it matters: Relative savings under a partial conventional meter are not “exact FLOP-metering”; unmetered ops and host/PCIe/dispatch dominate the regimes where wall-time speedup collapses to ~1× (ViT/GPT). Clinical/systems readers will call the energy and savings figures misleading.>
   <minimal fix: Rename to “metered-MAC proxy”; publish full op mix and sensitivity; measure end-to-end host+device J/sample; stop saying “exact” for a partial ledger.>

5. [MAJOR] Abstract and §4.1 disagree on peak efficiency: abstract says 511 Msamples/s is 87% of 540.8 Mpeak; body says 95% (and 511/540.8 ≈ 94.5%).
   <location: Abstract; §4.1>
   <why it matters: Internal inconsistency on the headline systems number; destroys trust in every other reported figure.>
   <minimal fix: Recompute once from raw bench logs; use one percentage everywhere; cite the artifact row that produces it.>

6. [MAJOR] “Sustained” 511 Msamples/s is demonstrated on a synthetic 1.2 M stress cohort; real ImageNette validation reports 122.2 sustained / 41.2 kernel-only Msamples/s — orders of context that the abstract elides.
   <location: Abstract; Table 1; §4.5>
   <why it matters: Headline throughput is best-case microbench, not application sustained rate; classic accelerator-paper overclaim.>
   <minimal fix: Lead with real-cohort rates; move 511 to “peak stress microbench”; report PCIe/enqueue amortization explicitly.>

7. [MAJOR] Energy 3.3153 nJ/sample is incremental board-level ΔP from xrt-smi at 1 Hz over 30 s, with a noted anomaly that tiny cohorts read below idle — methodology is too coarse for three-decimal-nJ claims.
   <location: §4.1; Abstract; §5.2>
   <why it matters: 1 Hz sampling cannot support nJ/sample precision; below-idle readings indicate measurement noise or power-state aliasing; reviewers will discard the energy number.>
   <minimal fix: Use higher-rate power rails / external meter; report confidence intervals and idle/load distributions; round to significant figures the instrument supports; drop “3.3153” false precision.>

8. [MAJOR] Patient-channel cost matrix and hazard structure (CIFAR truck cost 5/2/1; GPT “negation tokens”) are asserted without validation that they match any clinical or operational harm model, yet the paper brandishes “suffering-aware” and patient harm as first-class results.
   <location: §2.1; §4.2 L5 tradeoff; Abstract>
   <why it matters: Without grounded C and pre-specified L5 clauses, “patient suffering” is an arbitrary weighted error rate; the disclosed ViT tradeoff cannot be interpreted as safety-relevant.>
   <minimal fix: Justify C from external source or sensitivity-sweep C; pre-register L-clauses; avoid clinical connotations (“patient”) unless a real clinical cost model is used.>

9. [MAJOR] Contribution #1 and #4 cite Slurm jobs and scripts as MEASURED evidence, but the draft does not embed seeds, τ, Δ, B, hardware SKU mapping per job, or raw metric dumps — reproducibility from “supplied files alone” fails for a third party with only this paper.
   <location: §1.2; §3.3–3.4; §4.2; Appendix A>
   <why it matters: Appendix A gives three commands; not job configs, bitfile hash, XRT version, or result JSON. Authority block points at a companion spec not included in the artifact body.>
   <minimal fix: Ship a frozen results tarball (configs, logs, xclbin hash, power CSVs, golden diffs) and cite paths; pin seeds/τ/Δ/B in the tables.>

10. [MAJOR] SAN-GPT “dataset” is next-token prediction on the repository’s own research documentation (vocab 2000) — contamination and non-standard benchmark make 52.2% savings and “L_GREEN 8/8” uninterpretable.
    <location: §3.1; §4.2>
    <why it matters: Training on the paper’s own docs is circular; no held-out external language benchmark; hostile referee will discard the GPT column.>
    <minimal fix: Use a public LM benchmark with held-out split; report perplexity/acc baselines; or drop GPT from headline claims.>

11. [MAJOR] Clock 135.2 MHz and “no DSPs/multipliers” are presented as virtues, but there is no area/LUT/BRAM/URAM report, no comparison to CPU/GPU scan, and no HLS vs RTL or alternative-architecture baseline — so efficiency claims lack a denominator.
    <location: §3.2; §4.1; §5.2>
    <why it matters: FPL/TPDS reviewers ask “vs what?” first; a priority encoder at 135 MHz on U250 may be trivial fabric use.>
    <minimal fix: Add post-place resource table, power breakdown, and throughput/W vs host AVX/GPU kernel on the same cohorts.>

12. [MAJOR] Anti-Goodhart / compassion-grid / NO_FEASIBLE gate is described as executable in the companion contract, but Table 2 shows ResNet/ViT L5 tradeoffs and still reports savings — unclear whether infeasible patient points were excluded from selection.
    <location: §2.2; §4.2 verdict column; §1.2>
    <why it matters: If checkpoints that worsen patient harm are still presented as SAN wins, the central selection rule is not enforced in the results.>
    <minimal fix: Publish gate outcomes per run; define dominance rules; do not headline multi-objective wins when L5 fails.>

13. [MAJOR] Wall-clock “1.08× speedup” for ResNet (0.196 vs 0.213 ms) is within likely measurement noise for CUDA synchronize microbench on small CIFAR models; ViT/GPT show 0.99×.
    <location: §4.2; Abstract>
    <why it matters: Abstract elevates 1.08× as a measured result; without repeated trials, CIs, or batch/latency protocol, it is not credible.>
    <minimal fix: Report N-run mean±std, batch size, warmup, and statistical test; remove 1.08× from abstract unless significant.>

14. [MAJOR] Real-image path uses SAN-ResNet-18 with frozen ImageNet-1k backbone fine-tuning layer4+heads on 4000 ImageNette photos, while the GPU study is SAN-ResNet-50 from scratch on CIFAR — architectures and protocols are not comparable, yet both are folded into one “SAN” narrative.
    <location: §4.5; §3.1; §4.2>
    <why it matters: Contribution #5 does not validate the same model family or training rule as #1/#4; risk of cherry-picked easier proxy.>
    <minimal fix: Align architecture and training protocol across GPU and FPGA sections, or clearly separate “kernel validation on exported confidences” from “SAN-ResNet-50 training study.”>

15. [MINOR] Theoretical peak math “512 bits/cycle × 4 samples/beat × 135.2 MHz = 540.8 Msamples/s” assumes full bandwidth and packing efficiency never stated (HBM/PCIe path, outstanding transfers, host supply rate).
    <location: §3.2; §4.1>
    <why it matters: Bus-limited peak may be miscomputed if platform port width/clock or multi-SLR routing differs; 87/95% utilization then meaningless.>
    <minimal fix: Cite Vitis interconnect report and measured host feed rate; define peak from the actual platform port.>

16. [MINOR] Prior-work citations are misaligned (e.g., [5] Deep Compression, [6] edge review, [10] adaptive NNs, [12] ShuffleNet) and do not substantiate “hardware-aware early exit” or FPGA early-exit CNN/transformer claims as used in §1.3.
    <location: §1.3; References>
    <why it matters: Signals shallow scholarship; referees in cs.AR will notice immediately.>
    <minimal fix: Replace with actual early-exit hardware papers and correct FPGA/EDL citations; re-read and re-cite.>

17. [MINOR] Status line says every empirical claim is backed by a measured artifact, but threshold ablation, larger proxies, rack-level energy, and ImageNet-1k are openly missing — self-contradictory framing.
    <location: Status; Abstract; §5.2; §4.3–4.4>
    <why it matters: Undermines the paper’s “honesty as contribution” rhetoric when the front matter over-promises.>
    <minimal fix: Narrow the guarantee to claims marked MEASURED in §4.1–4.2 and §4.5 only.>

18. [MINOR] Companion contract verdict language (L1–L8, I_GREEN, HOST_SAN_SCAN_PASS) is undefined in the paper body, so Table 2 “verdict” column is opaque to external readers.
    <location: §4.2; Appendix A; §2.2>
    <why it matters: Non-reproducible acceptance criteria; dissertation defense will ask for the clause definitions first.>
    <minimal fix: Add a table of L-clauses with predicates and thresholds in the main text.>

19. [NIT] Orthography/meta claims “EN-US” yet hyphenation and coinages (“ImageNet-completo-sized,” “Mercyful”) are inconsistent; date/meta last_validated in 2026 may confuse archival readers.
    <location: docs:meta; §3.2; §1.3>
    <why it matters: Minor credibility friction only.>
    <minimal fix: Normalize terminology; explain project-internal codenames on first use.>

20. [NIT] AI disclosure cites GAIDeT-ICMJE 2025 and offload models for a non-clinical cs.LG/cs.AR draft — irrelevant standard noise that invites process questions without adding evidence.
    <location: §7>
    <why it matters: Distracts; does not improve scientific verifiability.>
    <minimal fix: One sentence on tool use + artifact hashes; drop clinical disclosure frameworks.>
```

Raw JSON: /tmp/llm-offload-RCqmzL/
