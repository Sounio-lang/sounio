# Technical Review of the Target-23 Two-Return Taylor-41 Checkpoint

## Internal Consistency Checks (all pass)

| Check | Expected | Stated | ✓ |
|---|---|---|---|
| Steps × h | 1686 / 256 | 6.5859375 | ✓ |
| Partition split | 843 + 843 | 843 + 843 | ✓ |
| Output words | 843×10 × 2 | 16,860 | ✓ |
| Word width | 1+31+192 | 224 bits | ✓ |
| Bisections per event | 50 − 8 = 42 | 42 each, 84 total | ✓ |
| Bracket width | 2⁻⁵⁰ ≈ 8.882×10⁻¹⁶ | matches bracket spans | ✓ |
| Contraction bound | < 1 required | 0.2073 | ✓ |
| Normal velocity sign | > 0 for neg→pos crossing | 32.05, 54.36 | ✓ |

The arithmetic is self-consistent throughout.

## What Is Genuinely Strong

**1. The event isolation is tight and well-formed.** The spatial radius ~2.77×10⁻¹⁶ translated through the normal velocities gives a theoretical time resolution floor of ~8.6×10⁻¹⁸ (event 1) and ~5.1×10⁻¹⁸ (event 2). The 2⁻⁵⁰ stopping criterion sits well above these floors, so the brackets are comfortably decidable—the sign determination is not marginal.

**2. The rounding-error bound is credible.** With 192 fractional bits (unit roundoff ~1.9×10⁻⁵⁸), a degree-40 Taylor polynomial evaluated over ~10 terms per step, accumulated across 1,686 steps, a local rounding radius of ~1.57×10⁻⁴⁷ is in the right ballpark—it is roughly 2⁻¹⁵⁵, which is about 2⁻¹⁹² scaled up by the accumulation factor one would expect from a few hundred operations per step.

**3. The exp majorant construction is sound.** Truncating exp(·) at degree 32 and bounding the tail with a geometric series starting at degree 33 is valid provided the argument max(μ,0)·h < 33. With h = 2⁻⁸ ≈ 0.0039, this requires μ < ~8,500, which is almost certainly satisfied for a smooth ODE on this timescale. The upward rounding then makes the amplification factor rigorous.

**4. The claim discipline is excellent.** The explicit enumeration of what is *not* proved (whole orbit, all leaves, H-PG, V7-B, novelty, promotion, open problem) is the right practice. The distinction between HLS CSim, HLS synthesis, and physical execution is stated correctly and honestly.

## Points Warranting Clarification or Scrutiny

**A. The "two-return" language.** The integration reaches t = 6.5859375, and the second event bracket is [6.58513628…, 6.58513628…]. This means event 2 occurs at approximately step 1685.86—barely inside the final step. Three questions:

- Is there a *third* crossing shortly after t = 6.5859 that was cut off by the integration horizon? If the Poincaré return time is periodic, the next return might be close.
- Was the stopping time chosen to be just past event 2, or is it coincidental?
- Does the trajectory tube at the final state overlap meaningfully with the initial box? If so, this is nearly a full period and the "bounded two-return" language understates it. If not, the two events may be interior crossings that do not constitute returns in the Poincaré-map sense.

**B. The DSP reduction (22,593 → 2,921).** A 7.7× reduction through "explicit multiplier and function sharing" is dramatic. For context, the naïve count for 224-bit fixed-point multiply-accumulate would be roughly (224/18)² ≈ 155 DSPs per full-precision multiply if using 18×25 DSP blocks, times the number of simultaneous operations. The sharing presumably time-multiplexes a smaller DSP pool across the Taylor coefficient recurrence. The report should state:

- How many cycles per step the shared design requires (latency directly affects whether the 100 MHz target is meaningful for throughput).
- Whether the CSim matched the *functional* output only, or also the scheduled cycle-accurate behavior.

**C. The partition boundary claims to carry "prior-event count explicitly."** This is important: it means partition 2 knows that event 1 already occurred. Does the integration logic branch on event count (e.g., different arm state, logging behavior)? If so, the partition boundary is not just a state checkpoint but also a control-flow checkpoint, and the verifier should confirm that the event-arm state transition is encoded unambiguously in the transcript.

**D. The geometric majorant for exp.** The statement "beginning at degree 33" implies the ratio test gives a contraction factor < 1 at that point. For the majorant to be valid, one needs the argument x = max(μ,0)·h to satisfy x/(k+1) < 1 for k ≥ 33, i.e., x < 34. This is trivially true for reasonable μ, but the report should state the actual μ used and confirm it was computed with directed rounding.

**E. "84 decisive event bisections" and the mutation suite.** The mutation suite "rejected 9/9 alterations" including changes to "an event normal." This is good, but 9 mutations is a small suite. What specific mutations were tested? In particular:

- Was a mutation that *flips the sign of the normal velocity* tested?
- Was a mutation that *widens the bracket to 2⁻⁴⁹* (just above the stopping criterion) tested?
- Was a mutation that *swaps the two event brackets* tested?

These are the most likely failure modes for an event locator.

## Assessment of the Hardware Claim Boundary

The report is admirably explicit:

> "HLS synthesis alone is not counted as physical FPGA execution."
> "dual-card physical execution remains false even if both partitions pass sequentially on one card."

The estimated 7.300 ns clock (136.99 MHz) at 100 MHz target with 23% DSP utilization is a reasonable post-synthesis estimate, but without place-and-route results, the actual Fmax could be materially lower due to routing congestion—especially with 277K LUTs and 300K FFs. The retained negative result (22,593 DSPs) is good practice.

## Summary

This is a well-structured checkpoint with strong claim discipline and internally consistent arithmetic. The mathematical machinery (Taylor-41 with Picard contraction, normalized-s coefficient storage, directed-rounded exp majorant, strict-sign bisection) is sound. The principal open questions are:

1. Whether the two events constitute genuine Poincaré returns or are interior section crossings.
2. The throughput cost of the DSP sharing and whether cycle-accurate behavior was verified.
3. Whether the mutation suite covers the highest-risk failure modes.
4. When physical (not HLS-simulated) execution will be available.

The identified next transitions—dependency-aware set representation for spatial coverage, and true dual-card replay—are the correct priorities.
