# Target-23 proof-carrying two-return Taylor-41 chain

Date: 2026-08-04

## Result

This checkpoint composes the prior target-23 Picard and scaled-Taylor work into
a regenerable chain of 1,686 validated steps. The chain uses signed
S1.I31.F192 state arithmetic, fixed `h=2^-8`, Taylor polynomial degree 40, and a
componentwise order-41 Lagrange remainder. It reaches time `6.5859375` and
isolates two strict negative-to-positive crossings of the section `w=0`.
Here "return" means a directed re-entry through that same section. The horizon
is the first fixed-grid endpoint after the second re-entry; no third crossing,
period closure, or overlap of the final box with the initial box is asserted.

The independently reconstructed event intervals are:

| Event | Time bracket | Normal velocity interval |
|---:|---|---|
| 1 | `[2.4075263046191643923066294519230723381, 2.4075263046191652804850491520483046770]` | `[32.0477746305846367198904735198462, 32.0477746305846933207051503471701]` |
| 2 | `[6.5851362857924939930853724945336580276, 6.5851362857924948812637921946588903666]` | `[54.3578188844193705101614757607780, 54.3578188844194789890551192473399]` |

Each bracket has exact width `2^-50`; both normal-velocity lower endpoints are
strictly positive. The largest propagated scalar radius is approximately
`2.7693836525253493e-16`. The largest local Taylor-and-rounding radius is
approximately `1.570436766083853e-47`, and the largest Picard contraction bound
is approximately `0.2073255406705322`.

## Why the chain is rigorous

The Taylor coefficients use the normalized variable `s=t/h`. The recurrence
multiplies every new coefficient by `h/(n+1)`, so the stored coefficient already
contains `h^n`; evaluating the endpoint at `s=1` is therefore their sum. A
strict Picard self-map bounds the complete trajectory tube for each step and
supplies the state box used to enclose the order-41 Lagrange term.

The previous scalar radius is amplified by an upward-rounded upper bound on
`exp(max(mu,0)h)`. That bound sums terms through degree 32 and encloses all
remaining terms with an explicit geometric majorant beginning at degree 33.
The largest directed-upward argument in the chain is approximately
`max(mu,0)h = 0.1640334381045432`, far below the majorant's required `x < 34`.
The local Taylor and directed-rounding radius is then added outward.

The event locator bisects only when the midpoint enclosure has a strict sign.
It finally rechecks a strict negative lower endpoint, a strict positive upper
endpoint, width at most `2^-50`, a Picard tube over the remaining time bracket,
and strictly positive `x*y-zs` throughout that tube.

## Independent evidence

The verifier does not import the generator. It reconstructs the exact-rational
interval operations, all 1,686 Picard/Taylor steps, both events, and the binary
transport transcript. It verified all 1,686 steps and 84 decisive event
bisections. The mutation suite rejected 12/12 alterations, including coherently
rehashed changes to a chain center, a normal endpoint, a sign-flipped normal, a
bracket widened beyond `2^-50`, swapped events, and a hardware output.

HLS CSim under Vitis 2025.1 on VM100 executed both 843-step partitions and
matched all 16,860 signed 224-bit words exactly. The initial unshared HLS
architecture generated RTL but required 22,593 DSPs, or 183% of one U250. That
negative result is retained. Explicit multiplier and function sharing reduced
the final estimate to:

| Measure | HLS estimate |
|---|---:|
| Target clock | 10.00 ns / 100 MHz |
| Estimated clock | 7.300 ns |
| Estimated Fmax | 136.99 MHz |
| BRAM_18K | 3 |
| DSP | 2,921 / 12,288 (23%) |
| FF | 300,585 / 3,456,000 (8%) |
| LUT | 277,010 / 1,728,000 (16%) |
| URAM | 0 |

These figures are HLS estimates, not placed-device utilization. Vivado's
out-of-context synthesis of the same kernel reported 1,366,018 CLB LUTs
(79.05%), 1,157.5 block-RAM tiles (43.06%), and 3,211 DSPs (26.13%). The full
placed design reported the following stronger device-level evidence:

| Measure | Placed result |
|---|---:|
| Kernel LUT | 1,362,459 / 1,618,800 user budget (84.16%) |
| Kernel register | 627,001 / 3,273,764 (19.15%) |
| Kernel BRAM | 1,157 / 2,503 (46.22%) |
| Kernel DSP | 3,211 / 12,281 (26.15%) |
| Inter-SLR connections | 30,288 |
| SLR0 / SLR1 / SLR2 / SLR3 LUT | 92.64% / 60.63% / 93.85% / 93.06% |

The full 100 MHz platform design could be placed, but physical optimization
remained at approximately `WNS=-118.16 ns`; that attempt was stopped and
retained as a negative timing-closure result. A 5 MHz link request was rejected
because the platform's supported minimum is 10 MHz. The 10 MHz link reused the
reviewed 100 MHz HLS schedule and completed placement, but reported
`WNS=-121.338 ns`. The run was stopped before routing because this deficit
was too large to treat the remaining physical-optimization passes as a
credible closure path. No xclbin was produced.

For this reviewed 100 MHz-scheduled XO, lowering only the link clock did not
repair the physical result. The next hardware design should break the long
combinational dependency chain: retain exact checkpoints in BRAM or DDR,
execute bounded proof stages as separate kernels or pipeline regions, and
verify the transcript at each boundary. A new attempt is accepted only with
routed `WNS >= 0` at a platform-supported frequency and exact XRT replay of all
16,860 output words.

The shared schedule trades area for time. HLS gives a conservative maximum of
3,778,879,861 cycles for a partition because its static bound permits expensive
event localization on every step; the executed transcript triggers it only
twice over both partitions. CSim establishes functional bit equality, not
cycle-accurate RTL behavior. Routed timing and physical wall time are separate
obligations.

The HLS CSim and estimate are green, but physical timing closure, physical FPGA
execution, and dual-U250 execution are false. The retained negative receipts
include both placed reports and both link attempts; HLS synthesis alone does
not satisfy the physical obligation.

## Two-card boundary

The transcript is split at the exact verified checkpoint after step 843. Each
partition has 843 steps and 8,430 output words. Partition 1 carries its starting
center, radius, time, event-arm state, and prior-event count explicitly, so the
two outputs concatenate without a hidden state conversion.
The independent verifier reconstructs this boundary and requires the encoded
arm state and prior-event count to agree with the first partition's history.

This partitioning supports concurrent replay on two identical U250 kernels once
two devices are visible. It is not an online parallel integration algorithm:
partition 1 starts from the already verified boundary checkpoint. On this run,
the cluster exposed one U250 at `0000:d8:00.1` on `dl380-proxmox`; no second
U250 was enumerated on the other cluster nodes. Therefore dual-card physical
execution remains false even if both partitions pass sequentially on one card.

## Claim boundary

This is a bounded two-return flow certificate for one frozen target-23 initial
box. It materially advances the earlier one-step result, but it does not prove
the whole periodic orbit, all points of leaf 331, all adaptive leaves, global
H-PG, V7-B eligibility, novelty priority, promotion, or an open problem.

The next mathematical transition is spatial rather than temporal: replace the
single scalar-radius initial box with a covering family or a dependency-aware
set representation and prove that the return construction covers the entire
target leaf. The next machine transition is a staged timing-clean kernel,
followed by single-card XRT replay; true dual-card replay additionally requires
the second U250 to be installed and visible to XRT.
