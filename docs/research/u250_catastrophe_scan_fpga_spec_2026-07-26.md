<!-- docs:meta
topic_id: repo.docs.research.u250-catastrophe-scan-fpga-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.u250-catastrophe-scan-fpga-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# AMD Alveo U250 catastrophe-scan accelerator — FPGA design spec (pre-hardware)

**Date:** 2026-07-26
**Status:** `EXECUTABLE` for the contract (gate green 2026-07-26); `ESTIMATE` for all hardware numbers (no FPGA installed, nothing synthesized)
**Parents:** `docs/internal/garden/seeds/2026-07-26-fpga-acceleration-opportunity.md` (the seed: two U250s arriving), `docs/research/routon_zd_spec_2026-07-26.md` (2-cycle criterion), `docs/research/l8_zd_census_benchmark_spec_2026-07-26.md` (L8 CPU baselines), `docs/research/l9_zd_census_spec_2026-07-26.md` (measured L9 CPU baselines, Z(9) confirmation)
**Reference outline:** `hardware/fpga/u250_catastrophe_scan/` (Vitis HLS kernel + XRT host, unsynthesized)
**Executable contract:** `scripts/research/fpga_census_kernel_model.c` (bit-accurate kernel model)
**Gate:** `scripts/ci/fpga_catastrophe_scan_gate.sh`

---

## 1. What this is

Preparation for the two AMD Alveo U250 cards announced in the garden seed: an HLS/RTL design for accelerating the **catastrophe scan** — the canonical zero-divisor census of the Cayley–Dickson tower — plus a CI-gated, bit-accurate software model of the kernel that the hardware must reproduce.

Three deliverables:

1. **Design.** A Phase-1 census kernel (fully specified, §3) and a Phase-2 exact-verification kernel (outline, §3.4). Target level: **L9** (512-dim algebra, `Z(9) = 249084`, confirmed in the L9 census spec).
2. **Bit-accurate contract.** `fpga_census_kernel_model.c` models the kernel datapath bit-for-bit (sign-as-bit encoding, index-XOR permutation as pure rewiring, popcount, exact nullity). The CI gate requires it to reproduce the audited integer criterion pair-by-pair at every level `b = 4..9` — including the L8 histogram — so any later HLS/RTL implementation has an exact golden reference.
3. **Independent L9 re-verification.** Running the contract at L9 reproduces `Z(9) = 249084` exactly (124,542 index pairs × 2 signs), agreeing pair-by-pair (all 130,305 candidates) between the model's two internal paths — the audited integer criterion and the bit-parity hardware datapath. The third out-of-sample confirmation of the growth law itself belongs to `docs/research/l9_zd_census_spec_2026-07-26.md` (parallel lane, with a full GF(65521) audit of all 260,610 pair-signs); the new content here is that the **hardware datapath** is exhaustively verified against the audited criterion at L9 as well.

All speedup and resource figures below are **estimates** derived from the U250 data sheet and first-order datapath sizing, not measurements. They exist to size the design before the cards arrive and will be replaced by synthesis/benchmark data.

---

## 2. The scan being accelerated

For `a = e_i ± e_j` in the level-`b` Cayley–Dickson algebra (`n = 2^b`), the audited exact criterion of `scripts/research/routon_zd_contract.py` is:

```
l = i ⊕ j;   p(k) = S[i,k]·S[j,k]·S[i,k⊕l]·S[j,k⊕l] ∈ {+1,−1}
(i, j) is a canonical ZD pair  iff  some p(k) = +1
nullity(L_a) = #{k : p(k) = +1} / 2        (exact; both signs simultaneously)
```

**Hardware encoding.** Signs are bits (0 = +1, 1 = −1); sign multiplication is XOR. With packed rows `sb[i]` and the difference vector `d = sb[i] ⊕ sb[j]`, the criterion collapses to:

```
p(k) = +1  ⇔  d[k] ⊕ d[k⊕l] = 0
v = d ⊕ perm_l(d);   bad = n − popcount(v);   nullity = bad >> 1
```

`perm_l` (bit `k` ← `d[k⊕l]`) is a **pure wiring permutation**: a `b`-stage conditional-swap mux network with zero arithmetic. The entire per-pair datapath is therefore: two row reads, one 512-bit XOR, one 512-bit permutation (wires + muxes), one 512-bit XOR, one popcount, one shift. **No multipliers, no DSPs, no floating point, no approximation** — the FPGA computes exactly the same integer criterion that the CPU contracts audit.

Workload at L9: `C(511, 2) = 130,305` candidate index pairs (signs come for free), each one wide-vector operation. The census is embarrassingly parallel over pairs: PEs take disjoint contiguous pair ranges.

---

## 3. Architecture

### 3.1 Phase 1 — census kernel (fully specified)

Per PE (`hardware/fpga/u250_catastrophe_scan/krnl_census.cpp`):

- **Sign-table replica**: `n × n` bits = **32 KB at L9** (512 rows × 512 bits), on-chip, dual-port BRAM → 2 row reads/cycle (`stab[i]`, `stab[j]`).
- **Pair iterator**: two-counter walk `(i, j)`, advanced once per cycle — no divider.
- **Datapath**: `perm_l` mux network (9 stages × 512 muxes), XORs, 512-bit popcount (carry-save adder tree). Target **II=1 at 250 MHz**.
- **Aggregation**: per-PE nullity histogram (257 bins × 32-bit) and per-label fiber counters (512 × 32-bit) in BRAM; partials summed on the host.
- **Host** (`host.cpp` outline): builds + packs the sign table once (~ms), DMAs 32 KB to the card, enqueues, reads back `16 PEs × (257 + 512 + 1) × 4 B ≈ 49 KB`, verifies against the growth law and the gated CPU model.

Configuration: **16 PEs** at L9. The table replica is the BRAM cost driver; the datapath is the LUT cost driver; both are small (§4).

### 3.2 Scaling to L10

`n = 1024`: rows are 1024-bit, table = 128 KB as bits (≈ 29 BRAM36 or 4 URAM per replica), pairs = `C(1023, 2) = 522,753`. Same kernel with `BITS=10`; either 8 wide PEs or 16 PEs with a 512-bit time-multiplexed datapath. Everything in this spec scales by powers of two; no redesign.

### 3.3 What the kernel deliberately does NOT do

No SVD, no floating point anywhere. The original `docs/research/catastrophe_cd.py` SVD scan is superseded at the contract level by the exact 2-cycle criterion (L7 spec §method, cross-checked in clause C8); the FPGA accelerates the exact criterion, not the numerical oracle.

### 3.4 Phase 2 — exact-verification kernel (outline only)

The CPU benchmarks show verification, not census, dominates wall clock (L8: 6.1 s exact GF(65521) rank vs 0.010 s census; L9, **measured**: 153.9 s vs 0.109 s — L9 census spec §3). A Phase-2 kernel would run **generic Gaussian elimination over GF(65521)** on `M(sgn) = I + sgn·Q` — the standard GE algorithm with no use of the `2×2`-block decomposition (that decomposition is the *proof* that GF-rank = Q-rank, not a shortcut the verifier may use, per the L8 spec §2). Independence of the audit is about the *algebra*; exploiting the matrix's *sparsity* is an implementation matter, exactly as the CPU verifier already does with its standard zero-entry skip: `M(sgn)` has `O(1)` nonzeros per row and **provably zero fill-in**, so a nonzero-tracking GE performs `O(n)` element updates per pivot and `O(n²)` per matrix. Elements are 16-bit; modular reduction mod `65521 = 2^16 − 15` is two folded adds. With `L` lanes of 16-bit multiply-add at 250 MHz, one matrix costs `~n²/L` cycles: **~8 µs (128 lanes) to ~17 µs (64 lanes)** at L9, vs **590 µs measured per matrix on CPU at L9** (degraded from 94 µs at L8 by cache pressure — the 512 KB elimination buffer no longer fits L2; on-chip BRAM has no such cliff). With 4–8 engines, the 260,610 L9 pair-sign matrices take **~0.3–1.1 s** (first-order estimate). Phase 2 is sketched for sizing only; it becomes a full spec after Phase 1 synthesis data exists.

---

## 4. Resource estimates (Phase 1, L9 configuration, 16 PEs)

XCU250 capacity (4 SLRs): **1,728K LUT, 3,456K FF, 12,288 DSP48E2, 2,688 BRAM36, 1,280 URAM**, 64 GB DDR4 @ 77 GB/s.

| block | basis | LUT | FF | BRAM36 | DSP |
|---|---|---|---|---|---|
| `perm_l` mux network | 9 stages × 512 2:1-mux | ~4,600 | ~0 | 0 | 0 |
| XORs (`d`, `v`) | 2 × 512 | ~1,000 | ~1,024 (pipeline) | 0 | 0 |
| popcount-512 | carry-save tree | ~600 | ~600 | 0 | 0 |
| pair FSM + hist update | counters, banked update | ~300 | ~500 | 0 | 0 |
| sign-table replica | 512 × 512 b, dual-port | 0 | 0 | **8** | 0 |
| hist + fiber RAMs | 257×32 + 512×32 | 0 | 0 | ~2 | 0 |
| **per PE** | | **~6,500** | **~2,100** | **~10** | **0** |
| **16 PEs** | | **~104K (6.0%)** | **~34K (1.0%)** | **~160 (6.0%)** | **0** |
| shell + host ifc (est.) | XDMA/AXI, aggregation | ~30K (1.7%) | ~50K | ~20 | 0 |
| **design total** | | **~134K (7.8%)** | **~84K (2.4%)** | **~180 (6.7%)** | **0** |

Assumptions, stated explicitly: one LUT per 2:1 mux bit (Vitis typically does better via LUT6 sharing); popcount ≈ 1.2 LUT/bit; 250 MHz closure on a 512-bit datapath within one SLR (place all 16 PEs in one SLR if routing allows, else 8+8 across two SLRs with per-SLR table replicas — BRAM cost unchanged, crossing SLRs only for the 32 KB table load and 49 KB readback). URAM is untouched at L9 and reserved for L10+ replicas and Phase-2 matrix buffers. Fits trivially on one U250; the second card is redundant capacity or a Phase-2 verify farm.

Phase-2 first-order estimate (outline only): per GE engine, 64 lanes × 16-bit multiply-add ≈ 25–40K LUT (LUT-based mod-`2^16−15` reduction) **or** ~64 DSP48E2, plus one 512×512×16 b matrix buffer = 4.2 Mb ≈ **~114 BRAM36** (double-buffered ~228, 8.5% of capacity — no L2-style capacity cliff, cf. the CPU's super-quadratic per-matrix degradation). Four to eight engines fit comfortably alongside Phase 1.

---

## 5. Speedup estimates over CPU

CPU baselines are **measured at L9** on this dev container (single thread, L9 census spec §3): C census **0.109 s**; exact GF(65521) verification **153.9 s** over all 260,610 pair-signs (590 µs/matrix at `N = 512`, up from 94 µs at `N = 256` because the elimination buffer outgrew L2). (L8, for reference: census 0.010 s, verification 6.1 s.)

| task (L9) | CPU (single thread, measured) | U250 estimate | speedup |
|---|---|---|---|
| census, kernel-only | 0.109 s (C) | 8,145 cycles ≈ **33 µs** @ 16 PEs, 250 MHz, II=1 | **~3,300×** |
| census, end-to-end | 0.109 s (C) | ~0.5–2 ms (PCIe DMA setup + 81 KB traffic dominates) | **~55–220×** |
| verification (Phase 2, outline) | 153.9 s (GF(65521) rank, C) | ~4–8 engines × ~8–17 µs/matrix → **~0.3–1.1 s** for 260,610 pair-signs | **~140–510×** |

Honest framing: the L9 census is already ~0.1 s on CPU, so Phase 1's value is (a) validating the design and toolflow on the exact contract, (b) per-scan latency for interactive sweeps, and (c) headroom — L10 (~0.9 s CPU census at the measured 8×/level) and beyond stay in the microsecond-to-millisecond regime on the card. Phase 2 attacks the phase that actually dominates CPU wall clock (153.9 s at L9; L10 projected 40–100 min on this hardware per the L9 spec — the regime where an FPGA verify farm pays for itself). The garden seed's hypothesis (100–1000×) is consistent with the **kernel-only** Phase-1 estimate and overlaps the Phase-2 outline estimate; the end-to-end Phase-1 figure lands just below it, a reminder that at L9 the scan is latency-bound, not compute-bound.

---

## 6. Contract clauses (CI-gated)

`scripts/research/fpga_census_kernel_model.c`, enforced by `scripts/ci/fpga_catastrophe_scan_gate.sh`:

| clause | statement | result |
|---|---|---|
| M1 | bit-parity datapath == audited integer criterion, every candidate pair, b = 4..9 (173,214 pairs total) | PASS |
| M2 | census triples == growth law at confirmed levels b = 4..8 (84, 588, 3036, 13884, 59772) | PASS |
| M3 | **census triples == Z(9) = 249084** — independent re-verification of the L9 census spec's third out-of-sample confirmation; new content is the hardware datapath's pair-by-pair agreement with the audited criterion at L9 | PASS |
| M4 | L8 nullity histogram == published histogram (31 values, 29886 pairs) | PASS |
| M5 | cycle model: 1 pair/cycle/PE → L9 = 130,305 cycles (1 PE), 8,145 cycles (16 PEs) ≈ 32.6 µs @ 250 MHz | reported |

M1 is the portability contract: the HLS kernel is correct iff it reproduces the bit-parity path, and the bit-parity path is here **exhaustively verified** equal to the audited integer criterion at six consecutive levels — by execution over every candidate pair, not by sampling and not by a symbolic derivation inside this document.

---

## 7. What this is NOT

- **Not measured hardware data.** No FPGA is installed; nothing has been synthesized, placed, routed, or benchmarked. Every §4/§5 number is an estimate with stated assumptions; the U250 resource counts themselves are data-sheet values.
- **Not a claim that the HLS outline compiles.** `krnl_census.cpp` / `host.cpp` are reference outlines written before toolchain access; the *semantics* they must implement are what is CI-gated (the C model), not the C++ text.
- **Not a new census method, and not the Z(9) confirmation.** The kernel implements the already-audited exact 2-cycle criterion; the novelty is the hardware mapping (sign-as-bit + rewiring permutation + popcount) and its bit-accurate gated model. Z(9) = 249084 was confirmed by the L9 census spec (parallel lane, full GF(65521) audit); the M3 clause re-verifies it in 0.43 s of CPU as a side effect of validating the hardware datapath at L9 — no FPGA involved.
- **Not the full ZD locus.** Canonical 2-unit sums `e_i ± e_j` only, as in the L4–L8 contracts.
- **Not a clinical claim.**

---

## 8. Reproduce

```bash
# contract (compile + run + spec-consistency checks; ~1 s)
bash scripts/ci/fpga_catastrophe_scan_gate.sh
# expect: FPGA_CENSUS_MODEL_VERDICT PASS, FPGA_CATASTROPHE_SCAN_GATE_OK

# model alone
cc -O2 -o /tmp/fpga_model scripts/research/fpga_census_kernel_model.c && /tmp/fpga_model
```

---

## 9. AI disclosure

Spec, kernel outline, and contract model drafted under human direction (2026-07-26). No clinical content. GAIDeT-ICMJE 2025.
