#!/usr/bin/env python3
"""
Phase 7B Step 1c — Deterministic Box Propagation Growth (1c-A)
                 & Split-Conformal Probabilistic Certificate (1c-B)

Oracle-side only (no Sounio engine edits yet):
- STEP 1c-A: Propagates exact float64 box intervals [lo, hi] through Qwen3-0.6B
  layers 1..28 on the 512-token audit set with Q8 weight intervals, recording
  per-token median_i(propagated_radius_i / true_|dh_i|) and its p95 across
  layers 1..L (stopping if median rho_l > 1e3, or reporting through L=28).
  STOP RULE: if rho_28 median > 10, deterministic box route is closed.
- STEP 1c-B: Fits and calibrates split-conformal score
    s_{t, j} = ||w_j||_2 * ||dh_hat(t)||_2 / sqrt(d)
  on >= 2,000 calibration tokens and evaluates strictly on >= 2,000 HELD-OUT
  tokens from disjoint prompts for alpha in {0.01, 0.001} using two estimators:
    (i)  per-layer residual-stream L2 norm linear combination fitted on cal only;
    (ii) second Q8 forward pass with a half-step shifted rounding grid (2x pass).

Saves report to:
/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase7b_step1c_report.json
"""

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_phase7b_eisa_teacher_forcing import (  # noqa: E402
    CHECKPOINT_DIR,
    DOMAINS,
    quantize_w_eisa_q8,
)

REPORT_PATH = Path(
    "/Users/demetriosagourakis/sounio-research/checkpoints/"
    "Qwen3-0.6B-ossm-sidecar/phase7b_step1c_report.json"
)
STOP_RULE_1CA_THRESHOLD = 10.0

# 6 Calibration prompts (>= 2,000 tokens total), evaluated independently per prompt.
CAL_PROMPTS = {
    **DOMAINS,
    "thermodynamics_cal": (
        "Classical equilibrium thermodynamics characterizes macroscopic physical systems using a small number "
        "of state variables including internal energy U, entropy S, volume V, temperature T, pressure P, and "
        "chemical potential mu. The first law expresses conservation of total energy as dU = TdS - PdV + sum_k mu_k dN_k "
        "for reversible quasi-static transformations along an equilibrium manifold. By the second law, the entropy "
        "of a thermally isolated system never decreases during any spontaneous process, establishing an arrow of "
        "irreversibility and bounding the maximum thermodynamic efficiency eta_max = 1 - T_c / T_h achievable by any "
        "Carnot heat engine operating between reservoirs at absolute temperatures T_h and T_c. Legendre transforms of "
        "internal energy yield the Helmholtz free energy F = U - TS, the enthalpy H = U + PV, and the Gibbs free "
        "energy G = H - TS, whose natural differentials govern equilibrium criteria under isothermal-isochoric and "
        "isothermal-isobaric boundary constraints. Maxwell relations follow immediately from the equality of mixed "
        "second partial derivatives of smooth thermodynamic potentials, connecting heat capacities, compressibilities, "
        "and thermal expansion coefficients without microscopic statistical assumptions. Near a continuous phase "
        "transition, susceptibility diverges and mean-field Landau expansions break down due to long-range fluctuations."
    ),
}

# 6 Completely Disjoint HELD-OUT prompts (>= 2,000 tokens total), unseen subjects.
HELD_OUT_PROMPTS = {
    "os_virtual_memory": (
        "Modern symmetric multiprocessing operating system kernels manage physical memory through hierarchical "
        "four-level or five-level hardware page tables translating virtual addresses into physical page frames. "
        "On x86-64 and ARM64 architectures, each memory page typically spans four kilobytes, with hugepage mappings "
        "spanning two megabytes or one gigabyte to reduce Translation Lookaside Buffer pressure and page walk latency. "
        "When a kernel modifies page permissions, promotes copy-on-write mappings after a fork system call, or "
        "unmaps regions during process teardown, hardware page-table walkers on remote processor cores may retain "
        "stale translation entries inside private TLB caches. Consequently, the kernel must issue inter-processor "
        "interrupts to synchronize a TLB shootdown across every active core referencing that address space identifier "
        "before returning freed frames to the buddy page allocator. For sub-page kernel object allocation, slab and "
        "slub allocators organize physically contiguous page blocks into per-CPU lockless freelist caches partitioned "
        "by object size class, minimizing false sharing across cache lines and avoiding global spinlock contention. "
        "During memory reclamation under pressure, kswapd scans inactive anonymous and file-backed LRU lists using "
        "hardware accessed bits, prioritizing clean page cache eviction before writing dirty pages back to NVMe "
        "storage queues or compressing anonymous pages inside zram pools. Asynchronous page faults overlap demand "
        "paging latency with runnable thread scheduling whenever a user thread touches a swapped-out or lazily "
        "allocated anonymous region."
    ),
    "analytic_number_theory": (
        "Analytic number theory studies the distribution of prime integers by embedding discrete arithmetic functions "
        "into meromorphic Dirichlet series defined on the complex half-plane of absolute convergence. For real parts "
        "strictly exceeding unity, the Riemann zeta function admits both the additive series sum_{n>=1} n^{-s} and "
        "Euler's multiplicative product over all rational primes prod_p (1 - p^{-s})^{-1}, directly encoding unique "
        "prime factorization in complex analysis. Extending zeta analytically to the entire complex plane except for "
        "a simple pole of residue one at s = 1 reveals a symmetric functional equation relating zeta(s) and zeta(1 - s) "
        "through the Gamma factor and sine symmetry. Hadamard and de la Vallee Poussin established the prime number "
        "theorem, proving that pi(x) ~ x / log(x), by demonstrating that zeta(1 + it) never vanishes along the "
        "boundary line Re(s) = 1. Contour integration of the logarithmic derivative -zeta'(s)/zeta(s) weighted by "
        "Perron kernels transforms zero-free regions inside the critical strip into quantitative error bounds for "
        "Chebyshev's weighted prime counting sum psi(x) = sum_{p^k <= x} log(p). Generalizing to primitive Dirichlet "
        "characters chi modulo q produces L-functions L(s, chi) whose non-vanishing at s = 1 proves Dirichlet's "
        "theorem on infinitely many primes in every coprime arithmetic progression, while Siegel-Walfisz and Bombieri-"
        "Vinogradov estimates quantify prime equidistribution across moduli on average."
    ),
    "renal_pharmacokinetics": (
        "Clinical therapeutic drug monitoring of vancomycin in critically ill adults relies on two-compartment "
        "population pharmacokinetic models parameterized by total systemic clearance CL, central volume of distribution "
        "V1, intercompartmental distributional clearance Q, and peripheral tissue volume V2. Because vancomycin exhibits "
        "minimal hepatic metabolism and undergoes predominantly glomerular filtration with limited tubular handling, "
        "systemic clearance correlates closely with measured urinary creatinine clearance or Cockcroft-Gault estimates "
        "adjusted for body weight, age, and baseline serum creatinine concentration. Rapid fluctuations in renal perfusion, "
        "augmented renal clearance following trauma or sepsis, or acute kidney injury shift elimination half-life "
        "substantially across sequential dosing intervals. Under consensus guidelines targeting an area under the "
        "twenty-four hour concentration-time curve to minimum inhibitory concentration ratio (AUC24 / MIC) between "
        "400 and 600 mg*h/L against methicillin-resistant Staphylococcus aureus, Bayesian maximum a posteriori estimation "
        "combines informative prior parameter covariances with one or two sparse serum concentrations drawn near peak "
        "distribution equilibrium and trough pre-dose timing. Unlike trough-only targeting, full Bayesian integration "
        "captures non-linear accumulation across loading doses, continuous renal replacement therapy effluent rates, "
        "and inter-occasion clearance variability, reducing nephrotoxicity exposure thresholds while maintaining "
        "microbiological bactericidal attainment in deep-seated bacteremia and infective endocarditis."
    ),
    "systems_graph_algorithms": (
        "def tarjan_strongly_connected_components(adj: list[list[int]]) -> list[list[int]]:\n"
        "    n = len(adj)\n"
        "    index_counter = 0\n"
        "    indices = [-1] * n\n"
        "    lowlink = [0] * n\n"
        "    on_stack = [False] * n\n"
        "    comp_stack: list[int] = []\n"
        "    components: list[list[int]] = []\n"
        "    call_stack: list[tuple[int, int]] = []\n\n"
        "    for start in range(n):\n"
        "        if indices[start] != -1:\n"
        "            continue\n"
        "        indices[start] = lowlink[start] = index_counter\n"
        "        index_counter += 1\n"
        "        comp_stack.append(start)\n"
        "        on_stack[start] = True\n"
        "        call_stack.append((start, 0))\n\n"
        "        while call_stack:\n"
        "            v, edge_i = call_stack[-1]\n"
        "            if edge_i < len(adj[v]):\n"
        "                w = adj[v][edge_i]\n"
        "                call_stack[-1] = (v, edge_i + 1)\n"
        "                if indices[w] == -1:\n"
        "                    indices[w] = lowlink[w] = index_counter\n"
        "                    index_counter += 1\n"
        "                    comp_stack.append(w)\n"
        "                    on_stack[w] = True\n"
        "                    call_stack.append((w, 0))\n"
        "                elif on_stack[w]:\n"
        "                    if indices[w] < lowlink[v]:\n"
        "                        lowlink[v] = indices[w]\n"
        "            else:\n"
        "                call_stack.pop()\n"
        "                if call_stack:\n"
        "                    parent = call_stack[-1][0]\n"
        "                    if lowlink[v] < lowlink[parent]:\n"
        "                        lowlink[parent] = lowlink[v]\n"
        "                if lowlink[v] == indices[v]:\n"
        "                    scc: list[int] = []\n"
        "                    while True:\n"
        "                        node = comp_stack.pop()\n"
        "                        on_stack[node] = False\n"
        "                        scc.append(node)\n"
        "                        if node == v:\n"
        "                            break\n"
        "                    components.append(scc)\n"
        "    return components\n"
    ),
    "distributed_consensus": (
        "Crash-fault-tolerant state machine replication protocols such as Raft decompose consensus over an ordered "
        "command log into leader election, continuous log replication, and deterministic safety invariants across "
        "arbitrary network partitions and packet reordering. Time is partitioned into monotonically increasing "
        "logical terms, each containing at most one elected leader guaranteed to possess every previously committed "
        "entry through the log completeness voting rule. Whenever a follower detects no AppendEntries heartbeat "
        "within its randomized election timeout window, it increments currentTerm, transitions to candidate state, "
        "persists its vote locally to durable write-ahead storage, and broadcasts RequestVote RPCs containing its "
        "lastLogIndex and lastLogTerm. A responder grants its vote at most once per term and rejects candidates "
        "whose log is strictly less up-to-date lexicographically on term and index. Once elected by a strict quorum "
        "majority, the leader dispatches AppendEntries consistency checks carrying prevLogIndex and prevLogTerm; "
        "followers truncate conflicting uncommitted suffixes and persist matching entries before acknowledging. An "
        "entry created in the current term commits only after replication across a quorum majority, indirectly "
        "committing all earlier entries by the log matching property and preventing figure-8 split overwrites."
    ),
    "cellular_bioenergetics": (
        "Eukaryotic mitochondrial oxidative phosphorylation couples exergonic electron transfer through respiratory "
        "complexes I through IV to transmembrane proton translocation across the inner mitochondrial membrane, "
        "generating the electrochemical proton-motive force predicted by Mitchell's chemiosmotic theory. Both the "
        "electrical membrane potential delta_psi of roughly minus 150 to 180 millivolts inside the matrix and the "
        "transmembrane pH gradient drive protons back into the matrix through the F0F1-ATP synthase nanomotor. Inside "
        "the membrane-embedded F0 sector, protonation and deprotonation of conserved glutamate residues in the rotating "
        "c-ring stator interface convert discrete electrochemical steps into mechanical torque transmitted along the "
        "asymmetric central gamma-epsilon stalk. According to Boyer's binding-change mechanism, 120-degree discrete "
        "rotations of the gamma shaft sequentially force the three catalytic alpha-beta dimers of the soluble F1 "
        "hexamer through Open, Loose, and Tight conformational states, condensing ADP and inorganic phosphate into "
        "ATP without direct chemical phosphorylation by respiratory carriers. Regulated proton leak through uncoupling "
        "proteins UCP1 in brown adipose tissue dissipates this electrochemical gradient directly as non-shivering "
        "thermogenic heat while accelerating NADH oxidation and tricarboxylic acid cycle flux."
    ),
    "postgres_mvcc_btree": (
        "Relational database engines implementing multiversion concurrency control maintain tuple-level visibility "
        "snapshots without blocking concurrent readers behind write locks. In PostgreSQL heap storage, every inserted "
        "or updated row version carries creation and expiration transaction identifiers xmin and xmax alongside hint "
        "bits cached lazily after clog commit-status lookups. A running transaction snapshot records the oldest active "
        "transaction identifier and the set of in-flight writers, allowing sequential and index scans to decide tuple "
        "visibility deterministically across serializable and repeatable-read isolation levels. Because updates append "
        "new physical tuple versions instead of overwriting slots in place, dead tuples accumulate on heap pages until "
        "periodic vacuum processes reclaim line pointers and freeze historical transaction identifiers before 32-bit "
        "epoch wraparound occurs. Concurrent Lehman-Yao B-link tree indexes support lock-coupled high-key right "
        "pointers so searchers traversing splitting interior or leaf nodes can follow sibling links rightward without "
        "holding parent latches across disk reads. Write-ahead logging writes full-page images on the first dirty "
        "modification after each checkpoint so crash recovery remains idempotent against torn eight-kilobyte OS pages."
    ),
    "orbital_mechanics_kalman": (
        "Autonomous spacecraft orbit determination integrates non-Keplerian perturbed equations of motion including "
        "Earth's zonal harmonic oblateness J2, third-body lunar and solar gravitational tides, atmospheric drag proportional "
        "to thermospheric density along the spacecraft velocity vector, and solar radiation pressure scaled by surface "
        "reflectivity and cross-sectional area-to-mass ratio. Because optical star-tracker bearings, range-rate Doppler "
        "transponders, and accelerometer telemetry arrive asynchronously with distinct measurement noise covariances, an "
        "extended or unscented Kalman filter maintains a six-dimensional Cartesian position-velocity state estimate "
        "plus empirical drag and clock bias nuisance parameters in an upper-triangular U-D factored covariance representation "
        "to preserve positive semi-definiteness under finite floating-point wordlength. Between ground-station tracking "
        "passes, sixth-order Runge-Kutta-Nystrom or Gauss-Jackson multistep integrators propagate both the reference "
        "trajectory and the thirty-six-element state transition matrix along variational partial derivatives. Whenever a "
        "radio-metric residual exceeds chi-square gating thresholds during momentum-wheel desaturation burns, adaptive "
        "process-noise inflation prevents filter divergence before orbital plane-change maneuver planning."
    ),
    "quantum_error_correction": (
        "Fault-tolerant quantum computation encodes logical qubits into entangled subspaces stabilized by commuting "
        "multi-qubit Pauli check operators drawn from surface codes or quantum low-density parity-check families. "
        "In a planar rotated surface code of distance d arranged on a two-dimensional lattice of data and ancilla qubits, "
        "weight-four X-type plaquette and Z-type star stabilizers detect single-qubit bit-flip and phase-flip errors "
        "as pairs of syndrome defect excitations whenever local Pauli noise violates parity checks across repeated "
        "measurement rounds. Because syndrome readout gates and ancilla measurements themselves suffer depolarizing faults "
        "and measurement-classification erasures, a spacetime decoding hypergraph connects spatial defects across "
        "adjacent measurement cycles with weighted diagonal and timelike fault edges. Minimum-weight perfect matching "
        "and union-find decoders pair syndrome defects in near-linear amortized time to infer a recovery Pauli frame "
        "homologous to the true error chain. As long as physical error probabilities per gate cycle lie strictly below "
        "the fault-tolerant threshold near one percent, logical error rates decay exponentially in the code distance."
    ),
    "rust_async_executor": (
        "fn poll_bounded_ring_channel<T>(ring: &mut [Option<T>], head: &mut usize, tail: &mut usize, cap: usize) -> Option<T> {\n"
        "    if *head == *tail {\n"
        "        return None;\n"
        "    }\n"
        "    let idx = *head % cap;\n"
        "    let item = ring[idx].take();\n"
        "    *head = head.wrapping_add(1);\n"
        "    item\n"
        "}\n\n"
        "In work-stealing asynchronous runtimes, state-machine futures compiled from async blocks implement a non-blocking "
        "poll state transition that either completes with an output value or registers the caller's intrusive waker "
        "handle inside an I/O reactor readiness slot before returning Pending. Each worker thread drains a local fixed-size "
        "FIFO run queue of runnable task pointers without atomic read-modify-write contention, falling back to a global "
        "inject queue every sixty-one polls to guarantee fairness against tight yield loops. When a worker's local deque "
        "empties, it probes sibling workers' Chase-Lev lock-free work-stealing deques from the tail index using "
        "compare-and-swap synchronization while the owner pops LIFO from the opposite end for cache locality. Timer wheels "
        "organized into hierarchical millisecond and second bitmask slots coalesce millions of pending deadline expirations "
        "with O(1) insertion and cancellation overhead."
    ),
}


# =============================================================================
# Part A Helpers — Exact f64 Forward + Tight Sound f64 Box Operators
# =============================================================================

_SILU_X_STAR = -1.2784645427610738
_SILU_Y_MIN = _SILU_X_STAR / (1.0 + math.exp(-_SILU_X_STAR))


def quantize_q8_block32_with_box(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bit-exact Q8_EISA quantization mirroring audit_phase7b_eisa_teacher_forcing.py:
      s = f16(max_abs / 127.0)
      q = clamp(round(w / s), -127, 127)
      w_q8 = (s * q).to(bf16).float() -> f64 (matches runtime Q8 BF16 execution or f64 Q8)
    Here we compute w_q8 in exact f64 (s * q) AND check BF16 weight vs s*q so BOTH
    w_true (model weights in BF16->f64) and w_q8_bf16 (or s*q in f64) lie strictly inside
    [w_q8 - R_W, w_q8 + R_W].
    Wait: to match the Phase 7B runtime where Q8 weights in run_full_28l_q8_audit are stored as
    (s * q).to(torch.bfloat16), we construct:
      w_true_f64 = w_bf16.double()
      w_q8_f64   = ((s * q).view(w.shape).to(torch.bfloat16)).double()
      r_block    = max(0.5005 * s_f64, max_over_block |w_true_f64 - w_q8_f64|)
    This guarantees BOTH w_true_f64 and w_q8_f64 satisfy |w - w_q8_f64| <= r_block!
    """
    w = w_bf16.float().cpu()
    rows, cols = w.shape
    w_b = w.view(rows, cols // 32, 32)
    m = torch.amax(torch.abs(w_b), dim=-1, keepdim=True)
    s = (m / 127.0).to(torch.float16).float()
    s = torch.where(s == 0.0, torch.ones_like(s), s)
    q = torch.clamp(torch.round(w_b / s), -127.0, 127.0)
    w_q8_bf16 = (s * q).view(rows, cols).to(torch.bfloat16)

    w_true_f64 = w_bf16.detach().cpu().double()
    w_q8_f64 = w_q8_bf16.double()
    abs_err_b = torch.abs(w_true_f64 - w_q8_f64).view(rows, cols // 32, 32)
    block_max_err = torch.amax(abs_err_b, dim=-1, keepdim=True)
    half_step = 0.5005 * s.double()
    r_block = torch.maximum(half_step, block_max_err).expand(rows, cols // 32, 32).reshape(rows, cols)
    return w_q8_f64, r_block


def f64_rms_norm_point(
    x: torch.Tensor, w: torch.Tensor, eps: float, hf_f32_emulate: bool = False
) -> torch.Tensor:
    if hf_f32_emulate:
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps)).double() * w
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def f64_rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def f64_forward_layers(
    h0: torch.Tensor,
    weights_by_layer: list[dict],
    cos: torch.Tensor,
    sin: torch.Tensor,
    causal_mask_zero_neg_inf: torch.Tensor,
    n_q: int,
    n_kv: int,
    head_dim: int,
    eps: float,
    use_q8: bool = False,
    hf_f32_emulate: bool = False,
) -> list[torch.Tensor]:
    """Runs exact f64 forward across layers 1..L, returning list of post-layer hidden states."""
    x = h0
    T = x.shape[0]
    g_rep = n_q // n_kv
    states = []
    scale = 1.0 / math.sqrt(float(head_dim))

    for lw in weights_by_layer:
        wq = lw["q_q8" if use_q8 else "q"]
        wk = lw["k_q8" if use_q8 else "k"]
        wv = lw["v_q8" if use_q8 else "v"]
        wo = lw["o_q8" if use_q8 else "o"]
        wg = lw["gate_q8" if use_q8 else "gate"]
        wu = lw["up_q8" if use_q8 else "up"]
        wd = lw["down_q8" if use_q8 else "down"]

        # Attention block
        res = x
        h = f64_rms_norm_point(x, lw["in_norm"], eps, hf_f32_emulate=hf_f32_emulate)
        q = (h @ wq.t()).view(T, n_q, head_dim)
        k = (h @ wk.t()).view(T, n_kv, head_dim)
        v = (h @ wv.t()).view(T, n_kv, head_dim)

        q = f64_rms_norm_point(q, lw["q_norm"], eps, hf_f32_emulate=hf_f32_emulate)
        k = f64_rms_norm_point(k, lw["k_norm"], eps, hf_f32_emulate=hf_f32_emulate)

        q = q * cos + f64_rotate_half(q) * sin
        k = k * cos + f64_rotate_half(k) * sin

        q_h = q.transpose(0, 1)  # (n_q, T, D)
        k_h = k.transpose(0, 1).repeat_interleave(g_rep, dim=0)  # (n_q, T, D)
        v_h = v.transpose(0, 1).repeat_interleave(g_rep, dim=0)  # (n_q, T, D)

        scores = (q_h @ k_h.transpose(-1, -2)) * scale + causal_mask_zero_neg_inf
        if hf_f32_emulate:
            probs = torch.softmax(scores.float(), dim=-1).double()
        else:
            probs = torch.softmax(scores, dim=-1)
        attn = (probs @ v_h).transpose(0, 1).contiguous().view(T, n_q * head_dim)
        x = res + (attn @ wo.t())

        # MLP block
        res = x
        h = f64_rms_norm_point(x, lw["post_norm"], eps, hf_f32_emulate=hf_f32_emulate)
        gate = F.silu(h @ wg.t()) * (h @ wu.t())
        x = res + (gate @ wd.t())
        states.append(x)

    return states


def box_linear(
    lo: torch.Tensor, hi: torch.Tensor, w_q8: torch.Tensor, r_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rump midpoint-radius box GEMM Y = X W^T with sound f64 rounding guard."""
    mx = 0.5 * (lo + hi)
    rx = 0.5 * (hi - lo)
    my = mx @ w_q8.t()
    ry = rx @ torch.abs(w_q8).t() + (torch.abs(mx) + rx) @ r_w.t()
    ry = ry + 1e-13 * (torch.abs(my) + ry) + 1e-15
    return my - ry, my + ry


def box_rms_norm(
    a: torch.Tensor, b: torch.Tensor, gamma: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tight monotone f64 box RMSNorm over last dimension."""
    d = float(a.shape[-1])
    sqrt_d = math.sqrt(d)

    # Exact square interval [u_lo, u_hi] of x^2 on [a, b]
    a2 = a * a
    b2 = b * b
    u_hi = torch.maximum(a2, b2)
    straddles_zero = (a <= 0.0) & (b >= 0.0)
    u_lo = torch.where(straddles_zero, torch.zeros_like(a), torch.minimum(a2, b2))

    sum_lo = u_lo.sum(dim=-1, keepdim=True)
    sum_hi = u_hi.sum(dim=-1, keepdim=True)
    d_eps = d * eps
    sig_lo = torch.clamp_min(sum_lo - u_lo + d_eps - 1e-12 * sum_lo, d_eps)
    sig_hi = sum_hi - u_hi + d_eps + 1e-12 * sum_hi

    s_for_lo = torch.where(a >= 0.0, sig_hi, sig_lo)
    s_for_hi = torch.where(b >= 0.0, sig_lo, sig_hi)
    n_lo = sqrt_d * a * torch.rsqrt(a2 + s_for_lo)
    n_hi = sqrt_d * b * torch.rsqrt(b2 + s_for_hi)

    # Multiply by weight gamma (preserving sign) + tiny f64 rounding guard
    lo_scaled = torch.where(gamma >= 0.0, n_lo * gamma, n_hi * gamma)
    hi_scaled = torch.where(gamma >= 0.0, n_hi * gamma, n_lo * gamma)
    guard = 1e-13 * torch.maximum(torch.abs(lo_scaled), torch.abs(hi_scaled)) + 1e-15
    return lo_scaled - guard, hi_scaled + guard


def box_rope(
    lo: torch.Tensor, hi: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m = 0.5 * (lo + hi)
    r = 0.5 * (hi - lo)
    mr = f64_rotate_half(m)
    rr = f64_rotate_half(r)
    my = m * cos + mr * sin
    ry = r * torch.abs(cos) + rr * torch.abs(sin) + 1e-13 * (torch.abs(my) + r + rr) + 1e-15
    return my - ry, my + ry


def box_softmax_causal(
    s_lo: torch.Tensor, s_hi: torch.Tensor, future_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact coordinatewise monotone causal softmax bounds:
      p_lo_s = e_lo_s / (e_lo_s + sum_{s' != s} e_hi_s')
      p_hi_s = e_hi_s / (e_hi_s + sum_{s' != s} e_lo_s')
    masked cleanly with M = max_{s' <= t} s_hi[t, s'].
    """
    s_hi_masked = s_hi.masked_fill(future_mask, float("-inf"))
    m_shift = torch.amax(s_hi_masked, dim=-1, keepdim=True)  # (H, T, 1)

    e_hi = torch.exp(torch.clamp_min(s_hi - m_shift, -745.0))
    e_lo = torch.exp(torch.clamp_min(s_lo - m_shift, -745.0))
    e_hi = e_hi.masked_fill(future_mask, 0.0)
    e_lo = e_lo.masked_fill(future_mask, 0.0)

    sum_hi = e_hi.sum(dim=-1, keepdim=True)
    sum_lo = e_lo.sum(dim=-1, keepdim=True)

    other_hi = torch.clamp_min(sum_hi - e_hi, 0.0) * (1.0 + 1e-13)
    other_lo = torch.clamp_min(sum_lo - e_lo, 0.0) * (1.0 - 1e-13)

    p_lo = torch.where(
        future_mask,
        torch.zeros_like(e_lo),
        torch.clamp_min((e_lo / torch.clamp_min(e_lo + other_hi, 1e-300)) * (1.0 - 1e-13), 0.0),
    )
    p_hi = torch.where(
        future_mask,
        torch.zeros_like(e_hi),
        torch.clamp_max((e_hi / torch.clamp_min(e_hi + other_lo, 1e-300)) * (1.0 + 1e-13), 1.0),
    )
    return p_lo, p_hi


def box_silu_mul(
    glo: torch.Tensor, ghi: torch.Tensor, ulo: torch.Tensor, uhi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    fg_lo = F.silu(glo)
    fg_hi = F.silu(ghi)
    s_hi = torch.maximum(fg_lo, fg_hi) + 1e-14
    s_lo = torch.minimum(fg_lo, fg_hi)
    contains_min = (glo <= _SILU_X_STAR) & (_SILU_X_STAR <= ghi)
    s_lo = torch.where(contains_min, torch.minimum(s_lo, torch.full_like(s_lo, _SILU_Y_MIN)), s_lo) - 1e-14

    p1 = s_lo * ulo
    p2 = s_lo * uhi
    p3 = s_hi * ulo
    p4 = s_hi * uhi
    out_lo = torch.minimum(torch.minimum(p1, p2), torch.minimum(p3, p4))
    out_hi = torch.maximum(torch.maximum(p1, p2), torch.maximum(p3, p4))
    guard = 1e-13 * torch.maximum(torch.abs(out_lo), torch.abs(out_hi)) + 1e-15
    return out_lo - guard, out_hi + guard


def run_step1c_a_box_growth(
    model: nn.Module,
    eval_ids_cpu: torch.Tensor,
) -> dict:
    print("\n" + "=" * 80)
    print("PHASE 7B STEP 1c-A — F64 BOX PROPAGATION GROWTH (512 TOKENS, LAYERS 1..28)")
    print("=" * 80)
    t_start = time.time()

    cfg = model.config
    n_q = int(cfg.num_attention_heads)
    n_kv = int(cfg.num_key_value_heads)
    head_dim = int(cfg.head_dim)
    eps = float(cfg.rms_norm_eps)
    g_rep = n_q // n_kv
    T = int(eval_ids_cpu.shape[1])
    scale = 1.0 / math.sqrt(float(head_dim))

    # Extract embeddings & rotary frequencies in exact f64
    with torch.no_grad():
        emb_f64 = model.model.embed_tokens.weight.detach().cpu().double()[eval_ids_cpu[0]]  # (T, 1024)
        pos_ids = torch.arange(T, dtype=torch.long).unsqueeze(0)
        final_norm_w = model.model.norm.weight.detach().cpu().double()

        # Reference check against HF model in f64 on first 64 tokens before running boxes
        check_len = 64
        model_f64_check = AutoModelForCausalLM.from_pretrained(
            str(CHECKPOINT_DIR), dtype=torch.float64, trust_remote_code=True
        ).cpu()
        model_f64_check.eval()
        rotary_inv = model_f64_check.model.rotary_emb.inv_freq.detach().cpu().double()
        freqs = torch.outer(pos_ids[0].double(), rotary_inv)
        emb_freq = torch.cat([freqs, freqs], dim=-1)
        cos = emb_freq.cos().unsqueeze(1)  # (T, 1, 128)
        sin = emb_freq.sin().unsqueeze(1)  # (T, 1, 128)
        hf_out = model_f64_check(eval_ids_cpu[:, :check_len], output_hidden_states=True)
        cos_hf_raw, sin_hf_raw = model_f64_check.model.rotary_emb(
            emb_f64[:check_len].unsqueeze(0), pos_ids[:, :check_len]
        )
        cos_hf = cos_hf_raw[0].unsqueeze(1).double()
        sin_hf = sin_hf_raw[0].unsqueeze(1).double()
        hf_h1 = hf_out.hidden_states[1][0]
        hf_h27 = hf_out.hidden_states[-2][0]
        hf_h28_normed = hf_out.hidden_states[-1][0]
        del model_f64_check

    # Build weights + block-32 Q8 boxes for all 28 layers
    weights_by_layer = []
    for layer in model.model.layers:
        lw = {
            "in_norm": layer.input_layernorm.weight.detach().cpu().double(),
            "post_norm": layer.post_attention_layernorm.weight.detach().cpu().double(),
            "q_norm": layer.self_attn.q_norm.weight.detach().cpu().double(),
            "k_norm": layer.self_attn.k_norm.weight.detach().cpu().double(),
        }
        for pname, mod in [
            ("q", layer.self_attn.q_proj),
            ("k", layer.self_attn.k_proj),
            ("v", layer.self_attn.v_proj),
            ("o", layer.self_attn.o_proj),
            ("gate", layer.mlp.gate_proj),
            ("up", layer.mlp.up_proj),
            ("down", layer.mlp.down_proj),
        ]:
            w_true = mod.weight.detach().cpu().double()
            w_q8, r_w = quantize_q8_block32_with_box(mod.weight.data)
            lw[pname] = w_true
            lw[f"{pname}_q8"] = w_q8
            lw[f"{pname}_rw"] = r_w
            lw[f"{pname}_q8_abs"] = torch.abs(w_q8)
        weights_by_layer.append(lw)

    future_mask_check = torch.triu(torch.ones((check_len, check_len), dtype=torch.bool), diagonal=1)
    causal_bias_check = torch.zeros((check_len, check_len), dtype=torch.float64).masked_fill(
        future_mask_check, float("-inf")
    )
    chk_states_emul = f64_forward_layers(
        emb_f64[:check_len],
        weights_by_layer,
        cos_hf,
        sin_hf,
        causal_bias_check,
        n_q,
        n_kv,
        head_dim,
        eps,
        use_q8=False,
        hf_f32_emulate=True,
    )
    err_l1_emul = float(torch.max(torch.abs(chk_states_emul[0] - hf_h1)).item())
    err_l28_emul = float(
        torch.max(
            torch.abs(
                f64_rms_norm_point(chk_states_emul[-1], final_norm_w, eps, hf_f32_emulate=True)
                - hf_h28_normed
            )
        ).item()
    )

    chk_states = f64_forward_layers(
        emb_f64[:check_len],
        weights_by_layer,
        cos[:check_len],
        sin[:check_len],
        causal_bias_check,
        n_q,
        n_kv,
        head_dim,
        eps,
        use_q8=False,
    )
    err_l1 = float(torch.max(torch.abs(chk_states[0] - hf_h1)).item())
    err_l27 = float(torch.max(torch.abs(chk_states[-2] - hf_h27)).item())
    err_l28 = float(
        torch.max(torch.abs(f64_rms_norm_point(chk_states[-1], final_norm_w, eps) - hf_h28_normed)).item()
    )
    print(
        f"HF parity check (64 tokens): hf_f32_emul L1={err_l1_emul:.3e} L28_normed={err_l28_emul:.3e} | "
        f"pure_f64 L1={err_l1:.3e} L27={err_l27:.3e} L28_normed={err_l28:.3e}"
    )
    assert err_l1 < 1e-5 and err_l28 < 1e-3, (
        f"Pure f64 forward diverges from HF beyond f32 internal cast tolerance: L1={err_l1}, L28={err_l28}"
    )

    # Full 512-token point trajectories in f64 (true vs Q8)
    future_mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
    causal_bias = torch.zeros((T, T), dtype=torch.float64).masked_fill(future_mask, float("-inf"))
    true_states = f64_forward_layers(
        emb_f64, weights_by_layer, cos, sin, causal_bias, n_q, n_kv, head_dim, eps, use_q8=False
    )
    q8_states = f64_forward_layers(
        emb_f64, weights_by_layer, cos, sin, causal_bias, n_q, n_kv, head_dim, eps, use_q8=True
    )

    # Layer-by-layer box propagation starting from zero-radius embeddings [emb, emb]
    x_lo = emb_f64.clone()
    x_hi = emb_f64.clone()
    layer_records = []
    first_exceed_1e3_layer = None

    for l_idx, lw in enumerate(weights_by_layer, start=1):
        # 1. Residual save + input RMSNorm
        res_lo, res_hi = x_lo, x_hi
        h_lo, h_hi = box_rms_norm(x_lo, x_hi, lw["in_norm"], eps)

        # 2. Q, K, V projections
        q_lo, q_hi = box_linear(h_lo, h_hi, lw["q_q8"], lw["q_rw"])
        k_lo, k_hi = box_linear(h_lo, h_hi, lw["k_q8"], lw["k_rw"])
        v_lo, v_hi = box_linear(h_lo, h_hi, lw["v_q8"], lw["v_rw"])

        q_lo, q_hi = q_lo.view(T, n_q, head_dim), q_hi.view(T, n_q, head_dim)
        k_lo, k_hi = k_lo.view(T, n_kv, head_dim), k_hi.view(T, n_kv, head_dim)
        v_lo, v_hi = v_lo.view(T, n_kv, head_dim), v_hi.view(T, n_kv, head_dim)

        # 3. q_norm, k_norm + RoPE
        q_lo, q_hi = box_rms_norm(q_lo, q_hi, lw["q_norm"], eps)
        k_lo, k_hi = box_rms_norm(k_lo, k_hi, lw["k_norm"], eps)
        q_lo, q_hi = box_rope(q_lo, q_hi, cos, sin)
        k_lo, k_hi = box_rope(k_lo, k_hi, cos, sin)

        # 4. Attention scores & causal softmax
        qh_lo = q_lo.transpose(0, 1)
        qh_hi = q_hi.transpose(0, 1)
        kh_lo = k_lo.transpose(0, 1).repeat_interleave(g_rep, dim=0)
        kh_hi = k_hi.transpose(0, 1).repeat_interleave(g_rep, dim=0)
        vh_lo = v_lo.transpose(0, 1).repeat_interleave(g_rep, dim=0)
        vh_hi = v_hi.transpose(0, 1).repeat_interleave(g_rep, dim=0)

        mq, rq = 0.5 * (qh_lo + qh_hi), 0.5 * (qh_hi - qh_lo)
        mk, rk = 0.5 * (kh_lo + kh_hi), 0.5 * (kh_hi - kh_lo)
        ms = (mq @ mk.transpose(-1, -2)) * scale
        rs = (
            rq @ torch.abs(mk).transpose(-1, -2)
            + torch.abs(mq) @ rk.transpose(-1, -2)
            + rq @ rk.transpose(-1, -2)
        ) * scale
        rs = rs + 1e-13 * (torch.abs(ms) + rs) + 1e-15
        s_lo = ms - rs
        s_hi = ms + rs

        p_lo, p_hi = box_softmax_causal(s_lo, s_hi, future_mask)

        # 5. Attention weighted sum + sound causal simplex hull clamp
        mp, rp = 0.5 * (p_lo + p_hi), 0.5 * (p_hi - p_lo)
        mv, rv = 0.5 * (vh_lo + vh_hi), 0.5 * (vh_hi - vh_lo)
        mout = mp @ mv
        rout = rp @ torch.abs(mv) + torch.abs(mp) @ rv + rp @ rv
        rout = rout + 1e-13 * (torch.abs(mout) + rout) + 1e-15
        ao_lo = mout - rout
        ao_hi = mout + rout

        # Simplex hull across causal keys s <= t: out[h, t, d] in [cummin_s(vh_lo), cummax_s(vh_hi)]
        hull_lo = torch.cummin(vh_lo, dim=1).values - 1e-13 * torch.abs(vh_lo).cummax(dim=1).values - 1e-15
        hull_hi = torch.cummax(vh_hi, dim=1).values + 1e-13 * torch.abs(vh_hi).cummax(dim=1).values + 1e-15
        ao_lo = torch.maximum(ao_lo, hull_lo)
        ao_hi = torch.minimum(ao_hi, hull_hi)

        ao_lo = ao_lo.transpose(0, 1).contiguous().view(T, n_q * head_dim)
        ao_hi = ao_hi.transpose(0, 1).contiguous().view(T, n_q * head_dim)

        # 6. o_proj + residual
        o_lo, o_hi = box_linear(ao_lo, ao_hi, lw["o_q8"], lw["o_rw"])
        x_lo = res_lo + o_lo
        x_hi = res_hi + o_hi

        # 7. MLP post_norm -> gate/up -> SwiGLU -> down -> residual
        res_lo, res_hi = x_lo, x_hi
        h_lo, h_hi = box_rms_norm(x_lo, x_hi, lw["post_norm"], eps)
        g_lo, g_hi = box_linear(h_lo, h_hi, lw["gate_q8"], lw["gate_rw"])
        u_lo, u_hi = box_linear(h_lo, h_hi, lw["up_q8"], lw["up_rw"])
        m_lo, m_hi = box_silu_mul(g_lo, g_hi, u_lo, u_hi)
        d_lo, d_hi = box_linear(m_lo, m_hi, lw["down_q8"], lw["down_rw"])
        x_lo = res_lo + d_lo
        x_hi = res_hi + d_hi

        # Audit enclosure soundness + rho_l = median_i(rad_i / |dh_i|)
        h_true_l = true_states[l_idx - 1]
        h_q8_l = q8_states[l_idx - 1]
        viol_true = int(((h_true_l < x_lo) | (h_true_l > x_hi)).sum().item())
        viol_q8 = int(((h_q8_l < x_lo) | (h_q8_l > x_hi)).sum().item())
        assert viol_true == 0 and viol_q8 == 0, (
            f"Layer {l_idx} box unsound! viol_true={viol_true}, viol_q8={viol_q8}"
        )

        rad = 0.5 * (x_hi - x_lo)
        abs_dh = torch.abs(h_q8_l - h_true_l)
        eps_nz = 1e-18
        per_coord_ratio = rad / torch.clamp_min(abs_dh, eps_nz)
        # Per-token median over coordinates, then median and p95 across tokens
        rho_tok_med = torch.median(per_coord_ratio, dim=-1).values
        rho_med = float(torch.median(rho_tok_med).item())
        rho_p95 = float(torch.quantile(rho_tok_med, 0.95).item())
        max_rad = float(rad.max().item())
        mean_abs_dh = float(abs_dh.mean().item())

        rec = {
            "layer": l_idx,
            "rho_median": rho_med,
            "rho_p95": rho_p95,
            "rad_median": float(torch.median(rad).item()),
            "rad_max": max_rad,
            "true_dh_coord_mean": mean_abs_dh,
            "enclosure_violations_true": viol_true,
            "enclosure_violations_q8": viol_q8,
            "finite": bool(torch.isfinite(rad).all().item()),
        }
        layer_records.append(rec)
        print(
            f"  L{l_idx:02d} | rho_med={rho_med:12.4e} | rho_p95={rho_p95:12.4e} | "
            f"rad_max={max_rad:11.4e} | viol_true={viol_true} viol_q8={viol_q8}"
        )

        if first_exceed_1e3_layer is None and (not rec["finite"] or rho_med > 1e3):
            first_exceed_1e3_layer = l_idx
            print(f"  --> Stopped Step 1c-A at L{l_idx}: rho_med exceeded 1e3 ({rho_med:.4e}).")
            break

    last_rec = layer_records[-1]
    stop_1ca = bool((first_exceed_1e3_layer is not None) or (last_rec["rho_median"] > STOP_RULE_1CA_THRESHOLD))
    wall_1ca = time.time() - t_start
    print(
        f"STEP 1c-A verdict: last_layer={last_rec['layer']}, "
        f"rho_median={last_rec['rho_median']:.4e}, "
        f"STOP_RULE_CLOSED={stop_1ca} (threshold > {STOP_RULE_1CA_THRESHOLD:.1f}), "
        f"wall={wall_1ca:.2f}s"
    )
    return {
        "n_tokens": T,
        "hf_f64_check_l1_max_diff": err_l1,
        "hf_f64_check_l28_max_diff": err_l28,
        "layers": layer_records,
        "first_exceed_1e3_layer": first_exceed_1e3_layer,
        "last_layer": last_rec["layer"],
        "last_rho_median": last_rec["rho_median"],
        "last_rho_p95": last_rec["rho_p95"],
        "stop_rule_threshold": STOP_RULE_1CA_THRESHOLD,
        "deterministic_box_route_closed": stop_1ca,
        "wall_time_seconds": round(wall_1ca, 2),
    }


# =============================================================================
# Part B Helpers — Split Conformal Probabilistic Certificate
# =============================================================================

def quantize_model_layers_q8_inplace(model: nn.Module, shifted_half_step: bool = False) -> None:
    """Quantizes all 28 decoder Linear layers in-place to block-32 Q8 (grid1 or shifted grid2)."""
    for layer in model.model.layers:
        for _, lin in layer.named_modules():
            if isinstance(lin, nn.Linear):
                w = lin.weight.data.float()
                w_b = w.view(w.shape[0], w.shape[1] // 32, 32)
                m = torch.amax(torch.abs(w_b), dim=-1, keepdim=True)
                s = (m / 127.0).to(torch.float16).float()
                s = torch.where(s == 0.0, torch.ones_like(s), s)
                if not shifted_half_step:
                    q = torch.clamp(torch.round(w_b / s), -127.0, 127.0)
                else:
                    q = torch.clamp(torch.floor(w_b / s) + 0.5, -127.5, 127.5)
                lin.weight.data = (s * q).view(w.shape).to(torch.bfloat16)


def sync_dev(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run_prompts_collect(
    model: nn.Module,
    prompt_ids: list[torch.Tensor],
    device: torch.device,
    capture_layer_norms: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    """
    Evaluates disjoint prompts independently (no cross-prompt attention), returning:
      - h_final: (N_total, 1024) float32 on CPU
      - layer_norms: (N_total, 28) float32 on CPU if capture_layer_norms else None
      - tok_per_sec (timed end-to-end forward including lm_head + norms)
    """
    layer_norm_bufs: list[torch.Tensor] = []
    hooks = []
    if capture_layer_norms:
        for layer in model.model.layers:
            def _make_hook():
                def _hook(_mod, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    # hs: (1, T, 1024) -> L2 norm per token
                    layer_norm_bufs.append(torch.norm(hs[0].float(), p=2, dim=-1).cpu())
                return _hook
            hooks.append(layer.register_forward_hook(_make_hook()))

    # Warmup once on a 16-token slice so MPS kernels are compiled
    with torch.no_grad():
        _ = model(prompt_ids[0][:, :16].to(device))
        sync_dev(device)
        if capture_layer_norms:
            layer_norm_bufs.clear()

    h_list = []
    all_norms = []
    total_toks = 0
    t0 = time.time()
    sync_dev(device)
    t_compute_start = time.perf_counter()

    with torch.no_grad():
        for ids in prompt_ids:
            layer_norm_bufs.clear()
            ids_dev = ids.to(device)
            out = model(ids_dev, output_hidden_states=True)
            h_last = out.hidden_states[-1][0].float().cpu()
            h_list.append(h_last)
            total_toks += int(ids.shape[1])
            if capture_layer_norms:
                # 28 tensors each of shape (T,) -> (T, 28)
                all_norms.append(torch.stack(layer_norm_bufs, dim=-1))
        sync_dev(device)

    elapsed = time.perf_counter() - t_compute_start
    for hk in hooks:
        hk.remove()

    h_cat = torch.cat(h_list, dim=0)
    norms_cat = torch.cat(all_norms, dim=0) if capture_layer_norms else None
    tok_s = total_toks / max(elapsed, 1e-9)
    return h_cat, norms_cat, tok_s


def time_forward_toks(
    model: nn.Module, prompt_ids: list[torch.Tensor], device: torch.device, capture_layer_norms: bool = False
) -> float:
    """Clean timing pass matching production inference (no all-29 hidden_states copy)."""
    layer_norm_bufs: list[torch.Tensor] = []
    hooks = []
    if capture_layer_norms:
        for layer in model.model.layers:
            def _make_hook():
                def _hook(_mod, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    layer_norm_bufs.append(torch.norm(hs[0].float(), p=2, dim=-1))
                return _hook
            hooks.append(layer.register_forward_hook(_make_hook()))

    with torch.no_grad():
        _ = model(prompt_ids[0][:, :16].to(device))
        sync_dev(device)
        layer_norm_bufs.clear()

        total_toks = 0
        t_start = time.perf_counter()
        for ids in prompt_ids:
            layer_norm_bufs.clear()
            _ = model(ids.to(device), output_hidden_states=False)
            if capture_layer_norms:
                _stacked = torch.stack(layer_norm_bufs, dim=-1)
            total_toks += int(ids.shape[1])
        sync_dev(device)
        elapsed = time.perf_counter() - t_start

    for hk in hooks:
        hk.remove()
    return total_toks / max(elapsed, 1e-9)


def fit_nonneg_layer_weights(X_cal: torch.Tensor, y_cal: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    Fits c >= 0 in y ≈ X @ c on CALIBRATION data only via projected L-BFGS / PGD
    minimizing ||X c - y||_2^2 + 1e-4 * ||c||_2^2.
    """
    X = X_cal.double()
    y = y_cal.double()
    # Normalize scale for well-conditioned PGD
    col_scale = torch.norm(X, p=2, dim=0).clamp_min(1.0)
    Xn = X / col_scale
    XtX = Xn.t() @ Xn + 1e-4 * torch.eye(X.shape[1], dtype=torch.float64)
    Xty = Xn.t() @ y
    L_lip = float(torch.linalg.eigvalsh(XtX)[-1].item())
    step = 1.0 / max(L_lip, 1e-12)

    u = torch.clamp_min(torch.linalg.solve(XtX, Xty), 0.0)
    for _ in range(4000):
        grad = XtX @ u - Xty
        u_next = torch.clamp_min(u - step * grad, 0.0)
        if float(torch.max(torch.abs(u_next - u)).item()) < 1e-12:
            u = u_next
            break
        u = u_next

    c = (u / col_scale).float()
    pred = (X @ c.double()).float()
    rmse = float(torch.sqrt(torch.mean((pred - y_cal) ** 2)).item())
    corr = float(torch.corrcoef(torch.stack([pred, y_cal]))[0, 1].item())
    assert float(pred.min().item()) > 0.0, "Estimator (i) produced non-positive dh_hat"
    return c, {"cal_rmse": round(rmse, 6), "cal_pearson_r": round(corr, 6)}


def compute_conformal_scores(
    h_q8: torch.Tensor,
    h_bf16: torch.Tensor,
    dh_hat: torch.Tensor,
    w_head_q8_dev: torch.Tensor,
    w_head_bf16_dev: torch.Tensor,
    w_norms_dev: torch.Tensor,
    sqrt_d: float,
    device: torch.device,
    top_k_comp: int = 16,
    q_list: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    """
    Computes exact per-token nonconformity scores R_t = max_{j in C_t} (|d_j| / s_j)
    where C_t = Top-{top_k_comp}(z_q8[t]) union {true_top[t]}, and optionally
    evaluates HELD-OUT certificates for every q in q_list across ALL 151,936 vocab rows.
    """
    N = h_q8.shape[0]
    chunk = 256
    r_scores = torch.empty(N, dtype=torch.float32)
    r_scores_all = torch.empty(N, dtype=torch.float32)
    cert_res = None
    if q_list is not None:
        cert_res = {
            q: {
                "certified_count": 0,
                "false_certificates": 0,
                "score_exceedances_cand": 0,
                "score_exceedances_all": 0,
            }
            for q in q_list
        }
        top1_matches_total = 0

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        B = end - start
        idx_b = torch.arange(B, device=device)

        hq = h_q8[start:end].to(device)
        hb = h_bf16[start:end].to(device)
        dhat = dh_hat[start:end].to(device).clamp_min(1e-6)  # (B,)

        zq = (w_head_q8_dev @ hq.t()).t()  # (B, V)
        zb = (w_head_bf16_dev @ hb.t()).t()  # (B, V)
        drift = torch.abs(zq - zb)
        s_mat = torch.outer(dhat / sqrt_d, w_norms_dev).clamp_min(1e-12)  # (B, V)
        ratio_mat = drift / s_mat

        top_q = torch.argmax(zq, dim=-1)
        top_b = torch.argmax(zb, dim=-1)
        comp_idx = torch.topk(zq, k=top_k_comp, dim=-1).indices  # (B, K)

        # Row-wise max over top_k_comp competitors UNION true_top
        cand_ratios = torch.gather(ratio_mat, 1, comp_idx)
        max_comp = torch.max(cand_ratios, dim=-1).values
        true_top_ratio = ratio_mat[idx_b, top_b]
        r_batch = torch.maximum(max_comp, true_top_ratio)
        r_scores[start:end] = r_batch.cpu()
        r_scores_all[start:end] = torch.max(ratio_mat, dim=-1).values.cpu()

        if cert_res is not None:
            match_b = top_q == top_b
            top1_matches_total += int(match_b.sum().item())
            z_top = zq[idx_b, top_q]
            s_top = s_mat[idx_b, top_q]
            for q_val in q_list:
                lhs = z_top - q_val * s_top
                upper = (zq + q_val * s_mat).clone()
                upper[idx_b, top_q] = float("-inf")
                rhs = torch.max(upper, dim=-1).values
                cert_b = (lhs - rhs) > 0.0
                false_b = cert_b & (~match_b)
                cert_res[q_val]["certified_count"] += int(cert_b.sum().item())
                cert_res[q_val]["false_certificates"] += int(false_b.sum().item())
                cert_res[q_val]["score_exceedances_cand"] += int((r_batch > q_val).sum().item())
                cert_res[q_val]["score_exceedances_all"] += int((r_scores_all[start:end].to(device) > q_val).sum().item())
                del upper

    if cert_res is not None:
        cert_res["_top1_match_count"] = top1_matches_total
    return r_scores, r_scores_all, cert_res


def split_conformal_quantile(cal_scores: torch.Tensor, alpha: float) -> float:
    n = int(cal_scores.shape[0])
    sorted_s, _ = torch.sort(cal_scores)
    rank_1idx = min(n, max(1, math.ceil((n + 1) * (1.0 - alpha))))
    return float(sorted_s[rank_1idx - 1].item())


def run_step1c_b_conformal(tokenizer, model: nn.Module, device: torch.device) -> dict:
    print("\n" + "=" * 80)
    print("PHASE 7B STEP 1c-B — SPLIT-CONFORMAL PROBABILISTIC CERTIFICATE (HELD-OUT AUDIT)")
    print("=" * 80)
    t_b = time.time()

    # Verify strict disjointness of prompt sets
    assert set(CAL_PROMPTS.keys()).isdisjoint(set(HELD_OUT_PROMPTS.keys()))
    cal_ids = [tokenizer(txt, return_tensors="pt").input_ids for txt in CAL_PROMPTS.values()]
    test_ids = [tokenizer(txt, return_tensors="pt").input_ids for txt in HELD_OUT_PROMPTS.values()]
    n_cal = sum(int(x.shape[1]) for x in cal_ids)
    n_test = sum(int(x.shape[1]) for x in test_ids)
    print(f"Disjoint prompts: CAL={len(cal_ids)} prompts ({n_cal} tokens), HELD-OUT={len(test_ids)} prompts ({n_test} tokens)")
    assert n_cal >= 2000, f"n_cal {n_cal} < 2000"
    assert n_test >= 2000, f"n_test {n_test} < 2000"

    # Extract original BF16 lm_head + quantize Q8_EISA lm_head
    w_head_float = model.lm_head.weight.data.float().cpu()
    w_head_bf16_dev = w_head_float.to(device)
    w_head_q8, _, _ = quantize_w_eisa_q8(w_head_float)
    w_head_q8_dev = w_head_q8.to(device)
    w_norms_dev = torch.norm(w_head_bf16_dev, p=2, dim=-1)
    dim_d = int(w_head_float.shape[1])
    sqrt_d = math.sqrt(float(dim_d))

    # Save backup of BF16 decoder linear weights on CPU for exact grid1 & grid2 resets
    bf16_layer_weights = []
    for layer in model.model.layers:
        saved = {}
        for name, lin in layer.named_modules():
            if isinstance(lin, nn.Linear):
                saved[name] = lin.weight.data.cpu().clone()
        bf16_layer_weights.append(saved)

    # 1. BF16 Reference Forward
    h_cal_bf16, _, _ = run_prompts_collect(model, cal_ids, device, capture_layer_norms=False)
    h_test_bf16, _, _ = run_prompts_collect(model, test_ids, device, capture_layer_norms=False)
    bf16_tok_s = time_forward_toks(model, test_ids, device, capture_layer_norms=False)
    print(f"BF16 held-out baseline throughput: {bf16_tok_s:.2f} tok/s")

    # 2. Q8 Grid 1 (standard Q8_EISA) + capture 28 residual-stream L2 norms for Estimator (i)
    quantize_model_layers_q8_inplace(model, shifted_half_step=False)
    h_cal_q8, norms_cal_q8, _ = run_prompts_collect(model, cal_ids, device, capture_layer_norms=True)
    h_test_q8, norms_test_q8, _ = run_prompts_collect(model, test_ids, device, capture_layer_norms=True)
    q8_grid1_tok_s = time_forward_toks(model, test_ids, device, capture_layer_norms=True)
    print(f"Q8 + Est (i) (28 residual norms) held-out throughput: {q8_grid1_tok_s:.2f} tok/s")

    # 3. Restore BF16 weights -> Q8 Grid 2 (shifted half-step rounding grid) for Estimator (ii)
    for layer, saved in zip(model.model.layers, bf16_layer_weights):
        for name, lin in layer.named_modules():
            if isinstance(lin, nn.Linear):
                lin.weight.data = saved[name].to(device)
    del bf16_layer_weights

    quantize_model_layers_q8_inplace(model, shifted_half_step=True)
    h_cal_q8_g2, _, _ = run_prompts_collect(model, cal_ids, device, capture_layer_norms=False)
    h_test_q8_g2, _, _ = run_prompts_collect(model, test_ids, device, capture_layer_norms=False)
    q8_grid2_only_tok_s = time_forward_toks(model, test_ids, device, capture_layer_norms=False)
    # Total cost of Estimator (ii) = running both Q8 grid 1 and Q8 grid 2 passes sequentially
    est_ii_tok_s = 1.0 / ((1.0 / q8_grid1_tok_s) + (1.0 / q8_grid2_only_tok_s))
    print(
        f"Q8 + Est (ii) (2x Q8 pass: grid1 + shifted grid2) throughput: {est_ii_tok_s:.2f} tok/s "
        f"(pass2={q8_grid2_only_tok_s:.2f} tok/s)"
    )

    # Fit Estimator (i) STRICTLY on calibration tokens
    y_cal_dh = torch.norm(h_cal_q8 - h_cal_bf16, p=2, dim=-1)
    y_test_dh = torch.norm(h_test_q8 - h_test_bf16, p=2, dim=-1)
    c_weights, fit_stats = fit_nonneg_layer_weights(norms_cal_q8, y_cal_dh)

    dh_hat_cal_i = (norms_cal_q8 @ c_weights).clamp_min(1e-6)
    dh_hat_test_i = (norms_test_q8 @ c_weights).clamp_min(1e-6)

    dh_hat_cal_ii = (torch.norm(h_cal_q8 - h_cal_q8_g2, p=2, dim=-1) / math.sqrt(2.0)).clamp_min(1e-6)
    dh_hat_test_ii = (torch.norm(h_test_q8 - h_test_q8_g2, p=2, dim=-1) / math.sqrt(2.0)).clamp_min(1e-6)

    test_rmse_i = float(torch.sqrt(torch.mean((dh_hat_test_i - y_test_dh) ** 2)).item())
    test_r_i = float(torch.corrcoef(torch.stack([dh_hat_test_i, y_test_dh]))[0, 1].item())
    test_r_ii = float(torch.corrcoef(torch.stack([dh_hat_test_ii, y_test_dh]))[0, 1].item())
    print(
        f"dh_hat quality on HELD-OUT: Est(i) cal_r={fit_stats['cal_pearson_r']:.4f}, "
        f"test_r={test_r_i:.4f}, test_rmse={test_rmse_i:.4f} | "
        f"Est(ii) test_r={test_r_ii:.4f}"
    )

    alphas = [0.01, 0.001]
    estimators_summary = {}

    for est_key, est_label, dh_cal, dh_test, est_tok_s, extra_info in [
        (
            "estimator_i_residual_norms",
            "(i) Q8 residual norms (1x pass)",
            dh_hat_cal_i,
            dh_hat_test_i,
            q8_grid1_tok_s,
            {"fit_cal": fit_stats, "held_out_pearson_r": round(test_r_i, 6), "held_out_rmse": round(test_rmse_i, 6)},
        ),
        (
            "estimator_ii_shifted_grid",
            "(ii) Shifted Q8 grid (2x pass)",
            dh_hat_cal_ii,
            dh_hat_test_ii,
            est_ii_tok_s,
            {"held_out_pearson_r": round(test_r_ii, 6)},
        ),
    ]:
        cal_cand, _, _ = compute_conformal_scores(
            h_cal_q8, h_cal_bf16, dh_cal, w_head_q8_dev, w_head_bf16_dev, w_norms_dev, sqrt_d, device,
            top_k_comp=16, q_list=None
        )
        q_map = {a: split_conformal_quantile(cal_cand, a) for a in alphas}
        q_list = [q_map[a] for a in alphas]

        _, _, test_eval = compute_conformal_scores(
            h_test_q8, h_test_bf16, dh_test, w_head_q8_dev, w_head_bf16_dev, w_norms_dev, sqrt_d, device,
            top_k_comp=16, q_list=q_list
        )
        top1_match_cnt = test_eval["_top1_match_count"]

        alpha_results = {}
        print(f"\n  {est_label} | HELD-OUT top1 agreement: {top1_match_cnt}/{n_test} ({top1_match_cnt/n_test*100:.2f}%):")
        print("    alpha |     q     | HELD-OUT cert / N (  %  ) | false_certs | cand_exceed (%) | tok/s vs BF16")
        for a in alphas:
            qv = q_map[a]
            r = test_eval[qv]
            cert_c = r["certified_count"]
            false_c = r["false_certificates"]
            exc_c = r["score_exceedances_cand"]
            cert_frac = cert_c / n_test
            false_rate = false_c / n_test
            exc_rate = exc_c / n_test
            alpha_results[str(a)] = {
                "alpha": a,
                "conformal_quantile_q": round(qv, 6),
                "held_out_certified_count": cert_c,
                "held_out_certified_frac": round(cert_frac, 6),
                "held_out_certified_pct": round(cert_frac * 100.0, 2),
                "held_out_false_certificates": false_c,
                "held_out_false_certificate_rate": round(false_rate, 6),
                "held_out_cand_exceedances": exc_c,
                "held_out_cand_exceedance_rate": round(exc_rate, 6),
                "held_out_allrow_exceedances": r["score_exceedances_all"],
                "held_out_allrow_exceedance_rate": round(r["score_exceedances_all"] / n_test, 6),
            }
            print(
                f"    {a:<5g} | {qv:9.4f} | {cert_c:5d}/{n_test} ({cert_frac*100:5.2f}%) | "
                f"{false_c:11d} | {exc_c:4d} ({exc_rate*100:5.2f}%) | "
                f"{est_tok_s:7.1f} ({est_tok_s/bf16_tok_s:4.2f}x)"
            )

        estimators_summary[est_key] = {
            "label": est_label,
            "held_out_top1_match_count": top1_match_cnt,
            "held_out_top1_match_frac": round(top1_match_cnt / n_test, 6),
            "throughput_tok_per_s": round(est_tok_s, 2),
            "throughput_ratio_vs_bf16": round(est_tok_s / bf16_tok_s, 4),
            "diagnostics": extra_info,
            "alphas": alpha_results,
        }

    wall_1cb = time.time() - t_b
    return {
        "guarantee_scope": "marginal_over_exchangeable_tokens_not_per_token_deterministic",
        "n_cal_prompts": len(cal_ids),
        "n_cal_tokens": n_cal,
        "n_held_out_prompts": len(test_ids),
        "n_held_out_tokens": n_test,
        "candidate_rows_definition": "top16(z_q8) union {true_top_bf16}",
        "bf16_baseline_tok_per_s": round(bf16_tok_s, 2),
        "estimators": estimators_summary,
        "wall_time_seconds": round(wall_1cb, 2),
    }


def main() -> None:
    t0 = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR), trust_remote_code=True)

    # Load model in BF16 on MPS (also used to initialize Step 1c-A f64 weights cleanly)
    model = AutoModelForCausalLM.from_pretrained(
        str(CHECKPOINT_DIR),
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Exact 512 tokens from Step 1 / Phase 7B
    eval_ids_512 = torch.cat(
        [tokenizer(txt, return_tensors="pt").input_ids for txt in DOMAINS.values()],
        dim=1,
    )[:, :512]

    res_1ca = run_step1c_a_box_growth(model, eval_ids_512)
    res_1cb = run_step1c_b_conformal(tokenizer, model, device)

    report = {
        "title": "Phase 7B Step 1c — Box Propagation Growth & Split-Conformal Audit",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "step1c_a_box_growth": res_1ca,
        "step1c_b_split_conformal": res_1cb,
        "total_wall_time_seconds": round(time.time() - t0, 2),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved Step 1c report to: {REPORT_PATH}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
