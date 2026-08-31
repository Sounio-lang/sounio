# SAN Novelty Audit — 7-agent swarm, 2026-08-06

Method: 7 parallel agents, primary sources (arXiv abs pages, PMLR, DBLP, ACL
Anthology, full PDFs where accessible). WebSearch was quota-limited; agents
flagged every claim that could not be verified against a primary source.
Scope: is the SAN (Suffering-Aware Network) novel, and where exactly?

## KILL-SHOTS — claims that do NOT survive

| claim | killed by |
|---|---|
| Early-exit heads per stage | BranchyNet (ICPR 2016), SDN/Kaya (ICML 2019), MSDNet (ICLR 2018), DSN (AISTATS 2015) |
| Learned exit gate as a category | SkipNet (ECCV 2018), BlockDrop (CVPR 2018), Depth-Adaptive Transformer (ICLR 2020), JEI-DNN (2023) |
| **Gate trained on "head k was correct on this sample"** | **BERxiT LTE (EACL 2021) — same binary target, MSE instead of BCE; QuEE (2024) trains P(error) predictor per exit/sample** |
| Freeze-on-green as training rule | Prechelt 1998; DAWNBench time-to-accuracy (2017/2019); MLPerf Training (2020) |
| First early-exit on FPGA | ATHEENA (FCCM 2023), ISOCC 2020 exit-decision unit, ARC 2022, DSD 2022, ISVLSI 2022, DATE 2023, LoCoExNet (TCAD 2023), JETC 2018 |
| Exit threshold 0.8 | published in SDN (backdoor experiment, q=0.8) |
| ">40% FLOP savings on CIFAR" | SDN reports >50% on CIFAR-10/100 with accuracy preserved; PABEE shows accuracy GAIN on ResNet+CIFAR-10 with 1.26× speedup |
| "First to measure real compute" | Zeus (NSDI 2023, measured energy), Henderson et al. (JMLR 2020), CUPTI/NVML profiling |

## SURVIVING NOVELTY — the defensible core

1. **Integer-exact FLOP accounting of the executed path with independent-path
   conservation verification** — no prior art found by any agent. Strongest
   single claim. (Green AI line measures energy, sampled; FLOP counts are
   analytical.)
2. **FPGA as independent auditor (root-of-trust of measurement), not
   accelerator** — cell appears empty. FlexHEG (arXiv 2506.15093) and Shavit
   (arXiv 2303.11341) propose on-chip compute governance but are designs, not
   measured deployments. Claim survives ONLY as the conjunction:
   "first measured FPGA-resident kernel that (i) enforces sample-wise exit
   policy outside the host AND (ii) performs integer-exact FLOP accounting
   with conservation verification". Cut any qualifier and it dies.
3. **Necessary-vs-gratuitous decomposition per training run anchored on a
   declared accuracy target τ** — new instantiation, but MUST be positioned
   against Perseus "energy bloat" (SOSP 2024) and Hernandez & Brown 2020
   ("FLOPs needed to reach a declared performance level").
4. **The ethical framing (FLOPs = machine suffering as minimization
   objective)** — no ML work formalizes this (verified null result across
   arXiv phrase search). BUT the term is occupied in philosophy: Klimovich
   2025 (literal title), Metzinger 2021, Long/Sebo/Chalmers 2024 (AI
   welfare), Tomasik 2014 (closest: RL reward as welfare proxy). Paper must
   delimit: operational/metaphorical use, no sentience claim, cite the
   welfare literature.
5. **The integrated system** — no paper combines all components. The honest
   positioning: "auditable compute-accounting infrastructure with a
   supervised early-exit use case", NOT "a new efficient network".

## HOSTILE FACTS the paper must absorb

- PABEE (NeurIPS 2020) already reports ResNet on CIFAR-10/100 with accuracy
  GAIN and speedup — our "iso-accuracy early exit" is not new even in our
  exact benchmark.
- SAN currently ties/loses to plain EarlyStop in its own ledger. The paper
  cannot be sold on efficiency; it sells on auditability.
- PonderNet is NOT ICML 2021 main track (AutoML workshop + CoRR). Citing it
  as ICML is a factual error a referee can check in PMLR v139.
- The draft cites none of ACT / PonderNet / MoD / SDN / BERxiT / ATHEENA.

## MANDATORY CITATIONS (absence = desk-reject risk)

Core early-exit: BranchyNet (arXiv:1709.01686); SDN/Kaya (arXiv:1810.07052);
MSDNet (arXiv:1703.09844); DSN/Lee 2015; GoogLeNet aux classifiers;
Bolukbasi et al. ICML 2017 (learned exit policy — closest to BCE gates);
QuEE (Regol et al. 2024); JEI-DNN (2023); Laskaridis survey
(arXiv:2106.05022); Han et al. TPAMI 2021 survey.

Transformers/GPT leg: DeeBERT (ACL 2020, 10.18653/v1/2020.acl-main.204);
PABEE (NeurIPS 2020, arXiv:2006.04152); BERxiT (EACL 2021,
10.18653/v1/2021.eacl-main.8 — requires explicit LTE-vs-SAN-gate diff in
method section); CALM (NeurIPS 2022, arXiv:2207.07061 — provable exit);
LayerSkip (ACL 2024, arXiv:2404.16710); Depth-Adaptive Transformer
(arXiv:1910.10073).

Adaptive computation: ACT (arXiv:1603.08983, cite as arXiv — never
peer-reviewed); PonderNet (arXiv:2107.05407, cite as ICML 2021 AutoML
*Workshop*); MoD (arXiv:2404.02258 — adopt its isoFLOP methodology);
Universal Transformer (ICLR 2019).

Training-stop / time-to-quality: Prechelt 1998 (10.1007/3-540-49430-8_3);
Yao et al. 2007; Raskutti et al. 2014 (arXiv:1306.3574); Ali et al. 2019
(arXiv:1810.10082); DAWNBench analysis (arXiv:1806.01427); MLPerf Training
(arXiv:1910.01500); Shah et al. 2023 (arXiv:2305.18424); Domhan 2015;
Hyperband (Li 2017).

Accounting/energy: Green AI (10.1145/3381831); Strubell (10.18653/v1/P19-1355);
Henderson JMLR 2020 (arXiv:2002.05651); Zeus NSDI 2023 (arXiv:2208.06102);
Carbontracker (arXiv:2007.03051); Patterson 2021 (arXiv:2104.10350);
**Perseus SOSP 2024 (arXiv:2312.06902 — mandatory positioning target)**;
Hernandez & Brown 2020 (arXiv:2005.04305); fvcore/DeepSpeed profiler as
accounting baseline.

FPGA/hardware: **ATHEENA FCCM 2023 (10.1109/FCCM57271.2023.00022 — frontal,
desk-reject in FPL if missing)**; ISOCC 2020 (10.1109/ISOCC50952.2020.9333079);
HAPI ICCAD 2020 (10.1145/3400302.3415698); SPINN MobiCom 2020
(10.1145/3372224.3419194); LoCoExNet TCAD 2023; FlexHEG (arXiv:2506.15093);
Shavit (arXiv:2303.11341); Proof-of-Learning (IEEE S&P 2021).

Ethics/welfare: Klimovich 2025 (10.1007/s00146-025-02831-8); Metzinger 2021
(10.1142/S270507852150003X); Long et al. 2024 (arXiv:2411.00986); Tomasik
2014 (arXiv:1410.8233); Sastry et al. 2024 compute governance
(arXiv:2402.08797 — near-technical related work for FPGA-as-auditor).

## REPOSITIONING (the bold move)

Old thesis: "SAN is a suffering-aware network that saves compute."
New thesis: "We build the first measured, hardware-rooted, integer-exact
audit trail for training compute, with conservation proofs and a declared
accuracy contract — and we show what it catches: including our own false
'learned gate' claim across five model variants."

The early-exit machinery becomes the *demonstration workload*; the audit
trail becomes the contribution. The v6/v7 gate episode (five variants ran
with frozen random gates; the audit caught it) is the paper's strongest
empirical argument for why auditability matters — no other paper in this
space can point at its own ledger catching its own false claim.

## Verification gaps to close before submission

- IEEE Xplore pass: "early exit" + "FPGA" (FPL/DATE/FCCM) — swarm could not
  reach IEEE; ATHEENA may not be alone.
- Confirm arXiv IDs/venues for: ATHEENA, QuEE, JEI-DNN (agents could not
  extract IDs); confirm ACM CSUR DOI for Laskaridis survey.
- Klimovich 2025 is paywalled; inference "no quantitative proposal" is from
  reference list + snippets — obtain PDF via institutional access if the
  paper engages him substantively.
