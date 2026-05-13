# ZD-SSM — Sounio's deployable surgical State-Space Model

Paper-B through Paper-I (the Butterfly Thesis) culminates in this
artifact: a Mamba-family SSM whose output projection is replaced
by a **ZdGatedHead** — a sedenion-structured surgical layer that
makes `ExactlyPrivate<T>`, `Editable<T>`, `CapabilityGated<T>`,
`Composable<T>`, `Audited<T>`, `Revivable<T>`, and
`Interpretable<T>` available as algebraic identities at inference
time, with every operation emitting a machine-checkable Lean
witness.

## Directory layout

```
artifacts/zd-ssm/
├── model.sio         # ZdGatedHead type spec + FFI contract
├── train_lora.sio    # rank-8 LoRA harness with SurgicalSGD
├── inference.sio     # runtime driver (gate-apply + kernel-mass check)
├── audit.sio         # Lean-witness emitter
├── benchmarks/
│   ├── muse_bench_eval.sio   # unlearn evaluation
│   ├── zsre_eval.sio         # edit evaluation
│   └── wmdp_eval.sio         # capability-gating evaluation
└── dashboard/
    ├── index.html    # web UI to request surgical op + download .lean
    └── README.md     # wiring notes
```

## Why sedenions at the head

See Paper B. The 168 projective zero-divisor classes of the
sedenions give, *for free*, a canonical basis for surgical
operations on a 16-dim hidden chunk. Applying this basis to every
16-dim slice of Mamba's 768-dim hidden state gives 48 surgical
chunks, each with the full Paper-G six-fold calculus available.

## Bootstrap flow

1. Download a public Mamba-130M checkpoint (Python driver —
   outside Sounio) into `artifacts/zd-ssm/weights/`.
2. `souc run artifacts/zd-ssm/train_lora.sio` — verifies the LoRA
   harness type-checks, runs the ZD optimizer contract.
3. Python driver calls the Sounio FFI (`zd_ssm_forward`) per step;
   Sounio side applies `zd_gated_apply` and, on user request,
   `audit_unlearn_header` / `audit_edit_header` / `audit_gate_header`.
4. Dashboard exposes a POST endpoint that returns the Lean
   witness `.lean` file for the user to re-verify.

## Result targets (NOT guarantees)

| Benchmark   | Baseline                  | ZD-SSM target                  |
|-------------|---------------------------|--------------------------------|
| MUSE-bench  | grad-ascent 0.008 residual | ≥ +20pt forget-quality, ≥95% retention |
| zsRE        | ROME ~0.88 locality       | > 0.99 locality                |
| WMDP        | activation-patch residual | danger-acc at chance, benign-acc ± 2% |

## Status

- `model.sio`, `train_lora.sio`, `inference.sio`, `audit.sio` —
  implemented as type-level harnesses.
- `benchmarks/*` — implemented as Sounio-side scoring scripts that
  consume JSON output from the Python driver.
- `dashboard/` — static HTML form + stub API (documented, not
  served).  A production deployment would add a small Rust or Go
  server behind the form.

No AI attribution. No `sorry` in the companion Lean proofs.

## Related documents

- Paper B `paper_b_zdssm.tex` — the 168 theorem
- Paper C `paper_c_surgical_ml.tex` — first 3 surgical types
- Paper G `paper_g_surgical_calculus.tex` — 6-fold closure
- Paper D `paper_d_ami.tex` — AMI interpretability
- Paper F `paper_f_regulatory.tex` — compile-time GDPR/EU-AI-Act/HIPAA
- Paper E `paper_e_odep.tex` — ODEP privacy definition
- Paper H `paper_h_forgettable_dynamics.tex` — surgical optimizer
- Paper I `paper_i_cayley_dickson_tiers.tex` — L1-L5 ladder
