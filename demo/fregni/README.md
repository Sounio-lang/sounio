# Sounio × PPCR Demo — Prof. Felipe Fregni

## What this is

A minimal, runnable demonstration of Sounio's unique selling point for clinical research: **uncertainty + provenance + confidence-gated execution**. The story is a dosing pipeline that rejects under-confident or wrongly-sourced inputs before they affect the analysis.

## Files

| File | Purpose |
|---|---|
| `fregni_demo.sio` | Main Sounio/Madaros program. Computes an AUC-based daily dose, rejects low-confidence simulated data, and rejects high-CV imputed data. |
| `bad_path.sio` | Compile-time provenance guard. Intentionally passes a `SimulationI64` to a `MeasuredGUMI64`-only extractor; must fail `bin/souc check`. |
| `reference.py` | Python reference arithmetic validating the integer-scaled Sounio calculations. |
| `run.sh` | One-command end-to-end runner. |
| `OUTPUT.md` | Captured real output from `run.sh`. |
| `TALK_TRACK.md` | 90-second narration for running the demo in a meeting. |

## Run it

```bash
cd demo/fregni
bash run.sh
```

## Environment assumptions

- OS: Linux x86_64
- Compiler: `bin/souc` from the repository root (Madares v0.80.0 at the time of writing)
- Python: 3.x for `reference.py`
- No GPU or external services required

## Clean-clone regression check

A fresh checkout of the repository needs only:

- `bin/souc` built/available at the repository root.
- Python 3.x on `PATH` for `reference.py`.

The demo uses only relative paths (`./run.sh`, `./fregni_demo.sio`, etc.) and does not depend on any unstaged file, environment variable, GPU, or network service. To verify:

```bash
cd demo/fregni
bash run.sh
```

## Honest scope

This is a teaching illustration, not a clinical tool. It uses fixed-point integer arithmetic (×100 scaling) and does not implement patient-specific renal adjustment, therapeutic drug monitoring, or regulatory-grade audit trails. Those are the kinds of PPCR-specific extensions a collaboration would build.
