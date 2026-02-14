# Sounio Research Benchmarks

Production-quality benchmarks demonstrating epistemic types and causal programming for academic publications.

## Benchmark Suite

### 1. QNN-MNIST: Epistemic Quaternionic Neural Networks
**Directory:** `qnn/`
**Paper:** Epistemic Types for Scientific Computing (PLDI 2027)
**Status:** Planned Month 4-5

**Objective:** Demonstrate that epistemic types enable:
- Automatic uncertainty propagation through neural networks
- Better calibration than standard Bayesian methods
- Type-safe gradient computation with uncertainty

**Metrics:**
- Test accuracy: Target 98%+
- Expected Calibration Error (ECE): Target < 0.05
- Uncertainty coverage: 95% confidence intervals should contain true labels 95% of the time

**Comparison Baselines:**
- PyTorch QNN (no uncertainty)
- Bayesian NN with dropout
- Ensemble methods

**Implementation:**
```sio
// Weights with epistemic uncertainty
struct QNNLayer {
    weights: Knowledge<Quaternion>,
    bias: Knowledge<Quaternion>
}

// Automatic uncertainty propagation through forward pass
fn forward(layer: QNNLayer, input: Knowledge<Quaternion>) -> Knowledge<Quaternion> {
    // Type system automatically propagates uncertainty
    return layer.weights * input + layer.bias
}
```

**Files:**
- `mnist_epistemic.sio` - Main training script
- `qnn_uncertainty.sio` - Epistemic QNN layers
- `calibration.sio` - ECE and reliability diagram computation
- `validate.sh` - Run all comparisons
- `results/` - Output data and figures for paper

---

### 2. PBPK: Pharmacokinetic Modeling with Causal Interventions
**Directory:** `pbpk/`
**Paper:** Causal Programming with do-Calculus Types (PLDI/UAI 2027)
**Status:** Planned Month 8-9

**Objective:** Show that causal+epistemic types enable:
- Type-safe do-operator for interventional queries
- Compile-time detection of non-identifiable causal queries
- Uncertainty propagation through ODE integration + causal reasoning

**Metrics:**
- Validation vs FDA-approved PBPK reference: < 5% error
- Identifiability: Reject non-identifiable queries at compile-time
- Performance: < 100ms per simulation

**Causal Graph:**
```
Drug Dose → Plasma Concentration → Clinical Effect
               ↑
Genotype → Metabolism Rate
```

**Key Feature:**
```sio
// Compile-time identifiability check
fn causal_effect(
    g: CausalGraph,
    intervention: do(Dose = 100.0 mg)
) -> CausalKnowledge<Concentration, ε, G>
where identifiable(g, Dose, Concentration)  // SMT-verified!
{
    // Type system proves this is sound
}
```

**Files:**
- `causal_intervention.sio` - PBPK model with do-operator
- `identifiability_tests.sio` - Test cases for non-identifiable queries
- `validation/fda_reference.csv` - Validation data
- `run.sh` - Run simulations
- `results/` - Concentration-time curves, AUC comparison

---

### 3. fMRI: Causal Connectivity Analysis
**Directory:** `fmri/`
**Paper:** Causal Programming with do-Calculus Types (PLDI/UAI 2027)
**Status:** Planned Month 10-11

**Objective:** Demonstrate causal types for neuroimaging:
- Distinguish spurious correlation from true causal paths
- Bootstrap uncertainty propagation with epistemic types
- Compile-time rejection of non-identifiable connectivity claims

**Dataset:** Human Connectome Project (HCP) resting-state fMRI

**Metrics:**
- Detect non-identifiable connections: Precision > 90%
- Uncertainty calibration: 95% CIs cover ground truth
- Comparison vs SPM/FSL/AFNI

**Key Feature:**
```sio
// Causal DAG from neuroanatomy
let brain_graph = CausalGraph::from_atlas(AAL_116)

// Type error if non-identifiable!
let causal_connection: CausalKnowledge<f64, ε> =
    compute_connectivity(
        fmri_data,
        brain_graph,
        source: "V1",
        target: "MT",
        adjustment_set: ["LGN"]  // SMT checks d-separation
    )
```

**Files:**
- `causal_connectivity.sio` - Main analysis pipeline
- `bootstrap_uncertainty.sio` - Epistemic resampling
- `validation/ground_truth.csv` - Known anatomical connections
- `run.sh` - Process HCP data
- `results/` - Connectivity matrices, ROC curves

---

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── qnn/
│   ├── mnist_epistemic.sio
│   ├── qnn_uncertainty.sio
│   ├── calibration.sio
│   ├── validate.sh
│   ├── data/                    # MNIST dataset
│   ├── results/                 # Figures for paper
│   └── validation/              # Baseline comparisons
├── pbpk/
│   ├── causal_intervention.sio
│   ├── identifiability_tests.sio
│   ├── run.sh
│   ├── data/                    # Physiological parameters
│   ├── results/                 # Concentration curves
│   └── validation/              # FDA reference data
└── fmri/
    ├── causal_connectivity.sio
    ├── bootstrap_uncertainty.sio
    ├── run.sh
    ├── data/                    # HCP dataset (not in git)
    ├── results/                 # Connectivity matrices
    └── validation/              # Anatomical ground truth
```

## Running Benchmarks

### Prerequisites
```bash
# Build Sounio compiler with all features
cd crates/souc && cargo build --release --features "smt,jit"

# Set up Python environment for baseline comparisons
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy matplotlib seaborn
```

### QNN-MNIST
```bash
cd benchmarks/qnn
./validate.sh
# Outputs: results/accuracy.png, results/calibration.png, results/comparison.csv
```

### PBPK
```bash
cd benchmarks/pbpk
./run.sh
# Outputs: results/concentration_time.png, results/validation_error.csv
```

### fMRI
```bash
cd benchmarks/fmri
# Note: Requires HCP dataset (download separately)
./run.sh
# Outputs: results/connectivity_matrix.png, results/roc_curve.png
```

## Validation Criteria

### For PLDI/ICFP Submission
- All benchmarks must run to completion
- Results must match paper figures
- Performance overhead < 10% vs baseline
- All identifiability tests must pass

### For Reproducibility
- Dockerized environment for all benchmarks
- Random seeds fixed for deterministic results
- All data dependencies documented
- Runtime < 1 hour per benchmark on standard hardware

## Timeline

```
Month 4-5:  QNN-MNIST implementation + validation
Month 8-9:  PBPK causal intervention + FDA validation
Month 10-11: fMRI connectivity + bootstrap uncertainty
Month 12:   All benchmarks integrated into CI/CD
```

## Expected Paper Contributions

**Epistemic Types Paper (PLDI 2027):**
- QNN-MNIST: Shows epistemic types improve calibration
- Performance analysis: Overhead measurements

**Causal Types Paper (PLDI/UAI 2027):**
- PBPK: Medical case study with do-operator
- fMRI: Neuroscience case study detecting spurious correlations
- Identifiability tests: Compile-time rejection of bad queries

## Citation

```bibtex
@misc{sounio-benchmarks-2027,
  title={Sounio Research Benchmarks},
  author={Sounio Research Team},
  year={2027},
  url={https://github.com/sounio-lang/sounio/tree/main/benchmarks}
}
```
