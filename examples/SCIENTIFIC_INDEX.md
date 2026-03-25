# Sounio Scientific Examples Index

Curated examples organized by research domain. All examples can be run with:

```bash
SOUC=./bin/souc
$SOUC run examples/<file>.sio
```

---

## 1. Getting Started

- [`hello.sio`](hello.sio) — First Sounio program (minimal main function)
- [`minimal.sio`](minimal.sio) — Minimal program returning a constant
- [`arithmetic.sio`](arithmetic.sio) — Basic addition and arithmetic operations
- [`fibonacci.sio`](fibonacci.sio) — Recursive Fibonacci sequence with control flow
- [`structs.sio`](structs.sio) — Struct definitions and field access
- [`effects_simple.sio`](effects_simple.sio) — Simple algebraic effects with IO
- [`wave1_hello_scientific.sio`](wave1_hello_scientific.sio) — Minimal scientific computing via interpreter
- [`scientific_hello.sio`](scientific_hello.sio) — f64 output from native ELF compilation
- [`showcase/type_safe_units.sio`](showcase/type_safe_units.sio) — Compile-time dimensional analysis preventing unit mismatches
- [`showcase/measurement_lab.sio`](showcase/measurement_lab.sio) — ISO GUM-compliant uncertainty propagation through calculations
- [`effects/comprehensive_effects.sio`](effects/comprehensive_effects.sio) — End-to-end test for effect handlers with continuations

## 2. Uncertainty Quantification / GUM

### Core Epistemic Modules

- [`epistemic/gum_demo.sio`](epistemic/gum_demo.sio) — GUM (Guide to Uncertainty in Measurement) module demo
- [`epistemic/core_demo.sio`](epistemic/core_demo.sio) — Epistemic core: Knowledge\<T\> type and uncertainty basics
- [`epistemic/combine_demo.sio`](epistemic/combine_demo.sio) — Combining independent uncertain measurements
- [`epistemic/correlation_demo.sio`](epistemic/correlation_demo.sio) — Correlated uncertainty propagation
- [`epistemic/coverage_demo.sio`](epistemic/coverage_demo.sio) — Coverage intervals and expanded uncertainty
- [`epistemic/budget_demo.sio`](epistemic/budget_demo.sio) — Uncertainty budget analysis
- [`epistemic/montecarlo_demo.sio`](epistemic/montecarlo_demo.sio) — Monte Carlo uncertainty propagation
- [`epistemic/multivariate_demo.sio`](epistemic/multivariate_demo.sio) — Multivariate uncertainty with covariance
- [`epistemic/stats_demo.sio`](epistemic/stats_demo.sio) — Epistemic statistics module demo
- [`epistemic/fusion_demo.sio`](epistemic/fusion_demo.sio) — Sensor/data fusion with uncertainty
- [`epistemic/interval_ieee_demo.sio`](epistemic/interval_ieee_demo.sio) — IEEE-compliant interval arithmetic
- [`epistemic/sobol_demo.sio`](epistemic/sobol_demo.sio) — Sobol sensitivity indices for variance decomposition

### Polynomial Chaos Expansion

- [`epistemic/pce_demo.sio`](epistemic/pce_demo.sio) — PCE for nonlinear uncertainty propagation (superior to first-order GUM)
- [`epistemic/pce_complete_demo.sio`](epistemic/pce_complete_demo.sio) — Complete PCE: univariate, bivariate, and multi-input systems
- [`epistemic/pce_test.sio`](epistemic/pce_test.sio) — PCE validation tests

### Epistemic Infrastructure

- [`epistemic/invariants_demo.sio`](epistemic/invariants_demo.sio) — Epistemic invariant checking
- [`epistemic/ledger_demo.sio`](epistemic/ledger_demo.sio) — Uncertainty provenance ledger
- [`epistemic/policy_demo.sio`](epistemic/policy_demo.sio) — Uncertainty policy enforcement
- [`epistemic/proptest_demo.sio`](epistemic/proptest_demo.sio) — Property-based testing with uncertainty
- [`epistemic/prov_demo.sio`](epistemic/prov_demo.sio) — Provenance tracking for uncertain values
- [`epistemic/refutation_demo.sio`](epistemic/refutation_demo.sio) — Epistemic refutation testing
- [`epistemic/roi_demo.sio`](epistemic/roi_demo.sio) — Return-on-investment for uncertainty reduction
- [`epistemic/slsa_demo.sio`](epistemic/slsa_demo.sio) — SLSA supply chain security for epistemic pipelines
- [`epistemic/traceability_demo.sio`](epistemic/traceability_demo.sio) — Measurement traceability chain

### Applied Epistemic Computing

- [`epistemic_bmi.sio`](epistemic_bmi.sio) — BMI calculation with first-order uncertainty propagation
- [`epistemic_propagation.sio`](epistemic_propagation.sio) — RSS uncertainty propagation with SMT verification
- [`epistemic_refinements.sio`](epistemic_refinements.sio) — Refinement types with SMT-verified epistemic bounds
- [`epistemic_dempster_shafer.sio`](epistemic_dempster_shafer.sio) — Dempster-Shafer evidence combination with type guarantees
- [`epistemic_smoke_native.sio`](epistemic_smoke_native.sio) — Native x86-64 ELF epistemic smoke test (measure, GUM arithmetic, print_f64)
- [`epistemic_gpu_pipeline.sio`](epistemic_gpu_pipeline.sio) — Full epistemic GPU pipeline with GUM through multi-backend computation
- [`clinical_trial_epistemic.sio`](clinical_trial_epistemic.sio) — Clinical trial pipeline: random sampling, PK simulation, Bayesian analysis
- [`financial_risk_epistemic.sio`](financial_risk_epistemic.sio) — Financial risk: GBM + OU spread with epistemic uncertainty
- [`science/uncertainty_propagation.sio`](science/uncertainty_propagation.sio) — GUM BMI and BSA uncertainty propagation (Mosteller formula)
- [`science/epistemic_cascade.sio`](science/epistemic_cascade.sio) — Uncertainty amplification through clinical measurement chains
- [`real_world/01_dose_uncertainty.sio`](real_world/01_dose_uncertainty.sio) — Pharmaceutical dose calculation with uncertainty and safety bounds
- [`real_world/04_gum_measurement_chain.sio`](real_world/04_gum_measurement_chain.sio) — ISO GUM-compliant measurement traceability with Type-A/Type-B uncertainty
- [`real_world/04_gum_measurement_simple.sio`](real_world/04_gum_measurement_simple.sio) — Simplified GUM measurement demo
- [`real_world/07_sensor_fusion.sio`](real_world/07_sensor_fusion.sio) — Bayesian GPS/IMU sensor fusion via Kalman filter with epistemic uncertainty

## 3. Pharmacokinetics / PBPK

### Core PK Models

- [`pharmacokinetic_model.sio`](pharmacokinetic_model.sio) — One-compartment PK with epistemic Knowledge\<T\> types
- [`pbpk_simple.sio`](pbpk_simple.sio) — Simplified one-compartment PBPK with uncertainty fields
- [`epistemic_ddi_simulator.sio`](epistemic_ddi_simulator.sio) — Drug-drug interaction simulator with GUM uncertainty propagation
- [`epistemic/pk_example.sio`](epistemic/pk_example.sio) — Pharmacokinetic example with rigorous epistemic semantics
- [`native_epistemic_pk.sio`](native_epistemic_pk.sio) — Native ELF PK: Gauss-Newton fitting, pure Sounio, no libc
- [`science/pkpd_simulation.sio`](science/pkpd_simulation.sio) — One-compartment PK with RK4 integration
- [`showcase/drug_dose_optimizer.sio`](showcase/drug_dose_optimizer.sio) — Two-compartment PK modeling with epistemic uncertainty
- [`lethal_dose_sedenion.sio`](lethal_dose_sedenion.sio) — Epistemic warfarin dosing decision (lethal INR risk assessment)

### PBPK Library Demos

- [`pbpk/population_demo.sio`](pbpk/population_demo.sio) — Population PK module demo
- [`pbpk/error_models_demo.sio`](pbpk/error_models_demo.sio) — PK error models (additive, proportional, combined)
- [`pbpk/types_demo.sio`](pbpk/types_demo.sio) — PBPK type system demo
- [`pbpk/covariate_demo.sio`](pbpk/covariate_demo.sio) — Covariate modeling for PK parameters

### Darwin PBPK Platform

- [`darwin_pbpk/rodgers_rowland_demo.sio`](darwin_pbpk/rodgers_rowland_demo.sio) — Rodgers-Rowland Kp prediction for tissue partition coefficients
- [`darwin_pbpk/simulation_demo.sio`](darwin_pbpk/simulation_demo.sio) — Darwin PBPK simulation platform with drug/patient combinations
- [`darwin_pbpk/tsit5_pbpk14_demo.sio`](darwin_pbpk/tsit5_pbpk14_demo.sio) — Tsitouras 5(4) ODE solver for 14-compartment PBPK
- [`paper_validation.sio`](paper_validation.sio) — Emax PD sensitivity + PBPK Euler convergence validation

### ODE Solvers for PK

- [`ode/rk4_demo.sio`](ode/rk4_demo.sio) — Runge-Kutta 4th order ODE solver
- [`ode/tsit5_demo.sio`](ode/tsit5_demo.sio) — Tsitouras 5(4) adaptive ODE solver
- [`ode/tsit5_multicomp_demo.sio`](ode/tsit5_multicomp_demo.sio) — Multi-compartment ODE integration
- [`ode/solver_demo.sio`](ode/solver_demo.sio) — General ODE solver interface
- [`ode/pbpk14_demo.sio`](ode/pbpk14_demo.sio) — 14-compartment PBPK ODE system
- [`ode/pbpk3_stable_demo.sio`](ode/pbpk3_stable_demo.sio) — 3-compartment stable PBPK demo
- [`real_world/02_pbpk_oral_absorption.sio`](real_world/02_pbpk_oral_absorption.sio) — Two-compartment PBPK oral absorption with RK4
- [`real_world/05_pkpd_data_analysis.sio`](real_world/05_pkpd_data_analysis.sio) — PKPD data analysis pipeline

### MedLang DSL

- [`medlang/lexer_demo.sio`](medlang/lexer_demo.sio) — MedLang DSL lexer for pharmacometric models
- [`medlang/parser_demo.sio`](medlang/parser_demo.sio) — MedLang DSL parser demo
- [`medlang/codegen_demo.sio`](medlang/codegen_demo.sio) — MedLang code generation
- [`medlang/population/model_demo.sio`](medlang/population/model_demo.sio) — Population PK model definition
- [`medlang/population/simulation_demo.sio`](medlang/population/simulation_demo.sio) — Population simulation runner
- [`medlang/population/estimation_demo.sio`](medlang/population/estimation_demo.sio) — Population parameter estimation
- [`medlang/population/variability_demo.sio`](medlang/population/variability_demo.sio) — Inter-individual variability modeling

## 4. Neuroscience / Connectomics

### Brain Connectome Analysis

- [`brain_orc_demo.sio`](brain_orc_demo.sio) — Brain ORC + associator field pipeline (ASD vs TD connectome analysis)
- [`brain_associator_demo.sio`](brain_associator_demo.sio) — Clinical brain network analysis (ASD and ADHD FC matrices)
- [`oct_connectome_demo.sio`](oct_connectome_demo.sio) — First-ever associator fields on octonion-labeled graphs across topologies
- [`clinical_curvature_analysis.sio`](clinical_curvature_analysis.sio) — Epistemic ORC pipeline on psychiatric networks (depression severity)

### fMRI Processing

- [`fmri/atlas_demo.sio`](fmri/atlas_demo.sio) — fMRI brain atlas parcellation
- [`fmri/connectivity_demo.sio`](fmri/connectivity_demo.sio) — Functional connectivity matrix computation
- [`fmri/connectivity_epistemic_demo.sio`](fmri/connectivity_epistemic_demo.sio) — Epistemic functional connectivity with uncertainty
- [`fmri/nifti_demo.sio`](fmri/nifti_demo.sio) — NIfTI neuroimaging file format I/O
- [`fmri/pipeline_demo.sio`](fmri/pipeline_demo.sio) — Full fMRI processing pipeline
- [`fmri/preprocess_demo.sio`](fmri/preprocess_demo.sio) — fMRI preprocessing (motion correction, filtering)
- [`fusion/eeg_fmri_demo.sio`](fusion/eeg_fmri_demo.sio) — EEG-fMRI multimodal data fusion

### Connectivity and Networks

- [`connectivity/network_metrics_demo.sio`](connectivity/network_metrics_demo.sio) — Network metrics (clustering, path length, modularity)
- [`connectivity/phase_demo.sio`](connectivity/phase_demo.sio) — Phase synchronization analysis
- [`graph/curvature_demo.sio`](graph/curvature_demo.sio) — Ollivier-Ricci curvature on graphs
- [`graph/entropy_demo.sio`](graph/entropy_demo.sio) — Graph entropy measures
- [`graph/coherence_demo.sio`](graph/coherence_demo.sio) — Graph coherence analysis

### Signal Processing (Neuro)

- [`signal/spectral_demo.sio`](signal/spectral_demo.sio) — Spectral analysis (FFT, power spectral density)
- [`signal/filter_demo.sio`](signal/filter_demo.sio) — Signal filtering (bandpass, notch, etc.)
- [`signal/epoch_demo.sio`](signal/epoch_demo.sio) — Signal epoching for event-related analysis
- [`signal/fractal_demo.sio`](signal/fractal_demo.sio) — Fractal analysis of neural time series

## 5. Hypercomplex Algebra

### Octonions

- [`octonion_example.sio`](octonion_example.sio) — Octonion neural network with native 8D support and G2 representations
- [`octonion_nn_demo.sio`](octonion_nn_demo.sio) — Octonion-based MLP with 8x parameter efficiency
- [`oct_conjecture_test.sio`](oct_conjecture_test.sio) — Testing three conjectures on octonion alternativity and associators
- [`onn_rotation_prediction.sio`](onn_rotation_prediction.sio) — 3D rotation prediction with octonion neural networks

### Sedenions

- [`sedenion_168_verify.sio`](sedenion_168_verify.sio) — Is 336 = 2 x 168 a coincidence? Octonion associators vs sedenion zero-divisors
- [`sedenion_unitarity_break.sio`](sedenion_unitarity_break.sio) — Where unitarity dies in the Cayley-Dickson tower
- [`sedenion_zero_div_hunt.sio`](sedenion_zero_div_hunt.sio) — Systematic zero-divisor search in sedenions (Moreno 1998)
- [`sedenion_hessian_brain_demo.sio`](sedenion_hessian_brain_demo.sio) — Mandelbrot second derivative for sedenion neural networks
- [`hsi_sedenion_demo.sio`](hsi_sedenion_demo.sio) — Sedenion neural networks for hyperspectral tissue classification
- [`hsi_tissue_classification.sio`](hsi_tissue_classification.sio) — HSI tissue classification with sedenion MLPs
- [`run_sedenion_benchmark.sio`](run_sedenion_benchmark.sio) — Sedenion benchmark with PAC learning bounds

### Cayley-Dickson Tower

- [`cayley_dickson_hessian_tower.sio`](cayley_dickson_hessian_tower.sio) — Mandelbrot Hessian null-space vs associator structure across the tower

### Quaternions

- [`gpu_hypercomplex.sio`](gpu_hypercomplex.sio) — Hypercomplex arithmetic benchmark (CPU baseline for GPU comparison)

## 6. Machine Learning

### Quaternionic Neural Networks

- [`qnn/01_hello_quaternion.sio`](qnn/01_hello_quaternion.sio) — Simplest introduction to quaternions in Sounio
- [`qnn/02_basic_linear.sio`](qnn/02_basic_linear.sio) — Single quaternion linear layer (weights, forward pass, activation)
- [`qnn_complete_demo.sio`](qnn_complete_demo.sio) — Full QNN training pipeline using all stdlib/qnn modules
- [`qnn_mnist.sio`](qnn_mnist.sio) — QNN MNIST digit classification with 4x parameter efficiency

### Neural Networks

- [`nn/dense_demo.sio`](nn/dense_demo.sio) — Dense layer module demo
- [`nn/dense2_demo.sio`](nn/dense2_demo.sio) — Extended dense layer demo
- [`nn/activation_demo.sio`](nn/activation_demo.sio) — Activation functions (ReLU, sigmoid, tanh, etc.)
- [`nn/autograd_demo.sio`](nn/autograd_demo.sio) — Autograd engine for neural networks
- [`nn/tensor_demo.sio`](nn/tensor_demo.sio) — Tensor operations module demo
- [`nn/mlp_xor_demo.sio`](nn/mlp_xor_demo.sio) — MLP learning XOR function
- [`nn/mlp_classifier_demo.sio`](nn/mlp_classifier_demo.sio) — MLP classifier demo
- [`nn/quaternion_demo.sio`](nn/quaternion_demo.sio) — Quaternion neural network layer
- [`nn/dense_quaternion_demo.sio`](nn/dense_quaternion_demo.sio) — Dense quaternion layer demo
- [`nn/g2_equivariant_demo.sio`](nn/g2_equivariant_demo.sio) — G2-equivariant neural network layers
- [`nn/optimizers_quaternion_demo.sio`](nn/optimizers_quaternion_demo.sio) — Quaternion-aware optimizers (Adam, SGD variants)
- [`nn/pbpk_example.sio`](nn/pbpk_example.sio) — Quaternion neural networks for PBPK modeling

### Automatic Differentiation

- [`autodiff/dual_demo.sio`](autodiff/dual_demo.sio) — Forward-mode AD via dual numbers
- [`autodiff/tape_demo.sio`](autodiff/tape_demo.sio) — Reverse-mode AD via tape-based backpropagation
- [`autodiff/grad_demo.sio`](autodiff/grad_demo.sio) — Gradient computation module demo
- [`autodiff/epistemic_dual_demo.sio`](autodiff/epistemic_dual_demo.sio) — Epistemic dual numbers (AD + uncertainty)
- [`epistemic/autodiff_demo.sio`](epistemic/autodiff_demo.sio) — Epistemic autodiff module demo
- [`ml/autodiff_demo.sio`](ml/autodiff_demo.sio) — ML autodiff module demo
- [`wave3_symbolic_calculus.sio`](wave3_symbolic_calculus.sio) — Symbolic differentiation and expression manipulation

## 7. Causal Inference

- [`causal/core_demo.sio`](causal/core_demo.sio) — Causal core: DAG construction and do-calculus
- [`causal/discovery_demo.sio`](causal/discovery_demo.sio) — Causal structure discovery from data
- [`causal/refutation_demo.sio`](causal/refutation_demo.sio) — Causal refutation testing (sensitivity analysis)
- [`causal/uplift_demo.sio`](causal/uplift_demo.sio) — Causal uplift modeling for treatment effects
- [`causal_model.sio`](causal_model.sio) — Causal DAG DSL with structural equations
- [`wave2_causal_intervention.sio`](wave2_causal_intervention.sio) — Causal intervention with do-calculus (observation vs intervention)
- [`wave2_causal_simpson.sio`](wave2_causal_simpson.sio) — Simpson's paradox resolution via causal reasoning (UC Berkeley admissions)
- [`epistemic/causal_demo.sio`](epistemic/causal_demo.sio) — Epistemic causal inference module demo
- [`science/simpsons_paradox.sio`](science/simpsons_paradox.sio) — Simpson's paradox detection and causal correction
- [`render/causal_dag.sio`](render/causal_dag.sio) — Causal DAG visualization renderer

## 8. GPU Computing

### Kernel Programming

- [`kernel_vec_add.sio`](kernel_vec_add.sio) — GPU kernel for vector addition (kernel fn syntax)
- [`kernel_matmul.sio`](kernel_matmul.sio) — GPU kernel for matrix multiplication (multiple kernels per file)
- [`kernel_epistemic_vec_add.sio`](kernel_epistemic_vec_add.sio) — Epistemic GPU vector addition with uncertainty propagation
- [`kernel_epistemic_wmma_matmul.sio`](kernel_epistemic_wmma_matmul.sio) — World-first: GUM uncertainty through tensor core WMMA operations
- [`kernel_source_level.sio`](kernel_source_level.sio) — Source-level kernel compilation via HLIR-to-GPU lowering

### GPU Library Demos

- [`gpu.sio`](gpu.sio) — GPU profile example (check, build to PTX)
- [`gpu/fft_demo.sio`](gpu/fft_demo.sio) — GPU-accelerated FFT
- [`gpu/smooth_demo.sio`](gpu/smooth_demo.sio) — GPU smoothing operations
- [`gpu/stats_demo.sio`](gpu/stats_demo.sio) — GPU-accelerated statistical computations
- [`gpu/mod_demo.sio`](gpu/mod_demo.sio) — GPU module overview
- [`gpu_epistemic_showcase.sio`](gpu_epistemic_showcase.sio) — End-to-end epistemic GPU computing showcase
- [`fractal/gpu/box_counting_demo.sio`](fractal/gpu/box_counting_demo.sio) — GPU-accelerated fractal box counting

## 9. Cybernetics / Computational Psychiatry

- [`cybernetic_demo.sio`](cybernetic_demo.sio) — Second-order cybernetics: all nine modules end-to-end
- [`computational_psychiatry_demo.sio`](computational_psychiatry_demo.sio) — Psychiatric phenomena emerging from second-order cybernetics
- [`deep_psychiatry_demo.sio`](deep_psychiatry_demo.sio) — Deep computational psychiatry via Bateson's full theory
- [`therapy_session.sio`](therapy_session.sio) — Therapy session modeled as Pask conversation with Bateson learning levels
- [`science/active_inference.sio`](science/active_inference.sio) — Friston active inference agent (free energy principle)

## 10. Scientific Computing

### Bayesian Statistics

- [`bayes/mcmc_demo.sio`](bayes/mcmc_demo.sio) — Markov chain Monte Carlo sampling
- [`bayes/vi_demo.sio`](bayes/vi_demo.sio) — Variational inference
- [`bayes/prior_demo.sio`](bayes/prior_demo.sio) — Prior distribution specification
- [`bayes/diagnostics_demo.sio`](bayes/diagnostics_demo.sio) — MCMC convergence diagnostics
- [`science/bayesian_clinical_trial.sio`](science/bayesian_clinical_trial.sio) — Bayesian sequential clinical trial (Beta-Binomial)
- [`science/markov_chain_monte_carlo.sio`](science/markov_chain_monte_carlo.sio) — Metropolis-Hastings MCMC for posterior inference
- [`real_world/03_trial_sample_size.sio`](real_world/03_trial_sample_size.sio) — Clinical trial sample size with Bayesian updating

### Probability and Statistics

- [`prob/distributions_demo.sio`](prob/distributions_demo.sio) — Probability distributions
- [`prob/mcmc_demo.sio`](prob/mcmc_demo.sio) — Probabilistic MCMC module demo
- [`prob/normal_demo.sio`](prob/normal_demo.sio) — Normal distribution operations
- [`prob/beta_demo.sio`](prob/beta_demo.sio) — Beta distribution operations
- [`prob/inference_demo.sio`](prob/inference_demo.sio) — Statistical inference
- [`stats/descriptive_demo.sio`](stats/descriptive_demo.sio) — Descriptive statistics (mean, variance, quantiles)
- [`stats/inferential_demo.sio`](stats/inferential_demo.sio) — Inferential statistics (hypothesis tests)
- [`stats/effect_sizes_demo.sio`](stats/effect_sizes_demo.sio) — Effect sizes: Cohen's d, Cliff's delta
- [`stats/multiple_testing_demo.sio`](stats/multiple_testing_demo.sio) — Multiple testing correction (Bonferroni, FDR)
- [`stats/resampling_demo.sio`](stats/resampling_demo.sio) — Bootstrap and permutation resampling
- [`stats/validation_demo.sio`](stats/validation_demo.sio) — Statistical validation procedures

### Optimization

- [`optimize/bfgs_demo.sio`](optimize/bfgs_demo.sio) — BFGS quasi-Newton optimization
- [`optimize/nelder_mead_demo.sio`](optimize/nelder_mead_demo.sio) — Nelder-Mead simplex optimization
- [`optimize/differential_evolution_demo.sio`](optimize/differential_evolution_demo.sio) — Differential evolution (global optimizer)
- [`optimize/levenberg_marquardt_demo.sio`](optimize/levenberg_marquardt_demo.sio) — Levenberg-Marquardt nonlinear least squares
- [`optimize/uncertainty_demo.sio`](optimize/uncertainty_demo.sio) — Optimization under uncertainty

### Linear Algebra

- [`linalg/matrix_demo.sio`](linalg/matrix_demo.sio) — Matrix operations module demo
- [`linalg/vector_demo.sio`](linalg/vector_demo.sio) — Vector operations module demo

### ODE Solvers

- [`ode/rk4_demo.sio`](ode/rk4_demo.sio) — Classical Runge-Kutta 4th order
- [`ode/tsit5_demo.sio`](ode/tsit5_demo.sio) — Tsitouras 5(4) adaptive step solver
- [`ode/solver_demo.sio`](ode/solver_demo.sio) — General ODE solver interface
- [`showcase/ode_predator_prey.sio`](showcase/ode_predator_prey.sio) — Lotka-Volterra dynamics with adaptive RK45 and epistemic parameters
- [`science/lotka_volterra_ecosystem.sio`](science/lotka_volterra_ecosystem.sio) — Lotka-Volterra with RK4, oscillation detection

### Algorithms

- [`algorithms/mandelbrot.sio`](algorithms/mandelbrot.sio) — ASCII Mandelbrot set at 80x40 resolution
- [`algorithms/quicksort.sio`](algorithms/quicksort.sio) — In-place quicksort on 10,000 integers
- [`algorithms/radix_sort.sio`](algorithms/radix_sort.sio) — LSD radix sort (base-256) on 100,000 integers
- [`algorithms/sieve.sio`](algorithms/sieve.sio) — Sieve of Eratosthenes up to 100,000
- [`algorithms/matrix_mul.sio`](algorithms/matrix_mul.sio) — Dense 64x64 matrix multiplication (262K ops)
- [`algorithms/complex_native_demo.sio`](algorithms/complex_native_demo.sio) — Multi-algorithm demo: quicksort + sieve + GCD + Fibonacci + Collatz
- [`simulation/nbody.sio`](simulation/nbody.sio) — N-body gravitational simulation (16 bodies, 2D, 1000 steps)

### Information Theory and Signal Processing

- [`science/entropy_information.sio`](science/entropy_information.sio) — Shannon information theory (entropy, MI, KL divergence, channel capacity)
- [`science/signal_analysis.sio`](science/signal_analysis.sio) — DFT-based signal analysis with peak detection
- [`science/kalman_filter_tracking.sio`](science/kalman_filter_tracking.sio) — 1D Kalman filter for state estimation
- [`showcase/spectral_analyzer.sio`](showcase/spectral_analyzer.sio) — FFT signal analysis with epistemic confidence on peak identification

### Physics

- [`science/maxwell_cl13.sio`](science/maxwell_cl13.sio) — Maxwell's equations in Cl(1,3) spacetime algebra
- [`physics/cl13_lorentz.sio`](physics/cl13_lorentz.sio) — Lorentz rotors in Cl(1,3) even subalgebra
- [`real_world/06_climate_ensemble.sio`](real_world/06_climate_ensemble.sio) — Climate model ensemble uncertainty aggregation

### Fractal Analysis

- [`fractal/curvature_demo.sio`](fractal/curvature_demo.sio) — Fractal curvature estimation
- [`fractal/dimension_demo.sio`](fractal/dimension_demo.sio) — Fractal dimension computation
- [`fractal/entropy_demo.sio`](fractal/entropy_demo.sio) — Fractal entropy analysis
- [`fractal/lacunarity_demo.sio`](fractal/lacunarity_demo.sio) — Lacunarity (gap distribution in fractals)
- [`fractal/multifractal_demo.sio`](fractal/multifractal_demo.sio) — Multifractal spectrum analysis
- [`fractal/kec_demo.sio`](fractal/kec_demo.sio) — KEC (Kolmogorov-Entropy-Complexity) demo

### Grand Challenges

- [`grand_challenges/euler_line.sio`](grand_challenges/euler_line.sio) — Proves collinearity of circumcenter, centroid, and orthocenter
- [`grand_challenges/toxic_peak.sio`](grand_challenges/toxic_peak.sio) — Verifies drug brain concentration never exceeds safety threshold
- [`grand_challenges/turbulence.sio`](grand_challenges/turbulence.sio) — RANS incompressibility constraint verification

### Genomics

- [`showcase/genome_motif_scanner.sio`](showcase/genome_motif_scanner.sio) — DNA motif scanning with PWMs and epistemic hit scoring

### Random Number Generation

- [`random/rng_demo.sio`](random/rng_demo.sio) — Random number generator module
- [`random/distributions_demo.sio`](random/distributions_demo.sio) — Random distribution sampling
- [`random/sampling_demo.sio`](random/sampling_demo.sio) — Sampling strategies (reservoir, stratified, etc.)
