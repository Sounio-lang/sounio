# Validation Metrics Module for Sounio

## Overview

The metrics.d module provides regulatory-grade validation metrics for PBPK model assessment.

Location: /mnt/e/workspace/sounio/stdlib/darwin_pbpk/validation/metrics.d
Size: 736 lines, 20KB

## Core Data Structures

### ValidationResult
- n_samples: Number of samples
- gmfe: Geometric Mean Fold Error (FDA/EMA primary metric)
- afe: Average Fold Error
- aafe: Absolute Average Fold Error
- r_squared: Coefficient of Determination
- pct_within_2fold: Percentage within 2-fold
- pct_within_1_5fold: Percentage within 1.5-fold
- bias: Mean prediction bias (log scale)

### ValidationStats
Incremental accumulator (no arrays needed):
- n, sum_log_fe, sum_log_ratio, sum_abs_log_ratio
- count_2fold, count_1_5fold, ss_res, sum_obs

### GMFEInterval
- lower, upper, point_estimate

## Key Functions

Mathematical: ln, log10, exp, sqrt, abs, max, min, pow
Fold Error: fold_error, geometric_mean_fold_error, average_fold_error
Percentage: is_within_fold, percentage_within_fold
Regression: calculate_r_squared, calculate_rmse, calculate_bias
Regulatory: meets_fda_criteria, meets_ema_criteria, validation_grade
Incremental: init_validation_stats, update_validation_stats, compute_validation_result
Interpretation: interpret_gmfe, interpret_bias

## FDA/EMA Acceptance Criteria

GMFE <= 1.25 OR 80% within 2-fold

Grading Scale:
- A (90%+): Excellent
- B (80%+): Good, acceptable for regulatory filing
- C (70%+): Moderate, needs justification
- D (60%+): Poor
- F (<60%): Unacceptable

## References

1. FDA (2020) - PBPK Analyses Format and Content
2. EMA (2018) - PBPK Modelling and Simulation Guideline
3. Guest et al. (2011) - Drug Metab Dispos 39(2):170-173

Version 1.0.0 - December 2025
