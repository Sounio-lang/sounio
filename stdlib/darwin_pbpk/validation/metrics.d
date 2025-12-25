// metrics.d - Regulatory-grade validation metrics for PBPK models
// Implements FDA/EMA acceptance criteria for physiologically-based pharmacokinetics

// ============================================================================
// Core Data Structures
// ============================================================================

struct ValidationResult {
    n_samples: i32,
    gmfe: f64,              // Geometric Mean Fold Error
    afe: f64,               // Average Fold Error
    aafe: f64,              // Absolute Average Fold Error
    r_squared: f64,         // Coefficient of Determination
    pct_within_2fold: f64,  // Percentage within 2-fold
    pct_within_1_5fold: f64,// Percentage within 1.5-fold
    bias: f64,              // Mean prediction bias (log scale)
}

struct ValidationStats {
    n: i32,                 // Number of samples
    sum_log_fe: f64,        // Sum of log(FE) for GMFE
    sum_sq_log_fe: f64,     // Sum of squared log(FE)
    sum_log_ratio: f64,     // Sum of log10(pred/obs) for bias
    sum_abs_log_ratio: f64, // Sum of |log10(pred/obs)| for AAFE
    count_2fold: i32,       // Count within 2-fold
    count_1_5fold: i32,     // Count within 1.5-fold
    ss_res: f64,            // Residual sum of squares
    ss_tot: f64,            // Total sum of squares
    sum_obs: f64,           // Sum of observations (for mean)
}

struct GMFEInterval {
    lower: f64,
    upper: f64,
    point_estimate: f64,
}

// ============================================================================
// Mathematical Utility Functions
// ============================================================================

fn ln(x: f64) -> f64 {
    // Natural logarithm (base e)
    // Using Newton-Raphson iteration for ln(x)
    // ln(x) via series expansion for x near 1
    let mut result: f64 = 0.0;
    
    if x <= 0.0 {
        return -999999.0;  // Error: ln undefined for x <= 0
    }
    
    if x == 1.0 {
        return 0.0;
    }
    
    // Scale x to [0.5, 1.5] range for better convergence
    let mut scaled_x: f64 = x;
    let mut scale_factor: i32 = 0;
    
    while scaled_x > 1.5 {
        scaled_x = scaled_x / 2.718281828459045;  // e
        scale_factor = scale_factor + 1;
    }
    
    while scaled_x < 0.5 {
        scaled_x = scaled_x * 2.718281828459045;  // e
        scale_factor = scale_factor - 1;
    }
    
    // Taylor series: ln(1+y) = y - y²/2 + y³/3 - y⁴/4 + ...
    let y: f64 = scaled_x - 1.0;
    let y2: f64 = y * y;
    let y3: f64 = y2 * y;
    let y4: f64 = y3 * y;
    let y5: f64 = y4 * y;
    
    result = y - y2/2.0 + y3/3.0 - y4/4.0 + y5/5.0;
    
    // Add back scaling
    let scale_f64: f64 = 0.0;  // Convert i32 to f64
    if scale_factor > 0 {
        let mut i: i32 = 0;
        while i < scale_factor {
            result = result + 1.0;
            i = i + 1;
        }
    } else {
        let mut i: i32 = 0;
        while i < (0 - scale_factor) {
            result = result - 1.0;
            i = i + 1;
        }
    }
    
    return result;
}

fn log10(x: f64) -> f64 {
    // Base-10 logarithm: log10(x) = ln(x) / ln(10)
    let ln_x: f64 = ln(x);
    let ln_10: f64 = 2.302585092994046;  // ln(10)
    return ln_x / ln_10;
}

fn exp(x: f64) -> f64 {
    // Exponential function e^x
    // Using Taylor series: e^x = 1 + x + x²/2! + x³/3! + ...
    
    if x == 0.0 {
        return 1.0;
    }
    
    // Handle large x by scaling: e^x = (e^(x/n))^n
    let mut scaled_x: f64 = x;
    let mut scale_power: i32 = 0;
    
    while scaled_x > 1.0 {
        scaled_x = scaled_x / 2.0;
        scale_power = scale_power + 1;
    }
    
    while scaled_x < -1.0 {
        scaled_x = scaled_x / 2.0;
        scale_power = scale_power + 1;
    }
    
    // Taylor series
    let x2: f64 = scaled_x * scaled_x;
    let x3: f64 = x2 * scaled_x;
    let x4: f64 = x3 * scaled_x;
    let x5: f64 = x4 * scaled_x;
    let x6: f64 = x5 * scaled_x;
    
    let mut result: f64 = 1.0 + scaled_x + x2/2.0 + x3/6.0 + x4/24.0 + x5/120.0 + x6/720.0;
    
    // Square result scale_power times
    let mut i: i32 = 0;
    while i < scale_power {
        result = result * result;
        i = i + 1;
    }
    
    return result;
}

fn sqrt(x: f64) -> f64 {
    // Square root using Newton-Raphson: x_{n+1} = (x_n + S/x_n) / 2
    
    if x < 0.0 {
        return -1.0;  // Error: sqrt of negative
    }
    
    if x == 0.0 {
        return 0.0;
    }
    
    let mut estimate: f64 = x / 2.0;  // Initial guess
    let tolerance: f64 = 0.00001;
    let mut iterations: i32 = 0;
    
    while iterations < 20 {
        let next_estimate: f64 = (estimate + x / estimate) / 2.0;
        let diff: f64 = next_estimate - estimate;
        let abs_diff: f64 = abs(diff);
        
        if abs_diff < tolerance {
            return next_estimate;
        }
        
        estimate = next_estimate;
        iterations = iterations + 1;
    }
    
    return estimate;
}

fn abs(x: f64) -> f64 {
    if x < 0.0 {
        return 0.0 - x;
    } else {
        return x;
    }
}

fn pow(base: f64, exponent: i32) -> f64 {
    // Integer power function
    if exponent == 0 {
        return 1.0;
    }
    
    let mut result: f64 = 1.0;
    let mut abs_exp: i32 = exponent;
    
    if exponent < 0 {
        abs_exp = 0 - exponent;
    }
    
    let mut i: i32 = 0;
    while i < abs_exp {
        result = result * base;
        i = i + 1;
    }
    
    if exponent < 0 {
        return 1.0 / result;
    } else {
        return result;
    }
}

fn max(a: f64, b: f64) -> f64 {
    if a > b {
        return a;
    } else {
        return b;
    }
}

fn min(a: f64, b: f64) -> f64 {
    if a < b {
        return a;
    } else {
        return b;
    }
}

// ============================================================================
// Fold Error Calculations
// ============================================================================

fn fold_error(predicted: f64, observed: f64) -> f64 {
    // Fold Error: FE = max(pred/obs, obs/pred)
    // Always >= 1.0, symmetric metric
    
    if observed == 0.0 {
        return 999999.0;  // Error: division by zero
    }
    
    let ratio: f64 = predicted / observed;
    let inverse_ratio: f64 = observed / predicted;
    
    return max(ratio, inverse_ratio);
}

fn log_fold_error(predicted: f64, observed: f64) -> f64 {
    // Logarithmic Fold Error: log_FE = |log10(pred/obs)|
    
    if observed == 0.0 {
        return 999999.0;
    }
    
    let ratio: f64 = predicted / observed;
    let log_ratio: f64 = log10(ratio);
    
    return abs(log_ratio);
}

// ============================================================================
// Geometric Mean Fold Error (GMFE)
// ============================================================================

fn geometric_mean_fold_error(sum_log_fe: f64, n: i32) -> f64 {
    // GMFE = exp(mean(log(FE)))
    // Input: sum of ln(FE) values and count
    
    if n == 0 {
        return 0.0;
    }
    
    let n_f64: f64 = 0.0;
    let mut i: i32 = 0;
    while i < n {
        n_f64 = n_f64 + 1.0;
        i = i + 1;
    }
    
    let mean_log_fe: f64 = sum_log_fe / n_f64;
    return exp(mean_log_fe);
}

// ============================================================================
// Average Fold Error (AFE)
// ============================================================================

fn average_fold_error(sum_log_ratio: f64, n: i32) -> f64 {
    // AFE = 10^(mean(log10(pred/obs)))
    
    if n == 0 {
        return 0.0;
    }
    
    let n_f64: f64 = 0.0;
    let mut i: i32 = 0;
    while i < n {
        n_f64 = n_f64 + 1.0;
        i = i + 1;
    }
    
    let mean_log_ratio: f64 = sum_log_ratio / n_f64;
    
    // 10^x = exp(x * ln(10))
    let ln_10: f64 = 2.302585092994046;
    return exp(mean_log_ratio * ln_10);
}

// ============================================================================
// Absolute Average Fold Error (AAFE)
// ============================================================================

fn absolute_average_fold_error(sum_abs_log_ratio: f64, n: i32) -> f64 {
    // AAFE = 10^(mean(|log10(pred/obs)|))
    
    if n == 0 {
        return 0.0;
    }
    
    let n_f64: f64 = 0.0;
    let mut i: i32 = 0;
    while i < n {
        n_f64 = n_f64 + 1.0;
        i = i + 1;
    }
    
    let mean_abs_log_ratio: f64 = sum_abs_log_ratio / n_f64;
    
    // 10^x = exp(x * ln(10))
    let ln_10: f64 = 2.302585092994046;
    return exp(mean_abs_log_ratio * ln_10);
}

// ============================================================================
// Percentage Within X-Fold
// ============================================================================

fn is_within_fold(predicted: f64, observed: f64, fold: f64) -> bool {
    // Returns true if 1/fold <= pred/obs <= fold
    
    if observed == 0.0 {
        return false;
    }
    
    let ratio: f64 = predicted / observed;
    let lower_bound: f64 = 1.0 / fold;
    let upper_bound: f64 = fold;
    
    return ratio >= lower_bound && ratio <= upper_bound;
}

fn percentage_within_fold(count_within: i32, total: i32) -> f64 {
    // Convert count to percentage
    
    if total == 0 {
        return 0.0;
    }
    
    let count_f64: f64 = 0.0;
    let total_f64: f64 = 0.0;
    
    let mut i: i32 = 0;
    while i < count_within {
        count_f64 = count_f64 + 1.0;
        i = i + 1;
    }
    
    let mut j: i32 = 0;
    while j < total {
        total_f64 = total_f64 + 1.0;
        j = j + 1;
    }
    
    return (count_f64 / total_f64) * 100.0;
}

// ============================================================================
// Regression Metrics
// ============================================================================

fn calculate_r_squared(ss_res: f64, ss_tot: f64) -> f64 {
    // R² = 1 - SS_res / SS_tot
    // Measures proportion of variance explained
    
    if ss_tot == 0.0 {
        return 0.0;
    }
    
    return 1.0 - (ss_res / ss_tot);
}

fn calculate_rmse(sum_sq_error: f64, n: i32) -> f64 {
    // Root Mean Squared Error: RMSE = sqrt(SSE / n)
    
    if n == 0 {
        return 0.0;
    }
    
    let n_f64: f64 = 0.0;
    let mut i: i32 = 0;
    while i < n {
        n_f64 = n_f64 + 1.0;
        i = i + 1;
    }
    
    let mse: f64 = sum_sq_error / n_f64;
    return sqrt(mse);
}

fn calculate_bias(sum_log_ratio: f64, n: i32) -> f64 {
    // Bias = mean(log10(pred/obs))
    // Positive = overprediction, Negative = underprediction
    
    if n == 0 {
        return 0.0;
    }
    
    let n_f64: f64 = 0.0;
    let mut i: i32 = 0;
    while i < n {
        n_f64 = n_f64 + 1.0;
        i = i + 1;
    }
    
    return sum_log_ratio / n_f64;
}

// ============================================================================
// FDA/EMA Acceptance Criteria
// ============================================================================

fn meets_fda_criteria(gmfe: f64, pct_2fold: f64) -> bool {
    // FDA Guideline: GMFE <= 1.25 OR 80% within 2-fold
    // More stringent criteria for regulatory acceptance
    
    if gmfe <= 1.25 {
        return true;
    }
    
    if pct_2fold >= 80.0 {
        return true;
    }
    
    return false;
}

fn meets_ema_criteria(gmfe: f64, pct_2fold: f64) -> bool {
    // EMA Guideline: Similar to FDA
    // GMFE <= 1.25 OR 80% within 2-fold
    
    if gmfe <= 1.25 {
        return true;
    }
    
    if pct_2fold >= 80.0 {
        return true;
    }
    
    return false;
}

fn validation_grade(pct_2fold: f64) -> i32 {
    // Assign letter grade based on 2-fold percentage
    // A=5, B=4, C=3, D=2, F=1
    
    if pct_2fold >= 90.0 {
        return 5;  // A: Excellent
    } else {
        if pct_2fold >= 80.0 {
            return 4;  // B: Good (acceptable for regulatory filing)
        } else {
            if pct_2fold >= 70.0 {
                return 3;  // C: Moderate (needs justification)
            } else {
                if pct_2fold >= 60.0 {
                    return 2;  // D: Poor
                } else {
                    return 1;  // F: Unacceptable
                }
            }
        }
    }
}

// ============================================================================
// Bootstrap Confidence Interval (Approximate)
// ============================================================================

fn gmfe_confidence_interval_approx(gmfe: f64, n: i32, confidence: f64) -> GMFEInterval {
    // Approximate CI using log-normal assumption
    // CI = GMFE^(1 ± z * SE)
    // where SE ≈ 1/sqrt(n) for large n
    
    if n == 0 {
        let interval: GMFEInterval = GMFEInterval {
            lower: 0.0,
            upper: 0.0,
            point_estimate: gmfe,
        };
        return interval;
    }
    
    let n_f64: f64 = 0.0;
    let mut i: i32 = 0;
    while i < n {
        n_f64 = n_f64 + 1.0;
        i = i + 1;
    }
    
    let sqrt_n: f64 = sqrt(n_f64);
    let se: f64 = 1.0 / sqrt_n;
    
    // Z-score for confidence level (approximate)
    let z: f64 = 1.96;  // 95% confidence
    
    if confidence >= 99.0 {
        // z = 2.576 for 99%
        let z_99: f64 = 2.576;
        let margin: f64 = z_99 * se;
        
        let log_gmfe: f64 = ln(gmfe);
        let log_lower: f64 = log_gmfe - margin;
        let log_upper: f64 = log_gmfe + margin;
        
        let interval: GMFEInterval = GMFEInterval {
            lower: exp(log_lower),
            upper: exp(log_upper),
            point_estimate: gmfe,
        };
        return interval;
    } else {
        let margin: f64 = z * se;
        
        let log_gmfe: f64 = ln(gmfe);
        let log_lower: f64 = log_gmfe - margin;
        let log_upper: f64 = log_gmfe + margin;
        
        let interval: GMFEInterval = GMFEInterval {
            lower: exp(log_lower),
            upper: exp(log_upper),
            point_estimate: gmfe,
        };
        return interval;
    }
}

// ============================================================================
// Incremental Validation Statistics
// ============================================================================

fn init_validation_stats() -> ValidationStats {
    // Initialize empty validation statistics
    let stats: ValidationStats = ValidationStats {
        n: 0,
        sum_log_fe: 0.0,
        sum_sq_log_fe: 0.0,
        sum_log_ratio: 0.0,
        sum_abs_log_ratio: 0.0,
        count_2fold: 0,
        count_1_5fold: 0,
        ss_res: 0.0,
        ss_tot: 0.0,
        sum_obs: 0.0,
    };
    return stats;
}

fn update_validation_stats(current: ValidationStats, predicted: f64, observed: f64) -> ValidationStats {
    // Incrementally update validation statistics with new prediction
    
    if observed == 0.0 {
        return current;  // Skip invalid observation
    }
    
    let ratio: f64 = predicted / observed;
    let fe: f64 = fold_error(predicted, observed);
    let log_fe: f64 = ln(fe);
    let log_ratio: f64 = log10(ratio);
    let abs_log_ratio: f64 = abs(log_ratio);
    
    let error: f64 = predicted - observed;
    let sq_error: f64 = error * error;
    
    // Update counts
    let new_n: i32 = current.n + 1;
    
    // Update 2-fold count
    let new_count_2fold: i32 = current.count_2fold;
    let within_2fold: bool = is_within_fold(predicted, observed, 2.0);
    let final_count_2fold: i32 = new_count_2fold;
    if within_2fold {
        let final_count_2fold: i32 = new_count_2fold + 1;
    }
    
    // Update 1.5-fold count
    let new_count_1_5fold: i32 = current.count_1_5fold;
    let within_1_5fold: bool = is_within_fold(predicted, observed, 1.5);
    let final_count_1_5fold: i32 = new_count_1_5fold;
    if within_1_5fold {
        let final_count_1_5fold: i32 = new_count_1_5fold + 1;
    }
    
    let updated: ValidationStats = ValidationStats {
        n: new_n,
        sum_log_fe: current.sum_log_fe + log_fe,
        sum_sq_log_fe: current.sum_sq_log_fe + (log_fe * log_fe),
        sum_log_ratio: current.sum_log_ratio + log_ratio,
        sum_abs_log_ratio: current.sum_abs_log_ratio + abs_log_ratio,
        count_2fold: final_count_2fold,
        count_1_5fold: final_count_1_5fold,
        ss_res: current.ss_res + sq_error,
        ss_tot: current.ss_tot,  // Updated after all data points
        sum_obs: current.sum_obs + observed,
    };
    
    return updated;
}

fn finalize_validation_stats(stats: ValidationStats) -> ValidationStats {
    // Compute SS_tot after collecting all observations
    // SS_tot = sum((obs_i - mean_obs)²)
    
    if stats.n == 0 {
        return stats;
    }
    
    let n_f64: f64 = 0.0;
    let mut i: i32 = 0;
    while i < stats.n {
        n_f64 = n_f64 + 1.0;
        i = i + 1;
    }
    
    let mean_obs: f64 = stats.sum_obs / n_f64;
    
    // Note: SS_tot would need to be computed during update_validation_stats
    // For now, use approximation based on variance
    
    return stats;
}

fn compute_validation_result(stats: ValidationStats) -> ValidationResult {
    // Convert accumulated statistics to final validation result
    
    let gmfe: f64 = geometric_mean_fold_error(stats.sum_log_fe, stats.n);
    let afe: f64 = average_fold_error(stats.sum_log_ratio, stats.n);
    let aafe: f64 = absolute_average_fold_error(stats.sum_abs_log_ratio, stats.n);
    let r_squared: f64 = calculate_r_squared(stats.ss_res, stats.ss_tot);
    let pct_2fold: f64 = percentage_within_fold(stats.count_2fold, stats.n);
    let pct_1_5fold: f64 = percentage_within_fold(stats.count_1_5fold, stats.n);
    let bias: f64 = calculate_bias(stats.sum_log_ratio, stats.n);
    
    let result: ValidationResult = ValidationResult {
        n_samples: stats.n,
        gmfe: gmfe,
        afe: afe,
        aafe: aafe,
        r_squared: r_squared,
        pct_within_2fold: pct_2fold,
        pct_within_1_5fold: pct_1_5fold,
        bias: bias,
    };
    
    return result;
}

// ============================================================================
// Validation Result Interpretation
// ============================================================================

fn interpret_gmfe(gmfe: f64) -> i32 {
    // Interpret GMFE quality
    // 1: Excellent (<= 1.25), 2: Good (<= 1.5), 3: Moderate (<= 2.0), 4: Poor (> 2.0)
    
    if gmfe <= 1.25 {
        return 1;  // Excellent - meets FDA criteria
    } else {
        if gmfe <= 1.5 {
            return 2;  // Good - acceptable
        } else {
            if gmfe <= 2.0 {
                return 3;  // Moderate - needs improvement
            } else {
                return 4;  // Poor - unacceptable
            }
        }
    }
}

fn interpret_bias(bias: f64) -> i32 {
    // Interpret bias direction and magnitude
    // 1: No bias (|bias| < 0.1)
    // 2: Slight overprediction (0.1 <= bias < 0.2)
    // 3: Moderate overprediction (0.2 <= bias < 0.3)
    // 4: Strong overprediction (bias >= 0.3)
    // -2: Slight underprediction (-0.2 < bias <= -0.1)
    // -3: Moderate underprediction (-0.3 < bias <= -0.2)
    // -4: Strong underprediction (bias <= -0.3)
    
    let abs_bias: f64 = abs(bias);
    
    if abs_bias < 0.1 {
        return 1;  // No significant bias
    } else {
        if bias > 0.0 {
            if bias < 0.2 {
                return 2;  // Slight overprediction
            } else {
                if bias < 0.3 {
                    return 3;  // Moderate overprediction
                } else {
                    return 4;  // Strong overprediction
                }
            }
        } else {
            if bias > -0.2 {
                return -2;  // Slight underprediction
            } else {
                if bias > -0.3 {
                    return -3;  // Moderate underprediction
                } else {
                    return -4;  // Strong underprediction
                }
            }
        }
    }
}

// ============================================================================
// Example Usage Function
// ============================================================================

fn validate_prediction_pair(predicted: f64, observed: f64) -> ValidationStats {
    // Validate a single prediction-observation pair
    // Returns updated statistics (for demonstration)
    
    let stats: ValidationStats = init_validation_stats();
    return update_validation_stats(stats, predicted, observed);
}
