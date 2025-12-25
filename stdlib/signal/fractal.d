// signal::fractal — Fractal Dimension and Complexity Analysis
//
// Nonlinear dynamics measures for biosignal complexity.
// Essential for computational psychiatry: altered complexity in psychiatric disorders.
//
// Methods:
// - Higuchi Fractal Dimension (HFD): Time-domain complexity
// - Detrended Fluctuation Analysis (DFA): Long-range correlations
// - Permutation Entropy: Ordinal pattern complexity
//
// References:
// - Higuchi (1988): "Approach to an irregular time series..."
// - Peng et al. (1994): "Mosaic organization of DNA nucleotides"
// - Bandt & Pompe (2002): "Permutation entropy: A natural complexity measure"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
}

// ============================================================================
// HIGUCHI FRACTAL DIMENSION
// ============================================================================

// Maximum k for Higuchi algorithm
fn MAX_K() -> i64 { 64 }

// Compute curve length for given k and m
fn higuchi_curve_length(signal: [f64; 1024], n: i64, k: i64, m: i64) -> f64 {
    // L_m(k) = (1/k) * sum_{i=1}^{floor((N-m)/k)} |X(m+i*k) - X(m+(i-1)*k)| * (N-1) / (floor((N-m)/k) * k)

    let n_terms = (n - m) / k
    if n_terms <= 0 {
        return 0.0
    }

    var sum = 0.0
    var i: i64 = 1
    while i <= n_terms {
        let idx1 = (m + i * k - 1) as usize  // -1 for 0-based indexing
        let idx2 = (m + (i - 1) * k - 1) as usize
        if idx1 < (n as usize) && idx2 < (n as usize) {
            let diff = signal[idx1] - signal[idx2]
            sum = sum + if diff < 0.0 { -diff } else { diff }
        }
        i = i + 1
    }

    let norm = ((n - 1) as f64) / ((n_terms * k) as f64)
    return sum * norm / (k as f64)
}

// Compute Higuchi Fractal Dimension
// signal: input time series
// n: signal length
// k_max: maximum k value (typically 8-64)
// Returns: (HFD, R² of linear fit)
fn higuchi_fd(signal: [f64; 1024], n: i64, k_max: i64) -> (f64, f64) {
    if n < 10 || k_max < 2 {
        return (0.0, 0.0)
    }

    var log_k: [f64; 64] = [0.0; 64]
    var log_L: [f64; 64] = [0.0; 64]
    var n_points: i64 = 0

    var k: i64 = 1
    while k <= k_max && k < n / 4 {
        // Average L_m(k) over m = 1..k
        var L_avg = 0.0
        var m: i64 = 1
        while m <= k {
            L_avg = L_avg + higuchi_curve_length(signal, n, k, m)
            m = m + 1
        }
        L_avg = L_avg / (k as f64)

        if L_avg > 0.0 {
            log_k[n_points as usize] = log(k as f64)
            log_L[n_points as usize] = log(L_avg)
            n_points = n_points + 1
        }

        k = k + 1
    }

    if n_points < 3 {
        return (0.0, 0.0)
    }

    // Linear regression: log(L) = -D * log(k) + c
    // D = -slope
    let fit = linear_regression(log_k, log_L, n_points)

    return (-fit.0, fit.2)  // (-slope = D, R²)
}

// ============================================================================
// DETRENDED FLUCTUATION ANALYSIS (DFA)
// ============================================================================

// DFA configuration
struct DFAConfig {
    min_box: i64,       // Minimum box size
    max_box: i64,       // Maximum box size
    n_boxes: i64,       // Number of box sizes
}

fn dfa_config_default() -> DFAConfig {
    DFAConfig {
        min_box: 4,
        max_box: 256,
        n_boxes: 20,
    }
}

// Compute integrated signal (cumulative sum of deviations from mean)
fn dfa_integrate(signal: [f64; 1024], n: i64) -> [f64; 1024] {
    var integrated: [f64; 1024] = [0.0; 1024]

    // Mean
    var sum = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + signal[i as usize]
        i = i + 1
    }
    let mean = sum / (n as f64)

    // Cumulative sum
    var cumsum = 0.0
    i = 0
    while i < n {
        cumsum = cumsum + (signal[i as usize] - mean)
        integrated[i as usize] = cumsum
        i = i + 1
    }

    return integrated
}

// Linear detrending within a box, returns variance of residuals
fn detrend_box_variance(data: [f64; 1024], start: i64, box_size: i64) -> f64 {
    // Fit linear trend: y = a + b*x
    var sum_x = 0.0
    var sum_y = 0.0
    var sum_xx = 0.0
    var sum_xy = 0.0

    var i: i64 = 0
    while i < box_size {
        let x = i as f64
        let y = data[(start + i) as usize]
        sum_x = sum_x + x
        sum_y = sum_y + y
        sum_xx = sum_xx + x * x
        sum_xy = sum_xy + x * y
        i = i + 1
    }

    let nf = box_size as f64
    let denom = nf * sum_xx - sum_x * sum_x

    if denom == 0.0 {
        return 0.0
    }

    let b = (nf * sum_xy - sum_x * sum_y) / denom
    let a = (sum_y - b * sum_x) / nf

    // Compute variance of residuals
    var sum_sq = 0.0
    i = 0
    while i < box_size {
        let trend = a + b * (i as f64)
        let residual = data[(start + i) as usize] - trend
        sum_sq = sum_sq + residual * residual
        i = i + 1
    }

    return sum_sq / nf
}

// Compute fluctuation for given box size
fn dfa_fluctuation(integrated: [f64; 1024], n: i64, box_size: i64) -> f64 {
    let n_boxes = n / box_size
    if n_boxes == 0 {
        return 0.0
    }

    var total_var = 0.0
    var b: i64 = 0
    while b < n_boxes {
        let start = b * box_size
        total_var = total_var + detrend_box_variance(integrated, start, box_size)
        b = b + 1
    }

    return sqrt(total_var / (n_boxes as f64))
}

// Compute DFA scaling exponent
// Returns: (alpha, R²)
fn dfa(signal: [f64; 1024], n: i64, config: DFAConfig) -> (f64, f64) {
    if n < config.max_box {
        return (0.0, 0.0)
    }

    // Integrate signal
    let integrated = dfa_integrate(signal, n)

    // Compute fluctuations for different box sizes (log-spaced)
    var log_n: [f64; 64] = [0.0; 64]
    var log_F: [f64; 64] = [0.0; 64]
    var n_points: i64 = 0

    let log_min = log(config.min_box as f64)
    let log_max = log(config.max_box as f64)
    let log_step = (log_max - log_min) / ((config.n_boxes - 1) as f64)

    var i: i64 = 0
    while i < config.n_boxes {
        let box_size = pow(2.718281828, log_min + (i as f64) * log_step) as i64

        if box_size >= 4 && box_size <= n / 4 {
            let F = dfa_fluctuation(integrated, n, box_size)

            if F > 0.0 {
                log_n[n_points as usize] = log(box_size as f64)
                log_F[n_points as usize] = log(F)
                n_points = n_points + 1
            }
        }
        i = i + 1
    }

    if n_points < 3 {
        return (0.0, 0.0)
    }

    // Linear regression: log(F) = alpha * log(n) + c
    let fit = linear_regression(log_n, log_F, n_points)

    return (fit.0, fit.2)  // (slope = alpha, R²)
}

// Interpret DFA alpha value
fn dfa_interpret(alpha: f64) -> i32 {
    // 0 = white noise
    // 1 = pink noise / 1/f
    // 2 = Brownian motion
    // 3 = anti-correlated
    if alpha < 0.5 { 3 }        // Anti-correlated
    else if alpha < 0.65 { 0 }  // White noise
    else if alpha < 1.15 { 1 }  // Pink noise / 1/f
    else { 2 }                  // Brownian / random walk
}

// ============================================================================
// PERMUTATION ENTROPY
// ============================================================================

// Factorial (up to 7! = 5040)
fn factorial(n: i64) -> i64 {
    if n <= 1 { 1 }
    else { n * factorial(n - 1) }
}

// Encode permutation pattern to index (0 to m!-1)
fn encode_permutation(pattern: [i64; 7], m: i64) -> i64 {
    // Lehmer code
    var code: i64 = 0
    var fact = factorial(m - 1)

    var i: i64 = 0
    while i < m - 1 {
        var count: i64 = 0
        var j = i + 1
        while j < m {
            if pattern[j as usize] < pattern[i as usize] {
                count = count + 1
            }
            j = j + 1
        }
        code = code + count * fact
        if m - 1 - i > 0 {
            fact = fact / (m - 1 - i)
        }
        i = i + 1
    }
    return code
}

// Compute Permutation Entropy
// m: embedding dimension (typically 3-7)
// tau: time delay (typically 1)
fn permutation_entropy(signal: [f64; 1024], n: i64, m: i64, tau: i64) -> f64 {
    if m > 7 || n < (m - 1) * tau + 1 {
        return 0.0
    }

    let n_patterns = factorial(m)
    var counts: [i64; 5040] = [0; 5040]  // Max 7! patterns
    var total: i64 = 0

    // Extract patterns and count
    var i: i64 = 0
    while i <= n - (m - 1) * tau - 1 {
        // Get values for this pattern
        var values: [f64; 7] = [0.0; 7]
        var indices: [i64; 7] = [0; 7]
        var j: i64 = 0
        while j < m {
            values[j as usize] = signal[(i + j * tau) as usize]
            indices[j as usize] = j
            j = j + 1
        }

        // Sort indices by values (simple bubble sort for small m)
        j = 0
        while j < m - 1 {
            var k = j + 1
            while k < m {
                if values[indices[k as usize] as usize] < values[indices[j as usize] as usize] {
                    let tmp = indices[j as usize]
                    indices[j as usize] = indices[k as usize]
                    indices[k as usize] = tmp
                }
                k = k + 1
            }
            j = j + 1
        }

        // Encode and count
        let code = encode_permutation(indices, m)
        counts[code as usize] = counts[code as usize] + 1
        total = total + 1

        i = i + 1
    }

    // Compute entropy
    var entropy = 0.0
    var j: i64 = 0
    while j < n_patterns {
        if counts[j as usize] > 0 {
            let p = (counts[j as usize] as f64) / (total as f64)
            entropy = entropy - p * log(p)
        }
        j = j + 1
    }

    // Normalize by max entropy
    let max_entropy = log(n_patterns as f64)
    if max_entropy > 0.0 {
        return entropy / max_entropy
    } else {
        return 0.0
    }
}

// ============================================================================
// LINEAR REGRESSION HELPER
// ============================================================================

// Simple linear regression: y = a + b*x
// Returns: (slope, intercept, R²)
fn linear_regression(x: [f64; 64], y: [f64; 64], n: i64) -> (f64, f64, f64) {
    var sum_x = 0.0
    var sum_y = 0.0
    var sum_xx = 0.0
    var sum_xy = 0.0
    var sum_yy = 0.0

    var i: i64 = 0
    while i < n {
        sum_x = sum_x + x[i as usize]
        sum_y = sum_y + y[i as usize]
        sum_xx = sum_xx + x[i as usize] * x[i as usize]
        sum_xy = sum_xy + x[i as usize] * y[i as usize]
        sum_yy = sum_yy + y[i as usize] * y[i as usize]
        i = i + 1
    }

    let nf = n as f64
    let denom = nf * sum_xx - sum_x * sum_x

    if denom == 0.0 {
        return (0.0, 0.0, 0.0)
    }

    let slope = (nf * sum_xy - sum_x * sum_y) / denom
    let intercept = (sum_y - slope * sum_x) / nf

    // R²
    let ss_tot = sum_yy - sum_y * sum_y / nf
    let ss_res = sum_yy - slope * sum_xy - intercept * sum_y

    let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 }

    return (slope, intercept, r_squared)
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn test_higuchi_known_signal() -> bool {
    // Generate Brownian motion (HFD ≈ 1.5)
    var signal: [f64; 1024] = [0.0; 1024]
    var rng: i64 = 42
    var walk = 0.0

    var i: i64 = 0
    while i < 512 {
        // Simple random walk
        rng = (rng * 1103515245 + 12345) % 2147483648
        let step = ((rng as f64) / 2147483648.0 - 0.5) * 2.0
        walk = walk + step
        signal[i as usize] = walk
        i = i + 1
    }

    let result = higuchi_fd(signal, 512, 16)
    let hfd = result.0
    let r2 = result.1

    // Brownian motion should have HFD around 1.5
    // Allow wide tolerance for short signal
    return hfd > 1.0 && hfd < 2.0 && r2 > 0.8
}

fn test_dfa_white_noise() -> bool {
    // White noise should have alpha ≈ 0.5
    var signal: [f64; 1024] = [0.0; 1024]
    var rng: i64 = 123

    var i: i64 = 0
    while i < 1024 {
        rng = (rng * 1103515245 + 12345) % 2147483648
        signal[i as usize] = (rng as f64) / 2147483648.0 - 0.5
        i = i + 1
    }

    let config = dfa_config_default()
    let result = dfa(signal, 1024, config)
    let alpha = result.0

    // White noise: alpha ≈ 0.5
    return alpha > 0.3 && alpha < 0.7
}

fn test_permutation_entropy() -> bool {
    // Monotonic signal should have low PE
    var monotonic: [f64; 1024] = [0.0; 1024]
    var i: i64 = 0
    while i < 100 {
        monotonic[i as usize] = i as f64
        i = i + 1
    }

    let pe_mono = permutation_entropy(monotonic, 100, 3, 1)

    // Random signal should have PE ≈ 1
    var random: [f64; 1024] = [0.0; 1024]
    var rng: i64 = 789
    i = 0
    while i < 100 {
        rng = (rng * 1103515245 + 12345) % 2147483648
        random[i as usize] = (rng as f64) / 2147483648.0
        i = i + 1
    }

    let pe_random = permutation_entropy(random, 100, 3, 1)

    return pe_mono < 0.5 && pe_random > 0.8
}

fn main() -> i32 {
    print("Testing signal::fractal module...\n")

    if !test_higuchi_known_signal() {
        print("FAIL: higuchi_known_signal\n")
        return 1
    }
    print("PASS: higuchi_known_signal\n")

    if !test_dfa_white_noise() {
        print("FAIL: dfa_white_noise\n")
        return 2
    }
    print("PASS: dfa_white_noise\n")

    if !test_permutation_entropy() {
        print("FAIL: permutation_entropy\n")
        return 3
    }
    print("PASS: permutation_entropy\n")

    print("All signal::fractal tests PASSED\n")
    0
}
