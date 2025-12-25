// stdlib/data/series.d
// Typed Series (Columns) with Full Epistemic Support
//
// Series are the building blocks of DataFrames - typed arrays
// with optional epistemic metadata (uncertainty + confidence).

extern "C" {
    fn sqrt(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

// ============================================================================
// UNCERTAINTY TYPE (inline for self-contained module)
// ============================================================================

pub struct Uncertainty {
    pub tag: i32,          // 0=Exact, 1=StdDev, 2=Interval
    pub std_u: f64,
    pub std_k: f64,
    pub interval_lo: f64,
    pub interval_hi: f64,
}

pub fn uncertainty_exact() -> Uncertainty {
    Uncertainty { tag: 0, std_u: 0.0, std_k: 1.0, interval_lo: 0.0, interval_hi: 0.0 }
}

pub fn uncertainty_std(u: f64) -> Uncertainty {
    Uncertainty { tag: 1, std_u: if u < 0.0 { 0.0 } else { u }, std_k: 1.0, interval_lo: 0.0, interval_hi: 0.0 }
}

pub fn uncertainty_interval(lo: f64, hi: f64) -> Uncertainty {
    let l = if lo < hi { lo } else { hi }
    let h = if lo < hi { hi } else { lo }
    Uncertainty { tag: 2, std_u: 0.0, std_k: 1.0, interval_lo: l, interval_hi: h }
}

pub fn uncert_to_std(u: Uncertainty) -> f64 {
    if u.tag == 0 { return 0.0 }
    if u.tag == 1 { return u.std_u }
    return (u.interval_hi - u.interval_lo) / 4.0
}

// ============================================================================
// EPISTEMIC VALUE TYPE
// ============================================================================

pub struct EpistemicValue {
    pub value: f64,
    pub uncert: Uncertainty,
    pub conf: f64,
    pub provenance_id: i64,
}

pub fn epistemic_std(value: f64, uncertainty: f64, confidence: f64) -> EpistemicValue {
    var c = confidence
    if c < 0.0 { c = 0.0 }
    if c > 1.0 { c = 1.0 }
    EpistemicValue {
        value: value,
        uncert: uncertainty_std(uncertainty),
        conf: c,
        provenance_id: 0,
    }
}

pub fn epistemic_exact(value: f64, confidence: f64) -> EpistemicValue {
    var c = confidence
    if c < 0.0 { c = 0.0 }
    if c > 1.0 { c = 1.0 }
    EpistemicValue {
        value: value,
        uncert: uncertainty_exact(),
        conf: c,
        provenance_id: 0,
    }
}

pub fn get_value(e: EpistemicValue) -> f64 { e.value }
pub fn get_conf(e: EpistemicValue) -> f64 { e.conf }
pub fn get_std_u(e: EpistemicValue) -> f64 { uncert_to_std(e.uncert) }

// ============================================================================
// FLOAT SERIES - Basic numeric column
// ============================================================================

pub struct FloatSeries {
    pub name: String,
    pub data: [f64],
}

pub fn float_series_new(name: String) -> FloatSeries {
    var empty: [f64] = []
    FloatSeries { name: name, data: empty }
}

pub fn float_series_from(name: String, data: [f64]) -> FloatSeries {
    FloatSeries { name: name, data: data }
}

pub fn float_series_len(s: FloatSeries) -> usize {
    s.data.len()
}

pub fn float_series_get(s: FloatSeries, idx: usize) -> f64 {
    if idx < s.data.len() { s.data[idx] } else { 0.0 }
}

pub fn float_series_push(s: FloatSeries, val: f64) -> FloatSeries {
    var d = s.data
    d.push(val)
    FloatSeries { name: s.name, data: d }
}

pub fn float_series_sum(s: FloatSeries) -> f64 {
    var total = 0.0
    var i: usize = 0
    while i < s.data.len() {
        total = total + s.data[i]
        i = i + 1
    }
    total
}

pub fn float_series_mean(s: FloatSeries) -> f64 {
    let n = s.data.len()
    if n == 0 { return 0.0 }
    float_series_sum(s) / (n as f64)
}

pub fn float_series_min(s: FloatSeries) -> f64 {
    if s.data.len() == 0 { return 0.0 }
    var m = s.data[0]
    var i: usize = 1
    while i < s.data.len() {
        if s.data[i] < m { m = s.data[i] }
        i = i + 1
    }
    m
}

pub fn float_series_max(s: FloatSeries) -> f64 {
    if s.data.len() == 0 { return 0.0 }
    var m = s.data[0]
    var i: usize = 1
    while i < s.data.len() {
        if s.data[i] > m { m = s.data[i] }
        i = i + 1
    }
    m
}

pub fn float_series_variance(s: FloatSeries) -> f64 {
    let n = s.data.len()
    if n < 2 { return 0.0 }
    let mean_val = float_series_mean(s)
    var sum_sq = 0.0
    var i: usize = 0
    while i < n {
        let diff = s.data[i] - mean_val
        sum_sq = sum_sq + diff * diff
        i = i + 1
    }
    sum_sq / ((n - 1) as f64)
}

pub fn float_series_std(s: FloatSeries) -> f64 {
    sqrt_f64(float_series_variance(s))
}

// ============================================================================
// INT SERIES - Integer column
// ============================================================================

pub struct IntSeries {
    pub name: String,
    pub data: [i64],
}

pub fn int_series_new(name: String) -> IntSeries {
    var empty: [i64] = []
    IntSeries { name: name, data: empty }
}

pub fn int_series_from(name: String, data: [i64]) -> IntSeries {
    IntSeries { name: name, data: data }
}

pub fn int_series_len(s: IntSeries) -> usize {
    s.data.len()
}

pub fn int_series_get(s: IntSeries, idx: usize) -> i64 {
    if idx < s.data.len() { s.data[idx] } else { 0 }
}

pub fn int_series_push(s: IntSeries, val: i64) -> IntSeries {
    var d = s.data
    d.push(val)
    IntSeries { name: s.name, data: d }
}

pub fn int_series_sum(s: IntSeries) -> i64 {
    var total: i64 = 0
    var i: usize = 0
    while i < s.data.len() {
        total = total + s.data[i]
        i = i + 1
    }
    total
}

pub fn int_series_to_float(s: IntSeries) -> FloatSeries {
    var fdata: [f64] = []
    var i: usize = 0
    while i < s.data.len() {
        fdata.push(s.data[i] as f64)
        i = i + 1
    }
    FloatSeries { name: s.name, data: fdata }
}

// ============================================================================
// STRING SERIES - Text column
// ============================================================================

pub struct StringSeries {
    pub name: String,
    pub data: [String],
}

pub fn string_series_new(name: String) -> StringSeries {
    var empty: [String] = []
    StringSeries { name: name, data: empty }
}

pub fn string_series_from(name: String, data: [String]) -> StringSeries {
    StringSeries { name: name, data: data }
}

pub fn string_series_len(s: StringSeries) -> usize {
    s.data.len()
}

pub fn string_series_get(s: StringSeries, idx: usize) -> String {
    if idx < s.data.len() { s.data[idx] } else { "" }
}

pub fn string_series_push(s: StringSeries, val: String) -> StringSeries {
    var d = s.data
    d.push(val)
    StringSeries { name: s.name, data: d }
}

// ============================================================================
// BOOL SERIES - Boolean column
// ============================================================================

pub struct BoolSeries {
    pub name: String,
    pub data: [bool],
}

pub fn bool_series_new(name: String) -> BoolSeries {
    var empty: [bool] = []
    BoolSeries { name: name, data: empty }
}

pub fn bool_series_from(name: String, data: [bool]) -> BoolSeries {
    BoolSeries { name: name, data: data }
}

pub fn bool_series_len(s: BoolSeries) -> usize {
    s.data.len()
}

pub fn bool_series_get(s: BoolSeries, idx: usize) -> bool {
    if idx < s.data.len() { s.data[idx] } else { false }
}

pub fn bool_series_count_true(s: BoolSeries) -> usize {
    var count: usize = 0
    var i: usize = 0
    while i < s.data.len() {
        if s.data[i] { count = count + 1 }
        i = i + 1
    }
    count
}

// ============================================================================
// EPISTEMIC SERIES - Numeric column with uncertainty tracking
// ============================================================================

pub struct EpistemicSeries {
    pub name: String,
    pub values: [f64],
    pub uncertainties: [f64],    // Standard uncertainty for each value
    pub confidences: [f64],      // Confidence for each value
}

pub fn epistemic_series_new(name: String) -> EpistemicSeries {
    var empty_v: [f64] = []
    var empty_u: [f64] = []
    var empty_c: [f64] = []
    EpistemicSeries {
        name: name,
        values: empty_v,
        uncertainties: empty_u,
        confidences: empty_c,
    }
}

pub fn epistemic_series_len(s: EpistemicSeries) -> usize {
    s.values.len()
}

pub fn epistemic_series_get(s: EpistemicSeries, idx: usize) -> EpistemicValue {
    if idx >= s.values.len() {
        return epistemic_exact(0.0, 0.0)
    }
    EpistemicValue {
        value: s.values[idx],
        uncert: uncertainty_std(s.uncertainties[idx]),
        conf: s.confidences[idx],
        provenance_id: 0,
    }
}

pub fn epistemic_series_push(s: EpistemicSeries, ev: EpistemicValue) -> EpistemicSeries {
    var vs = s.values
    var us = s.uncertainties
    var cs = s.confidences
    vs.push(ev.value)
    us.push(uncert_to_std(ev.uncert))
    cs.push(ev.conf)
    EpistemicSeries {
        name: s.name,
        values: vs,
        uncertainties: us,
        confidences: cs,
    }
}

pub fn epistemic_series_push_with_u(s: EpistemicSeries, val: f64, uncert: f64, conf: f64) -> EpistemicSeries {
    var vs = s.values
    var us = s.uncertainties
    var cs = s.confidences
    vs.push(val)
    us.push(uncert)
    cs.push(conf)
    EpistemicSeries {
        name: s.name,
        values: vs,
        uncertainties: us,
        confidences: cs,
    }
}

// Epistemic mean with proper uncertainty propagation
pub fn epistemic_series_mean(s: EpistemicSeries) -> EpistemicValue {
    let n = s.values.len()
    if n == 0 {
        return epistemic_exact(0.0, 0.0)
    }

    // Calculate mean value
    var sum_val = 0.0
    var i: usize = 0
    while i < n {
        sum_val = sum_val + s.values[i]
        i = i + 1
    }
    let mean_val = sum_val / (n as f64)

    // GUM: Combined uncertainty for mean of independent measurements
    // u(mean)^2 = (1/n^2) * sum(u_i^2)
    var sum_u_sq = 0.0
    i = 0
    while i < n {
        let ui = s.uncertainties[i]
        sum_u_sq = sum_u_sq + ui * ui
        i = i + 1
    }
    let mean_u = sqrt_f64(sum_u_sq) / (n as f64)

    // Confidence: minimum of all confidences
    var min_conf = 1.0
    i = 0
    while i < n {
        if s.confidences[i] < min_conf {
            min_conf = s.confidences[i]
        }
        i = i + 1
    }

    epistemic_std(mean_val, mean_u, min_conf)
}

// Epistemic sum with proper uncertainty propagation
pub fn epistemic_series_sum(s: EpistemicSeries) -> EpistemicValue {
    let n = s.values.len()
    if n == 0 {
        return epistemic_exact(0.0, 0.0)
    }

    var sum_val = 0.0
    var sum_u_sq = 0.0
    var min_conf = 1.0
    var i: usize = 0

    while i < n {
        sum_val = sum_val + s.values[i]
        let ui = s.uncertainties[i]
        sum_u_sq = sum_u_sq + ui * ui
        if s.confidences[i] < min_conf {
            min_conf = s.confidences[i]
        }
        i = i + 1
    }

    let combined_u = sqrt_f64(sum_u_sq)
    epistemic_std(sum_val, combined_u, min_conf)
}

// Epistemic standard deviation
pub fn epistemic_series_std(s: EpistemicSeries) -> EpistemicValue {
    let n = s.values.len()
    if n < 2 {
        return epistemic_exact(0.0, 0.0)
    }

    // Calculate sample mean
    var sum_val = 0.0
    var i: usize = 0
    while i < n {
        sum_val = sum_val + s.values[i]
        i = i + 1
    }
    let mean_val = sum_val / (n as f64)

    // Calculate sample variance
    var sum_sq = 0.0
    i = 0
    while i < n {
        let diff = s.values[i] - mean_val
        sum_sq = sum_sq + diff * diff
        i = i + 1
    }
    let variance = sum_sq / ((n - 1) as f64)
    let std_val = sqrt_f64(variance)

    // Uncertainty in std: approximate using bootstrap-like approach
    // u(s) ~ s / sqrt(2*(n-1))
    let std_u = std_val / sqrt_f64(2.0 * ((n - 1) as f64))

    // Minimum confidence
    var min_conf = 1.0
    i = 0
    while i < n {
        if s.confidences[i] < min_conf {
            min_conf = s.confidences[i]
        }
        i = i + 1
    }

    epistemic_std(std_val, std_u, min_conf)
}

// Epistemic min (value with minimum)
pub fn epistemic_series_min(s: EpistemicSeries) -> EpistemicValue {
    let n = s.values.len()
    if n == 0 {
        return epistemic_exact(0.0, 0.0)
    }

    var min_idx: usize = 0
    var min_val = s.values[0]
    var i: usize = 1
    while i < n {
        if s.values[i] < min_val {
            min_val = s.values[i]
            min_idx = i
        }
        i = i + 1
    }

    epistemic_std(min_val, s.uncertainties[min_idx], s.confidences[min_idx])
}

// Epistemic max (value with maximum)
pub fn epistemic_series_max(s: EpistemicSeries) -> EpistemicValue {
    let n = s.values.len()
    if n == 0 {
        return epistemic_exact(0.0, 0.0)
    }

    var max_idx: usize = 0
    var max_val = s.values[0]
    var i: usize = 1
    while i < n {
        if s.values[i] > max_val {
            max_val = s.values[i]
            max_idx = i
        }
        i = i + 1
    }

    epistemic_std(max_val, s.uncertainties[max_idx], s.confidences[max_idx])
}

// Convert FloatSeries to EpistemicSeries with uniform uncertainty
pub fn float_to_epistemic(s: FloatSeries, default_u: f64, default_conf: f64) -> EpistemicSeries {
    var es = epistemic_series_new(s.name)
    var i: usize = 0
    while i < s.data.len() {
        es = epistemic_series_push_with_u(es, s.data[i], default_u, default_conf)
        i = i + 1
    }
    es
}

// Extract just values from EpistemicSeries
pub fn epistemic_to_float(s: EpistemicSeries) -> FloatSeries {
    FloatSeries { name: s.name, data: s.values }
}

// ============================================================================
// SERIES OPERATIONS
// ============================================================================

// Element-wise addition of float series
pub fn float_series_add(a: FloatSeries, b: FloatSeries) -> FloatSeries {
    let n = if a.data.len() < b.data.len() { a.data.len() } else { b.data.len() }
    var result: [f64] = []
    var i: usize = 0
    while i < n {
        result.push(a.data[i] + b.data[i])
        i = i + 1
    }
    FloatSeries { name: a.name ++ "_plus_" ++ b.name, data: result }
}

// Element-wise multiplication of float series
pub fn float_series_mul(a: FloatSeries, b: FloatSeries) -> FloatSeries {
    let n = if a.data.len() < b.data.len() { a.data.len() } else { b.data.len() }
    var result: [f64] = []
    var i: usize = 0
    while i < n {
        result.push(a.data[i] * b.data[i])
        i = i + 1
    }
    FloatSeries { name: a.name ++ "_times_" ++ b.name, data: result }
}

// Scalar multiplication
pub fn float_series_scale(s: FloatSeries, scalar: f64) -> FloatSeries {
    var result: [f64] = []
    var i: usize = 0
    while i < s.data.len() {
        result.push(s.data[i] * scalar)
        i = i + 1
    }
    FloatSeries { name: s.name, data: result }
}

// Filter float series by boolean mask
pub fn float_series_filter(s: FloatSeries, mask: BoolSeries) -> FloatSeries {
    let n = if s.data.len() < mask.data.len() { s.data.len() } else { mask.data.len() }
    var result: [f64] = []
    var i: usize = 0
    while i < n {
        if mask.data[i] {
            result.push(s.data[i])
        }
        i = i + 1
    }
    FloatSeries { name: s.name, data: result }
}

// Compare float series element-wise (greater than)
pub fn float_series_gt(s: FloatSeries, threshold: f64) -> BoolSeries {
    var result: [bool] = []
    var i: usize = 0
    while i < s.data.len() {
        result.push(s.data[i] > threshold)
        i = i + 1
    }
    BoolSeries { name: s.name ++ "_gt", data: result }
}

// Compare float series element-wise (less than)
pub fn float_series_lt(s: FloatSeries, threshold: f64) -> BoolSeries {
    var result: [bool] = []
    var i: usize = 0
    while i < s.data.len() {
        result.push(s.data[i] < threshold)
        i = i + 1
    }
    BoolSeries { name: s.name ++ "_lt", data: result }
}

// ============================================================================
// TESTS
// ============================================================================

fn main() -> i32 {
    print("Testing Series module...\n")

    // Test FloatSeries
    var fs = float_series_new("weight")
    fs = float_series_push(fs, 70.5)
    fs = float_series_push(fs, 65.0)
    fs = float_series_push(fs, 80.2)

    let n = float_series_len(fs)
    if n != 3 { return 1 }

    let mean = float_series_mean(fs)
    if mean < 71.0 || mean > 72.5 { return 2 }

    let min_val = float_series_min(fs)
    if min_val < 64.9 || min_val > 65.1 { return 3 }

    let max_val = float_series_max(fs)
    if max_val < 80.1 || max_val > 80.3 { return 4 }

    print("FloatSeries: PASS\n")

    // Test IntSeries
    var is = int_series_new("count")
    is = int_series_push(is, 10)
    is = int_series_push(is, 20)
    is = int_series_push(is, 30)

    let sum_int = int_series_sum(is)
    if sum_int != 60 { return 5 }

    print("IntSeries: PASS\n")

    // Test StringSeries
    var ss = string_series_new("names")
    ss = string_series_push(ss, "Alice")
    ss = string_series_push(ss, "Bob")

    let name = string_series_get(ss, 0)
    if name != "Alice" { return 6 }

    print("StringSeries: PASS\n")

    // Test BoolSeries
    var bs = bool_series_new("flags")
    bs = BoolSeries { name: "flags", data: [true, false, true, true] }

    let count_t = bool_series_count_true(bs)
    if count_t != 3 { return 7 }

    print("BoolSeries: PASS\n")

    // Test EpistemicSeries
    var es = epistemic_series_new("measurement")
    es = epistemic_series_push_with_u(es, 100.0, 2.0, 0.95)
    es = epistemic_series_push_with_u(es, 102.0, 2.0, 0.90)
    es = epistemic_series_push_with_u(es, 98.0, 2.0, 0.92)

    let es_mean = epistemic_series_mean(es)
    let mv = get_value(es_mean)
    let mu = get_std_u(es_mean)
    let mc = get_conf(es_mean)

    if mv < 99.0 || mv > 101.0 { return 8 }
    if mu < 1.0 || mu > 1.5 { return 9 }
    if mc > 0.91 { return 10 }  // Should be min confidence (0.90)

    print("EpistemicSeries mean: PASS\n")

    // Test epistemic sum
    let es_sum = epistemic_series_sum(es)
    let sv = get_value(es_sum)
    let su = get_std_u(es_sum)

    if sv < 299.0 || sv > 301.0 { return 11 }
    if su < 3.0 || su > 4.0 { return 12 }  // sqrt(3 * 4) ~ 3.46

    print("EpistemicSeries sum: PASS\n")

    // Test filter
    let mask = float_series_gt(fs, 70.0)
    let filtered = float_series_filter(fs, mask)
    if float_series_len(filtered) != 2 { return 13 }

    print("Filter: PASS\n")

    // Test element-wise operations
    var fs1 = float_series_from("a", [1.0, 2.0, 3.0])
    var fs2 = float_series_from("b", [4.0, 5.0, 6.0])

    let added = float_series_add(fs1, fs2)
    if float_series_get(added, 0) < 4.9 || float_series_get(added, 0) > 5.1 { return 14 }

    let scaled = float_series_scale(fs1, 2.0)
    if float_series_get(scaled, 1) < 3.9 || float_series_get(scaled, 1) > 4.1 { return 15 }

    print("Element-wise ops: PASS\n")

    print("All Series tests PASSED\n")
    0
}
