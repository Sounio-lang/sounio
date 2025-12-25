//! Calibration Algorithms for Quantization
//!
//! Provides algorithms to determine optimal quantization parameters:
//! - MinMax: Track min/max values (simplest, may be sensitive to outliers)
//! - Histogram: Build histogram and find optimal clipping threshold
//! - Entropy: Minimize KL divergence between original and quantized distributions
//! - Percentile: Clip outliers based on percentile thresholds
//!
//! # Calibration Workflow
//!
//! ```text
//! 1. Create CalibrationCollector with chosen method
//! 2. Feed representative data samples via collect()
//! 3. Call compute_params() to get optimal QuantParams
//! ```

use super::quantize::{QuantDtype, QuantParams, QuantScheme};

// ============================================================================
// Calibration Methods
// ============================================================================

/// Calibration algorithm selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationMethod {
    /// Track min/max values (simple, may be sensitive to outliers)
    MinMax,

    /// Build histogram and find optimal clipping threshold
    Histogram {
        /// Number of histogram bins
        num_bins: u32,
    },

    /// Minimize KL divergence (entropy) between original and quantized distributions
    Entropy {
        /// Number of histogram bins for distribution
        num_bins: u32,
        /// Number of threshold candidates to try
        num_quantiles: u32,
    },

    /// Percentile-based clipping (robust to outliers)
    Percentile {
        /// Lower percentile (e.g., 0.001 for 0.1%)
        lower: f32,
        /// Upper percentile (e.g., 0.999 for 99.9%)
        upper: f32,
    },

    /// Mean squared error minimization
    Mse {
        /// Number of threshold candidates to try
        num_candidates: u32,
    },
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        CalibrationMethod::Histogram { num_bins: 2048 }
    }
}

// ============================================================================
// Calibration Statistics
// ============================================================================

/// Statistics collected during calibration
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    /// Observed minimum value
    pub min_val: f32,
    /// Observed maximum value
    pub max_val: f32,
    /// Histogram bins (if using histogram method)
    pub histogram: Option<Vec<u64>>,
    /// Histogram bin edges
    pub bin_edges: Option<Vec<f32>>,
    /// Running sum for mean calculation
    pub sum: f64,
    /// Running sum of squares for variance
    pub sum_sq: f64,
    /// Number of samples collected
    pub count: usize,
    /// All collected values (for percentile, limited size)
    collected_values: Option<Vec<f32>>,
    /// Max values to store for percentile calculation
    max_stored_values: usize,
}

impl CalibrationStats {
    /// Create new statistics tracker
    pub fn new(method: &CalibrationMethod) -> Self {
        let (histogram, bin_edges) = match method {
            CalibrationMethod::Histogram { num_bins }
            | CalibrationMethod::Entropy { num_bins, .. } => {
                (Some(vec![0u64; *num_bins as usize]), None)
            }
            _ => (None, None),
        };

        let collected_values = match method {
            CalibrationMethod::Percentile { .. } | CalibrationMethod::Mse { .. } => {
                Some(Vec::with_capacity(100_000))
            }
            _ => None,
        };

        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            histogram,
            bin_edges,
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
            collected_values,
            max_stored_values: 1_000_000,
        }
    }

    /// Get the mean of collected values
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Get the variance of collected values
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            let mean = self.mean();
            self.sum_sq / self.count as f64 - mean * mean
        }
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

// ============================================================================
// Calibration Collector
// ============================================================================

/// Collector for calibration data
pub struct CalibrationCollector {
    /// Calibration method
    method: CalibrationMethod,
    /// Collected statistics
    stats: CalibrationStats,
    /// Whether bin edges have been initialized
    edges_initialized: bool,
}

impl CalibrationCollector {
    /// Create a new calibration collector
    pub fn new(method: CalibrationMethod) -> Self {
        Self {
            stats: CalibrationStats::new(&method),
            method,
            edges_initialized: false,
        }
    }

    /// Collect a batch of values for calibration
    pub fn collect(&mut self, values: &[f32]) {
        for &val in values {
            // Skip NaN/Inf
            if !val.is_finite() {
                continue;
            }

            // Update min/max
            self.stats.min_val = self.stats.min_val.min(val);
            self.stats.max_val = self.stats.max_val.max(val);

            // Update running stats
            self.stats.sum += val as f64;
            self.stats.sum_sq += (val as f64).powi(2);
            self.stats.count += 1;

            // Store values for percentile/MSE methods
            if let Some(ref mut collected) = self.stats.collected_values
                && collected.len() < self.stats.max_stored_values
            {
                collected.push(val);
            }
        }

        // Initialize histogram bin edges after first batch
        if !self.edges_initialized && self.stats.count > 0 {
            self.initialize_histogram_edges();
            self.edges_initialized = true;
        }

        // Update histogram
        self.update_histogram(values);
    }

    /// Initialize histogram bin edges based on observed range
    fn initialize_histogram_edges(&mut self) {
        if let Some(ref histogram) = self.stats.histogram {
            let num_bins = histogram.len();
            let range = self.stats.max_val - self.stats.min_val;
            let bin_width = if range == 0.0 {
                1.0
            } else {
                range / num_bins as f32
            };

            let mut edges = Vec::with_capacity(num_bins + 1);
            for i in 0..=num_bins {
                edges.push(self.stats.min_val + i as f32 * bin_width);
            }
            self.stats.bin_edges = Some(edges);
        }
    }

    /// Update histogram with new values
    fn update_histogram(&mut self, values: &[f32]) {
        if let (Some(histogram), Some(edges)) = (&mut self.stats.histogram, &self.stats.bin_edges) {
            let num_bins = histogram.len();
            for &val in values {
                if !val.is_finite() {
                    continue;
                }
                // Find bin index
                let idx = if val <= edges[0] {
                    0
                } else if val >= edges[num_bins] {
                    num_bins - 1
                } else {
                    let bin_width = (edges[num_bins] - edges[0]) / num_bins as f32;
                    ((val - edges[0]) / bin_width).floor() as usize
                };
                histogram[idx.min(num_bins - 1)] += 1;
            }
        }
    }

    /// Compute optimal quantization parameters from collected data
    pub fn compute_params(&self, dtype: QuantDtype, scheme: QuantScheme) -> QuantParams {
        if self.stats.count == 0 {
            return QuantParams::symmetric_int8(1.0);
        }

        match self.method {
            CalibrationMethod::MinMax => self.calibrate_minmax(dtype, scheme),
            CalibrationMethod::Histogram { .. } => self.calibrate_histogram(dtype, scheme),
            CalibrationMethod::Entropy { num_quantiles, .. } => {
                self.calibrate_entropy(dtype, scheme, num_quantiles)
            }
            CalibrationMethod::Percentile { lower, upper } => {
                self.calibrate_percentile(dtype, scheme, lower, upper)
            }
            CalibrationMethod::Mse { num_candidates } => {
                self.calibrate_mse(dtype, scheme, num_candidates)
            }
        }
    }

    /// MinMax calibration: use observed min/max directly
    fn calibrate_minmax(&self, dtype: QuantDtype, scheme: QuantScheme) -> QuantParams {
        match scheme {
            QuantScheme::Symmetric | QuantScheme::PerTensor => {
                QuantParams::from_minmax_symmetric(self.stats.min_val, self.stats.max_val, dtype)
            }
            QuantScheme::Asymmetric => {
                QuantParams::from_minmax_asymmetric(self.stats.min_val, self.stats.max_val, dtype)
            }
            QuantScheme::PerChannel { .. } | QuantScheme::PerGroup { .. } => {
                // Per-channel handled separately
                QuantParams::from_minmax_symmetric(self.stats.min_val, self.stats.max_val, dtype)
            }
        }
    }

    /// Histogram calibration: find optimal clipping threshold
    fn calibrate_histogram(&self, dtype: QuantDtype, scheme: QuantScheme) -> QuantParams {
        let (histogram, edges) = match (&self.stats.histogram, &self.stats.bin_edges) {
            (Some(h), Some(e)) => (h, e),
            _ => return self.calibrate_minmax(dtype, scheme),
        };

        // Find optimal threshold by minimizing quantization error
        let num_bins = histogram.len();
        let mut best_threshold = self.stats.max_val.abs().max(self.stats.min_val.abs());
        let mut best_error = f64::MAX;

        // Try different thresholds (from 50% to 100% of the range)
        for threshold_pct in 50..=100 {
            let threshold_bin = (num_bins * threshold_pct) / 100;
            let threshold = if threshold_bin < edges.len() {
                edges[threshold_bin]
                    .abs()
                    .max(edges[num_bins - threshold_bin].abs())
            } else {
                continue;
            };

            let error = self.compute_quantization_error(histogram, edges, threshold, dtype);
            if error < best_error {
                best_error = error;
                best_threshold = threshold;
            }
        }

        // Create params with optimal threshold
        match scheme {
            QuantScheme::Symmetric | QuantScheme::PerTensor => {
                QuantParams::from_minmax_symmetric(-best_threshold, best_threshold, dtype)
            }
            QuantScheme::Asymmetric => {
                let min_clipped = self.stats.min_val.max(-best_threshold);
                let max_clipped = self.stats.max_val.min(best_threshold);
                QuantParams::from_minmax_asymmetric(min_clipped, max_clipped, dtype)
            }
            _ => QuantParams::from_minmax_symmetric(-best_threshold, best_threshold, dtype),
        }
    }

    /// Compute quantization error for a given threshold
    fn compute_quantization_error(
        &self,
        histogram: &[u64],
        edges: &[f32],
        threshold: f32,
        dtype: QuantDtype,
    ) -> f64 {
        let (qmin, qmax) = dtype.range();
        let scale = (2.0 * threshold) / (qmax - qmin) as f32;

        let mut error = 0.0;
        for (i, &count) in histogram.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let bin_center = (edges[i] + edges[i + 1]) / 2.0;

            // Quantize and dequantize
            let clipped = bin_center.clamp(-threshold, threshold);
            let quantized = (clipped / scale).round() as i32;
            let dequantized = quantized as f32 * scale;

            // Accumulate squared error
            let err = (bin_center - dequantized) as f64;
            error += err * err * count as f64;
        }

        error
    }

    /// Entropy (KL divergence) calibration
    fn calibrate_entropy(
        &self,
        dtype: QuantDtype,
        scheme: QuantScheme,
        num_quantiles: u32,
    ) -> QuantParams {
        let (histogram, edges) = match (&self.stats.histogram, &self.stats.bin_edges) {
            (Some(h), Some(e)) => (h, e),
            _ => return self.calibrate_minmax(dtype, scheme),
        };

        // Normalize histogram to probability distribution
        let total: u64 = histogram.iter().sum();
        if total == 0 {
            return self.calibrate_minmax(dtype, scheme);
        }

        let p: Vec<f64> = histogram.iter().map(|&c| c as f64 / total as f64).collect();

        // Try different thresholds and find one with minimum KL divergence
        let num_bins = histogram.len();
        let mut best_threshold = self.stats.max_val.abs().max(self.stats.min_val.abs());
        let mut min_kl = f64::MAX;

        for q in 1..=num_quantiles {
            let percentile = 0.5 + 0.5 * (q as f32 / num_quantiles as f32);
            let threshold_bin = (num_bins as f32 * percentile) as usize;

            if threshold_bin >= edges.len() {
                continue;
            }

            let threshold = edges[threshold_bin].abs();

            // Compute quantized distribution
            let q_dist = self.compute_quantized_distribution(&p, edges, threshold, dtype);

            // Compute KL divergence
            let kl = Self::kl_divergence(&p, &q_dist);

            if kl < min_kl {
                min_kl = kl;
                best_threshold = threshold;
            }
        }

        match scheme {
            QuantScheme::Symmetric | QuantScheme::PerTensor => {
                QuantParams::from_minmax_symmetric(-best_threshold, best_threshold, dtype)
            }
            _ => QuantParams::from_minmax_symmetric(-best_threshold, best_threshold, dtype),
        }
    }

    /// Compute quantized distribution for KL divergence
    fn compute_quantized_distribution(
        &self,
        p: &[f64],
        edges: &[f32],
        threshold: f32,
        dtype: QuantDtype,
    ) -> Vec<f64> {
        let num_bins = p.len();
        let (qmin, qmax) = dtype.range();
        let num_quant_bins = (qmax - qmin + 1) as usize;
        let scale = (2.0 * threshold) / (qmax - qmin) as f32;

        let mut q_dist = vec![0.0; num_bins];

        for (i, &prob) in p.iter().enumerate() {
            if prob == 0.0 {
                continue;
            }

            let bin_center = (edges[i] + edges[i + 1]) / 2.0;
            let clipped = bin_center.clamp(-threshold, threshold);
            let quantized = (clipped / scale).round() as i32;
            let dequantized = quantized as f32 * scale;

            // Map dequantized value back to histogram bin
            let target_bin = if dequantized <= edges[0] {
                0
            } else if dequantized >= edges[num_bins] {
                num_bins - 1
            } else {
                let bin_width = (edges[num_bins] - edges[0]) / num_bins as f32;
                ((dequantized - edges[0]) / bin_width).floor() as usize
            };

            q_dist[target_bin.min(num_bins - 1)] += prob;
        }

        q_dist
    }

    /// Compute KL divergence between two distributions
    fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
        let epsilon = 1e-10;
        let mut kl = 0.0;

        for (i, &pi) in p.iter().enumerate() {
            if pi > epsilon {
                let qi = q[i].max(epsilon);
                kl += pi * (pi / qi).ln();
            }
        }

        kl
    }

    /// Percentile calibration: clip outliers
    fn calibrate_percentile(
        &self,
        dtype: QuantDtype,
        scheme: QuantScheme,
        lower: f32,
        upper: f32,
    ) -> QuantParams {
        let values = match &self.stats.collected_values {
            Some(v) if !v.is_empty() => v,
            _ => return self.calibrate_minmax(dtype, scheme),
        };

        // Sort values
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Find percentile values
        let lower_idx = ((sorted.len() as f32 * lower) as usize).min(sorted.len() - 1);
        let upper_idx = ((sorted.len() as f32 * upper) as usize).min(sorted.len() - 1);

        let min_val = sorted[lower_idx];
        let max_val = sorted[upper_idx];

        match scheme {
            QuantScheme::Symmetric | QuantScheme::PerTensor => {
                QuantParams::from_minmax_symmetric(min_val, max_val, dtype)
            }
            QuantScheme::Asymmetric => QuantParams::from_minmax_asymmetric(min_val, max_val, dtype),
            _ => QuantParams::from_minmax_symmetric(min_val, max_val, dtype),
        }
    }

    /// MSE calibration: minimize mean squared error
    fn calibrate_mse(
        &self,
        dtype: QuantDtype,
        scheme: QuantScheme,
        num_candidates: u32,
    ) -> QuantParams {
        let values = match &self.stats.collected_values {
            Some(v) if !v.is_empty() => v,
            _ => return self.calibrate_minmax(dtype, scheme),
        };

        let abs_max = self.stats.min_val.abs().max(self.stats.max_val.abs());
        let mut best_threshold = abs_max;
        let mut best_mse = f64::MAX;

        // Try different thresholds
        for i in 1..=num_candidates {
            let threshold = abs_max * (i as f32 / num_candidates as f32);
            let (_, qmax) = dtype.range();
            let scale = threshold / qmax as f32;

            // Compute MSE for this threshold
            let mut mse = 0.0;
            for &val in values {
                let clipped = val.clamp(-threshold, threshold);
                let quantized = (clipped / scale).round() as i32;
                let dequantized = quantized as f32 * scale;
                let err = (val - dequantized) as f64;
                mse += err * err;
            }
            mse /= values.len() as f64;

            if mse < best_mse {
                best_mse = mse;
                best_threshold = threshold;
            }
        }

        match scheme {
            QuantScheme::Symmetric | QuantScheme::PerTensor => {
                QuantParams::from_minmax_symmetric(-best_threshold, best_threshold, dtype)
            }
            _ => QuantParams::from_minmax_symmetric(-best_threshold, best_threshold, dtype),
        }
    }

    /// Reset the collector
    pub fn reset(&mut self) {
        self.stats = CalibrationStats::new(&self.method);
        self.edges_initialized = false;
    }

    /// Get the number of samples collected
    pub fn sample_count(&self) -> usize {
        self.stats.count
    }

    /// Get observed min value
    pub fn min_val(&self) -> f32 {
        self.stats.min_val
    }

    /// Get observed max value
    pub fn max_val(&self) -> f32 {
        self.stats.max_val
    }

    /// Get the calibration method
    pub fn method(&self) -> CalibrationMethod {
        self.method
    }
}

// ============================================================================
// Per-Channel Calibration
// ============================================================================

/// Calibrator for per-channel quantization
pub struct PerChannelCalibrator {
    /// Calibrators per channel
    calibrators: Vec<CalibrationCollector>,
    /// Axis for per-channel quantization
    axis: u32,
    /// Number of channels
    num_channels: usize,
}

impl PerChannelCalibrator {
    /// Create a new per-channel calibrator
    pub fn new(num_channels: usize, axis: u32, method: CalibrationMethod) -> Self {
        let calibrators = (0..num_channels)
            .map(|_| CalibrationCollector::new(method))
            .collect();

        Self {
            calibrators,
            axis,
            num_channels,
        }
    }

    /// Collect values for a specific channel
    pub fn collect_channel(&mut self, channel: usize, values: &[f32]) {
        if channel < self.num_channels {
            self.calibrators[channel].collect(values);
        }
    }

    /// Collect a 2D tensor [channels, elements] (axis=0) or [elements, channels] (axis=1)
    pub fn collect_tensor(&mut self, data: &[f32], shape: &[usize]) {
        if shape.len() < 2 {
            return;
        }

        let axis = self.axis as usize;
        if axis >= shape.len() {
            return;
        }

        // Assuming contiguous memory layout
        let num_channels = shape[axis];
        if num_channels != self.num_channels {
            return;
        }

        // Collect per-channel
        if axis == 0 {
            // Shape: [channels, ...]
            let elements_per_channel = data.len() / num_channels;
            for c in 0..num_channels {
                let start = c * elements_per_channel;
                let end = start + elements_per_channel;
                self.collect_channel(c, &data[start..end]);
            }
        } else {
            // For other axis, need to stride through data
            let stride: usize = shape[axis + 1..].iter().product();
            let outer_size: usize = shape[..axis].iter().product();

            for c in 0..num_channels {
                let mut channel_data = Vec::new();
                for outer in 0..outer_size {
                    for inner in 0..stride {
                        let idx = outer * num_channels * stride + c * stride + inner;
                        if idx < data.len() {
                            channel_data.push(data[idx]);
                        }
                    }
                }
                self.collect_channel(c, &channel_data);
            }
        }
    }

    /// Compute per-channel quantization parameters
    pub fn compute_params(&self, dtype: QuantDtype) -> super::quantize::PerChannelQuantParams {
        let mut scales = Vec::with_capacity(self.num_channels);
        let mut zero_points = Vec::with_capacity(self.num_channels);

        for calibrator in &self.calibrators {
            let params = calibrator.compute_params(dtype, QuantScheme::Symmetric);
            scales.push(params.scale);
            zero_points.push(params.zero_point);
        }

        super::quantize::PerChannelQuantParams::new(scales, zero_points, self.axis, dtype)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_calibration() {
        let mut collector = CalibrationCollector::new(CalibrationMethod::MinMax);
        collector.collect(&[-1.0, 0.0, 1.0, 2.0, -2.0]);

        let params = collector.compute_params(QuantDtype::Int8, QuantScheme::Symmetric);
        assert_eq!(params.zero_point, 0);
        // scale should be 2.0 / 127
        assert!((params.scale - 2.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_histogram_calibration() {
        let mut collector =
            CalibrationCollector::new(CalibrationMethod::Histogram { num_bins: 256 });

        // Generate normal-ish distribution
        let values: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.01).collect();
        collector.collect(&values);

        let params = collector.compute_params(QuantDtype::Int8, QuantScheme::Symmetric);

        // Should have reasonable scale
        assert!(params.scale > 0.0);
        assert!(params.scale < 0.02); // Should clip outliers
    }

    #[test]
    fn test_entropy_calibration() {
        let mut collector = CalibrationCollector::new(CalibrationMethod::Entropy {
            num_bins: 256,
            num_quantiles: 100,
        });

        let values: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.01).collect();
        collector.collect(&values);

        let params = collector.compute_params(QuantDtype::Int8, QuantScheme::Symmetric);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_percentile_calibration() {
        let mut collector = CalibrationCollector::new(CalibrationMethod::Percentile {
            lower: 0.01,
            upper: 0.99,
        });

        // Values with outliers
        let mut values: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.01).collect();
        values.push(100.0); // Outlier
        values.push(-100.0); // Outlier
        collector.collect(&values);

        let params = collector.compute_params(QuantDtype::Int8, QuantScheme::Symmetric);

        // Should clip outliers, so scale should be based on ~1.0, not 100.0
        assert!(params.scale < 1.0);
    }

    #[test]
    fn test_mse_calibration() {
        let mut collector = CalibrationCollector::new(CalibrationMethod::Mse {
            num_candidates: 100,
        });

        let values: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.01).collect();
        collector.collect(&values);

        let params = collector.compute_params(QuantDtype::Int8, QuantScheme::Symmetric);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_per_channel_calibrator() {
        let mut calibrator = PerChannelCalibrator::new(4, 0, CalibrationMethod::MinMax);

        // Collect different ranges for each channel
        calibrator.collect_channel(0, &[-1.0, 1.0]);
        calibrator.collect_channel(1, &[-2.0, 2.0]);
        calibrator.collect_channel(2, &[-0.5, 0.5]);
        calibrator.collect_channel(3, &[-1.5, 1.5]);

        let params = calibrator.compute_params(QuantDtype::Int8);

        assert_eq!(params.num_channels, 4);
        // Each channel should have different scale
        assert!((params.scales[0] - 1.0 / 127.0).abs() < 1e-6);
        assert!((params.scales[1] - 2.0 / 127.0).abs() < 1e-6);
        assert!((params.scales[2] - 0.5 / 127.0).abs() < 1e-6);
        assert!((params.scales[3] - 1.5 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_calibration_stats() {
        let mut collector = CalibrationCollector::new(CalibrationMethod::MinMax);
        collector.collect(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(collector.sample_count(), 5);
        assert_eq!(collector.min_val(), 1.0);
        assert_eq!(collector.max_val(), 5.0);
    }

    #[test]
    fn test_reset() {
        let mut collector = CalibrationCollector::new(CalibrationMethod::MinMax);
        collector.collect(&[1.0, 2.0, 3.0]);
        assert_eq!(collector.sample_count(), 3);

        collector.reset();
        assert_eq!(collector.sample_count(), 0);
        assert_eq!(collector.min_val(), f32::INFINITY);
    }
}
