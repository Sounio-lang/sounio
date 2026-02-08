//! Quantization Types and Operations
//!
//! Core infrastructure for INT8/INT4 quantization:
//! - Quantization schemes (symmetric, asymmetric, per-channel)
//! - Scale and zero-point parameters
//! - Quantization error analysis
//! - Quantize/dequantize operations
//!
//! # Quantization Formula
//!
//! ```text
//! Quantize:   q = clamp(round(x / scale) + zero_point, qmin, qmax)
//! Dequantize: x = (q - zero_point) * scale
//! ```
//!
//! # Supported Dtypes
//!
//! | Dtype | Range | Use Case |
//! |-------|-------|----------|
//! | INT8  | [-128, 127] | Weights, activations |
//! | UINT8 | [0, 255] | Activations (ReLU) |
//! | INT4  | [-8, 7] | Extreme compression |
//! | UINT4 | [0, 15] | Extreme compression |

use std::fmt;

// ============================================================================
// Quantization Data Types
// ============================================================================

/// Target quantized data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantDtype {
    /// Signed 8-bit integer [-128, 127]
    Int8,
    /// Unsigned 8-bit integer [0, 255]
    UInt8,
    /// Signed 4-bit integer [-8, 7] (packed, 2 per byte)
    Int4,
    /// Unsigned 4-bit integer [0, 15] (packed, 2 per byte)
    UInt4,
    /// FP8 E4M3 format (existing in Sounio)
    FP8E4M3,
    /// FP8 E5M2 format (existing in Sounio)
    FP8E5M2,
}

impl QuantDtype {
    /// Get the valid range (qmin, qmax) for this dtype
    pub fn range(&self) -> (i32, i32) {
        match self {
            QuantDtype::Int8 => (-128, 127),
            QuantDtype::UInt8 => (0, 255),
            QuantDtype::Int4 => (-8, 7),
            QuantDtype::UInt4 => (0, 15),
            QuantDtype::FP8E4M3 => (-448, 448), // Approximate max for E4M3
            QuantDtype::FP8E5M2 => (-57344, 57344), // Approximate max for E5M2
        }
    }

    /// Get the number of bits for this dtype
    pub fn bits(&self) -> u32 {
        match self {
            QuantDtype::Int8 | QuantDtype::UInt8 => 8,
            QuantDtype::Int4 | QuantDtype::UInt4 => 4,
            QuantDtype::FP8E4M3 | QuantDtype::FP8E5M2 => 8,
        }
    }

    /// Check if this is a signed type
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            QuantDtype::Int8 | QuantDtype::Int4 | QuantDtype::FP8E4M3 | QuantDtype::FP8E5M2
        )
    }

    /// Check if this is a packed type (multiple values per byte)
    pub fn is_packed(&self) -> bool {
        matches!(self, QuantDtype::Int4 | QuantDtype::UInt4)
    }

    /// Get the number of values packed per byte
    pub fn values_per_byte(&self) -> u32 {
        if self.is_packed() { 2 } else { 1 }
    }

    /// Get the minimum CUDA compute capability required
    pub fn min_compute_capability(&self) -> (u32, u32) {
        match self {
            QuantDtype::Int8 | QuantDtype::UInt8 => (6, 1), // dp4a requires sm_61
            QuantDtype::Int4 | QuantDtype::UInt4 => (7, 5), // INT4 tensor cores sm_75
            QuantDtype::FP8E4M3 | QuantDtype::FP8E5M2 => (8, 9), // FP8 requires sm_89
        }
    }
}

impl fmt::Display for QuantDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantDtype::Int8 => write!(f, "int8"),
            QuantDtype::UInt8 => write!(f, "uint8"),
            QuantDtype::Int4 => write!(f, "int4"),
            QuantDtype::UInt4 => write!(f, "uint4"),
            QuantDtype::FP8E4M3 => write!(f, "fp8_e4m3"),
            QuantDtype::FP8E5M2 => write!(f, "fp8_e5m2"),
        }
    }
}

// ============================================================================
// Quantization Scheme
// ============================================================================

/// Quantization scheme determining how scale/zero_point are computed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QuantScheme {
    /// Symmetric quantization: zero_point = 0
    /// scale = max(|min|, |max|) / qmax
    /// Best for weights (centered around 0)
    #[default]
    Symmetric,

    /// Asymmetric quantization: uses both scale and zero_point
    /// scale = (max - min) / (qmax - qmin)
    /// zero_point = qmin - round(min / scale)
    /// Best for activations (may be non-negative after ReLU)
    Asymmetric,

    /// Per-channel quantization: different scale/zp per output channel
    /// Typically used for conv/linear weights
    PerChannel {
        /// The axis along which to quantize (usually 0 for output channels)
        axis: u32,
    },

    /// Per-tensor quantization: single scale/zp for entire tensor
    /// Simpler but potentially less accurate
    PerTensor,

    /// Per-group quantization: divide tensor into groups along axis
    /// Compromise between per-tensor and per-channel
    PerGroup {
        /// The axis along which to group
        axis: u32,
        /// Number of elements per group
        group_size: u32,
    },
}

impl QuantScheme {
    /// Check if this scheme uses zero_point
    pub fn has_zero_point(&self) -> bool {
        matches!(self, QuantScheme::Asymmetric)
    }

    /// Check if this is a per-channel scheme
    pub fn is_per_channel(&self) -> bool {
        matches!(self, QuantScheme::PerChannel { .. })
    }

    /// Get the granularity axis if applicable
    pub fn axis(&self) -> Option<u32> {
        match self {
            QuantScheme::PerChannel { axis } | QuantScheme::PerGroup { axis, .. } => Some(*axis),
            _ => None,
        }
    }
}

impl fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantScheme::Symmetric => write!(f, "symmetric"),
            QuantScheme::Asymmetric => write!(f, "asymmetric"),
            QuantScheme::PerChannel { axis } => write!(f, "per_channel(axis={})", axis),
            QuantScheme::PerTensor => write!(f, "per_tensor"),
            QuantScheme::PerGroup { axis, group_size } => {
                write!(f, "per_group(axis={}, size={})", axis, group_size)
            }
        }
    }
}

// ============================================================================
// Quantization Parameters
// ============================================================================

/// Quantization parameters: scale and zero_point
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Scale factor: float_value = (quant_value - zero_point) * scale
    pub scale: f32,
    /// Zero point offset (0 for symmetric quantization)
    pub zero_point: i32,
    /// Target quantized dtype
    pub dtype: QuantDtype,
    /// Quantization scheme used
    pub scheme: QuantScheme,
    /// Valid range (qmin, qmax)
    pub range: (i32, i32),
}

impl QuantParams {
    /// Create new quantization parameters
    pub fn new(scale: f32, zero_point: i32, dtype: QuantDtype, scheme: QuantScheme) -> Self {
        Self {
            scale,
            zero_point,
            dtype,
            scheme,
            range: dtype.range(),
        }
    }

    /// Create symmetric INT8 parameters
    pub fn symmetric_int8(scale: f32) -> Self {
        Self {
            scale,
            zero_point: 0,
            dtype: QuantDtype::Int8,
            scheme: QuantScheme::Symmetric,
            range: (-128, 127),
        }
    }

    /// Create asymmetric UINT8 parameters
    pub fn asymmetric_uint8(scale: f32, zero_point: i32) -> Self {
        Self {
            scale,
            zero_point,
            dtype: QuantDtype::UInt8,
            scheme: QuantScheme::Asymmetric,
            range: (0, 255),
        }
    }

    /// Create symmetric INT4 parameters
    pub fn symmetric_int4(scale: f32) -> Self {
        Self {
            scale,
            zero_point: 0,
            dtype: QuantDtype::Int4,
            scheme: QuantScheme::Symmetric,
            range: (-8, 7),
        }
    }

    /// Create asymmetric UINT4 parameters
    pub fn asymmetric_uint4(scale: f32, zero_point: i32) -> Self {
        Self {
            scale,
            zero_point,
            dtype: QuantDtype::UInt4,
            scheme: QuantScheme::Asymmetric,
            range: (0, 15),
        }
    }

    /// Compute parameters from observed min/max values (symmetric)
    pub fn from_minmax_symmetric(min_val: f32, max_val: f32, dtype: QuantDtype) -> Self {
        let (qmin, qmax) = dtype.range();
        let abs_max = min_val.abs().max(max_val.abs());
        let scale = if abs_max == 0.0 {
            1.0
        } else {
            abs_max / qmax as f32
        };

        Self {
            scale,
            zero_point: 0,
            dtype,
            scheme: QuantScheme::Symmetric,
            range: (qmin, qmax),
        }
    }

    /// Compute parameters from observed min/max values (asymmetric)
    pub fn from_minmax_asymmetric(min_val: f32, max_val: f32, dtype: QuantDtype) -> Self {
        let (qmin, qmax) = dtype.range();
        let data_range = max_val - min_val;
        let scale = if data_range == 0.0 {
            1.0
        } else {
            data_range / (qmax - qmin) as f32
        };
        let zero_point = ((qmin as f32) - min_val / scale).round() as i32;
        let zero_point = zero_point.clamp(qmin, qmax);

        Self {
            scale,
            zero_point,
            dtype,
            scheme: QuantScheme::Asymmetric,
            range: (qmin, qmax),
        }
    }

    /// Quantize a single f32 value
    pub fn quantize(&self, value: f32) -> i32 {
        let scaled = value / self.scale;
        let shifted = scaled + self.zero_point as f32;
        let rounded = shifted.round() as i32;
        rounded.clamp(self.range.0, self.range.1)
    }

    /// Dequantize a single quantized value
    pub fn dequantize(&self, quant_value: i32) -> f32 {
        (quant_value - self.zero_point) as f32 * self.scale
    }

    /// Quantize a vector of f32 values to i8
    pub fn quantize_to_i8(&self, values: &[f32]) -> Vec<i8> {
        values.iter().map(|&v| self.quantize(v) as i8).collect()
    }

    /// Quantize a vector of f32 values to u8
    pub fn quantize_to_u8(&self, values: &[f32]) -> Vec<u8> {
        values.iter().map(|&v| self.quantize(v) as u8).collect()
    }

    /// Dequantize a vector of i8 values
    pub fn dequantize_i8(&self, values: &[i8]) -> Vec<f32> {
        values.iter().map(|&v| self.dequantize(v as i32)).collect()
    }

    /// Dequantize a vector of u8 values
    pub fn dequantize_u8(&self, values: &[u8]) -> Vec<f32> {
        values.iter().map(|&v| self.dequantize(v as i32)).collect()
    }

    /// Pack two INT4 values into a single byte
    pub fn pack_int4(lo: i8, hi: i8) -> u8 {
        let lo_bits = (lo as u8) & 0x0F;
        let hi_bits = ((hi as u8) & 0x0F) << 4;
        lo_bits | hi_bits
    }

    /// Unpack a byte into two INT4 values
    pub fn unpack_int4(packed: u8) -> (i8, i8) {
        let lo = (packed & 0x0F) as i8;
        let hi = ((packed >> 4) & 0x0F) as i8;
        // Sign extend if needed
        let lo = if lo > 7 { lo - 16 } else { lo };
        let hi = if hi > 7 { hi - 16 } else { hi };
        (lo, hi)
    }
}

impl fmt::Display for QuantParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantParams(scale={:.6}, zp={}, dtype={}, scheme={})",
            self.scale, self.zero_point, self.dtype, self.scheme
        )
    }
}

// ============================================================================
// Per-Channel Quantization Parameters
// ============================================================================

/// Per-channel quantization parameters (different scale/zp per channel)
#[derive(Debug, Clone)]
pub struct PerChannelQuantParams {
    /// Scale per channel
    pub scales: Vec<f32>,
    /// Zero points per channel
    pub zero_points: Vec<i32>,
    /// Axis for per-channel quantization
    pub axis: u32,
    /// Target dtype
    pub dtype: QuantDtype,
    /// Number of channels
    pub num_channels: usize,
}

impl PerChannelQuantParams {
    /// Create new per-channel parameters
    pub fn new(scales: Vec<f32>, zero_points: Vec<i32>, axis: u32, dtype: QuantDtype) -> Self {
        let num_channels = scales.len();
        assert_eq!(scales.len(), zero_points.len());
        Self {
            scales,
            zero_points,
            axis,
            dtype,
            num_channels,
        }
    }

    /// Create symmetric per-channel parameters from per-channel max values
    pub fn from_channel_maxes(max_per_channel: &[f32], axis: u32, dtype: QuantDtype) -> Self {
        let (_, qmax) = dtype.range();
        let scales: Vec<f32> = max_per_channel
            .iter()
            .map(|&max| if max == 0.0 { 1.0 } else { max / qmax as f32 })
            .collect();
        let zero_points = vec![0; scales.len()];

        Self::new(scales, zero_points, axis, dtype)
    }

    /// Get parameters for a specific channel
    pub fn params_for_channel(&self, channel: usize) -> QuantParams {
        QuantParams {
            scale: self.scales[channel],
            zero_point: self.zero_points[channel],
            dtype: self.dtype,
            scheme: QuantScheme::PerChannel { axis: self.axis },
            range: self.dtype.range(),
        }
    }
}

// ============================================================================
// Quantization Error Analysis
// ============================================================================

/// Quantization error statistics
#[derive(Debug, Clone, Default)]
pub struct QuantError {
    /// Mean squared error
    pub mse: f64,
    /// Mean absolute error
    pub mae: f64,
    /// Maximum absolute error
    pub max_error: f64,
    /// Signal-to-noise ratio in dB
    pub snr_db: f64,
    /// Percentage of values that were clipped
    pub clip_ratio: f64,
    /// Number of values analyzed
    pub count: usize,
}

impl QuantError {
    /// Create a new error report
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if the error is acceptable (SNR > threshold)
    pub fn is_acceptable(&self, min_snr_db: f64) -> bool {
        self.snr_db >= min_snr_db
    }

    /// Check if clipping is excessive
    pub fn has_excessive_clipping(&self, max_clip_ratio: f64) -> bool {
        self.clip_ratio > max_clip_ratio
    }
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantError(MSE={:.6}, MAE={:.6}, Max={:.6}, SNR={:.2}dB, Clip={:.2}%)",
            self.mse,
            self.mae,
            self.max_error,
            self.snr_db,
            self.clip_ratio * 100.0
        )
    }
}

/// Analyzer for computing quantization error
pub struct QuantErrorAnalyzer {
    /// Sum of squared original values (for SNR)
    sum_sq_original: f64,
    /// Sum of squared errors
    sum_sq_error: f64,
    /// Sum of absolute errors
    sum_abs_error: f64,
    /// Maximum error seen
    max_error: f64,
    /// Number of clipped values
    clipped_count: usize,
    /// Total count
    count: usize,
}

impl QuantErrorAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            sum_sq_original: 0.0,
            sum_sq_error: 0.0,
            sum_abs_error: 0.0,
            max_error: 0.0,
            clipped_count: 0,
            count: 0,
        }
    }

    /// Add a single value pair (original, dequantized)
    pub fn add(&mut self, original: f32, dequantized: f32, was_clipped: bool) {
        let error = (original - dequantized).abs() as f64;
        self.sum_sq_original += (original as f64).powi(2);
        self.sum_sq_error += error.powi(2);
        self.sum_abs_error += error;
        self.max_error = self.max_error.max(error);
        if was_clipped {
            self.clipped_count += 1;
        }
        self.count += 1;
    }

    /// Analyze a batch of values
    pub fn analyze_batch(&mut self, original: &[f32], params: &QuantParams) {
        for &val in original {
            let quant = params.quantize(val);
            let dequant = params.dequantize(quant);
            let was_clipped = quant == params.range.0 || quant == params.range.1;
            self.add(val, dequant, was_clipped);
        }
    }

    /// Compute final error statistics
    pub fn compute(&self) -> QuantError {
        if self.count == 0 {
            return QuantError::default();
        }

        let mse = self.sum_sq_error / self.count as f64;
        let mae = self.sum_abs_error / self.count as f64;
        let signal_power = self.sum_sq_original / self.count as f64;
        let noise_power = mse;
        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY
        };
        let clip_ratio = self.clipped_count as f64 / self.count as f64;

        QuantError {
            mse,
            mae,
            max_error: self.max_error,
            snr_db,
            clip_ratio,
            count: self.count,
        }
    }

    /// Reset the analyzer
    pub fn reset(&mut self) {
        self.sum_sq_original = 0.0;
        self.sum_sq_error = 0.0;
        self.sum_abs_error = 0.0;
        self.max_error = 0.0;
        self.clipped_count = 0;
        self.count = 0;
    }
}

impl Default for QuantErrorAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Quantized Tensor
// ============================================================================

/// A quantized tensor with scale information
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data (stored as bytes)
    pub data: Vec<u8>,
    /// Quantization parameters
    pub params: QuantParams,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Tensor name (for debugging)
    pub name: String,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(data: Vec<u8>, params: QuantParams, shape: Vec<usize>, name: String) -> Self {
        Self {
            data,
            params,
            shape,
            name,
        }
    }

    /// Get the number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get compression ratio compared to FP32
    pub fn compression_ratio(&self) -> f64 {
        let fp32_size = self.num_elements() * 4;
        fp32_size as f64 / self.size_bytes() as f64
    }

    /// Dequantize to f32 vector
    pub fn dequantize(&self) -> Vec<f32> {
        match self.params.dtype {
            QuantDtype::Int8 => {
                let i8_data: Vec<i8> = self.data.iter().map(|&b| b as i8).collect();
                self.params.dequantize_i8(&i8_data)
            }
            QuantDtype::UInt8 => self.params.dequantize_u8(&self.data),
            QuantDtype::Int4 | QuantDtype::UInt4 => {
                let mut result = Vec::with_capacity(self.num_elements());
                for &packed in &self.data {
                    let (lo, hi) = QuantParams::unpack_int4(packed);
                    result.push(self.params.dequantize(lo as i32));
                    result.push(self.params.dequantize(hi as i32));
                }
                result.truncate(self.num_elements());
                result
            }
            _ => vec![0.0; self.num_elements()], // FP8 handled elsewhere
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Pack two INT4 values into a single byte (standalone function)
pub fn pack_int4(lo: i8, hi: i8) -> u8 {
    QuantParams::pack_int4(lo, hi)
}

/// Unpack a byte into two INT4 values (standalone function)
pub fn unpack_int4(packed: u8) -> (i8, i8) {
    QuantParams::unpack_int4(packed)
}

/// Quantize a f32 tensor to INT8
pub fn quantize_tensor_int8(
    data: &[f32],
    shape: Vec<usize>,
    name: String,
    symmetric: bool,
) -> QuantizedTensor {
    // Find min/max
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute params
    let params = if symmetric {
        QuantParams::from_minmax_symmetric(min_val, max_val, QuantDtype::Int8)
    } else {
        QuantParams::from_minmax_asymmetric(min_val, max_val, QuantDtype::UInt8)
    };

    // Quantize
    let quantized: Vec<u8> = if symmetric {
        params
            .quantize_to_i8(data)
            .into_iter()
            .map(|v| v as u8)
            .collect()
    } else {
        params.quantize_to_u8(data)
    };

    QuantizedTensor::new(quantized, params, shape, name)
}

/// Quantize a f32 tensor to INT4 (packed)
pub fn quantize_tensor_int4(data: &[f32], shape: Vec<usize>, name: String) -> QuantizedTensor {
    // Find min/max
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute symmetric INT4 params
    let params = QuantParams::from_minmax_symmetric(min_val, max_val, QuantDtype::Int4);

    // Quantize and pack
    let mut packed = Vec::with_capacity(data.len().div_ceil(2));
    for chunk in data.chunks(2) {
        let lo = params.quantize(chunk[0]) as i8;
        let hi = if chunk.len() > 1 {
            params.quantize(chunk[1]) as i8
        } else {
            0
        };
        packed.push(QuantParams::pack_int4(lo, hi));
    }

    QuantizedTensor::new(packed, params, shape, name)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_dtype_range() {
        assert_eq!(QuantDtype::Int8.range(), (-128, 127));
        assert_eq!(QuantDtype::UInt8.range(), (0, 255));
        assert_eq!(QuantDtype::Int4.range(), (-8, 7));
        assert_eq!(QuantDtype::UInt4.range(), (0, 15));
    }

    #[test]
    fn test_symmetric_int8_params() {
        let params = QuantParams::symmetric_int8(0.1);
        assert_eq!(params.zero_point, 0);
        assert_eq!(params.dtype, QuantDtype::Int8);
        assert_eq!(params.range, (-128, 127));
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let params = QuantParams::symmetric_int8(0.1);
        let values = vec![0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0];

        for &val in &values {
            let quant = params.quantize(val);
            let dequant = params.dequantize(quant);
            // Error should be at most half the scale
            assert!((val - dequant).abs() <= params.scale / 2.0 + 1e-6);
        }
    }

    #[test]
    fn test_from_minmax_symmetric() {
        let params = QuantParams::from_minmax_symmetric(-2.0, 2.0, QuantDtype::Int8);
        assert_eq!(params.zero_point, 0);
        // scale should be 2.0 / 127 ≈ 0.0157
        assert!((params.scale - 2.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_minmax_asymmetric() {
        let params = QuantParams::from_minmax_asymmetric(0.0, 1.0, QuantDtype::UInt8);
        // scale should be 1.0 / 255 ≈ 0.00392
        assert!((params.scale - 1.0 / 255.0).abs() < 1e-6);
        // zero_point should be 0 since min is 0
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn test_int4_pack_unpack() {
        let lo: i8 = 3;
        let hi: i8 = -2;
        let packed = QuantParams::pack_int4(lo, hi);
        let (unpacked_lo, unpacked_hi) = QuantParams::unpack_int4(packed);
        assert_eq!(unpacked_lo, lo);
        assert_eq!(unpacked_hi, hi);
    }

    #[test]
    fn test_quant_error_analyzer() {
        let params = QuantParams::symmetric_int8(0.01);
        let mut analyzer = QuantErrorAnalyzer::new();

        // Analyze values
        let values: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.01).collect();
        analyzer.analyze_batch(&values, &params);

        let error = analyzer.compute();
        assert!(error.snr_db > 30.0); // Good SNR for this range
        assert!(error.clip_ratio < 0.01); // Very little clipping
    }

    #[test]
    fn test_quantized_tensor() {
        let data = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let tensor = quantize_tensor_int8(&data, vec![5], "test".into(), true);

        assert_eq!(tensor.num_elements(), 5);
        assert_eq!(tensor.size_bytes(), 5);
        assert!((tensor.compression_ratio() - 4.0).abs() < 0.1);

        // Dequantize and check
        let dequant = tensor.dequantize();
        for (orig, deq) in data.iter().zip(dequant.iter()) {
            assert!((orig - deq).abs() < tensor.params.scale);
        }
    }

    #[test]
    fn test_quantize_tensor_int4() {
        let data = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let tensor = quantize_tensor_int4(&data, vec![6], "test".into());

        assert_eq!(tensor.num_elements(), 6);
        // 6 values packed into 3 bytes
        assert_eq!(tensor.size_bytes(), 3);
        // 8x compression vs FP32
        assert!((tensor.compression_ratio() - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_per_channel_params() {
        let max_per_channel = vec![1.0, 2.0, 0.5, 1.5];
        let params =
            PerChannelQuantParams::from_channel_maxes(&max_per_channel, 0, QuantDtype::Int8);

        assert_eq!(params.num_channels, 4);
        assert_eq!(params.scales.len(), 4);

        // Check individual channel params
        let ch0_params = params.params_for_channel(0);
        assert!((ch0_params.scale - 1.0 / 127.0).abs() < 1e-6);
    }
}
