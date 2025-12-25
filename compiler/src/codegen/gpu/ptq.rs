//! Post-Training Quantization (PTQ) Framework
//!
//! Provides a complete PTQ workflow for quantizing trained models:
//! - Calibration pass to determine optimal quantization parameters
//! - Weight quantization (typically per-channel INT8)
//! - Activation quantization (typically per-tensor INT8)
//! - Quantized module generation
//!
//! # Workflow
//!
//! ```text
//! ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
//! │  FP32 Model │───▶│ Calibration │───▶│  Quantize   │───▶│ INT8 Model  │
//! │   Weights   │    │    Pass     │    │   Weights   │    │   Deploy    │
//! └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
//!                          │
//!                    ┌─────▼─────┐
//!                    │Calibration│
//!                    │   Data    │
//!                    └───────────┘
//! ```

use std::collections::HashMap;

use super::calibration::{CalibrationCollector, CalibrationMethod, PerChannelCalibrator};
use super::quantize::{
    PerChannelQuantParams, QuantDtype, QuantError, QuantErrorAnalyzer, QuantParams, QuantScheme,
    QuantizedTensor,
};

// ============================================================================
// PTQ Configuration
// ============================================================================

/// Configuration for weight quantization
#[derive(Debug, Clone)]
pub struct WeightQuantConfig {
    /// Quantization data type
    pub dtype: QuantDtype,
    /// Quantization scheme (typically per-channel for weights)
    pub scheme: QuantScheme,
    /// Whether to skip biases (usually kept in FP32)
    pub skip_bias: bool,
    /// Layers to exclude from quantization
    pub excluded_layers: Vec<String>,
}

impl Default for WeightQuantConfig {
    fn default() -> Self {
        Self {
            dtype: QuantDtype::Int8,
            scheme: QuantScheme::PerChannel { axis: 0 }, // Per output channel
            skip_bias: true,
            excluded_layers: Vec::new(),
        }
    }
}

/// Configuration for activation quantization
#[derive(Debug, Clone)]
pub struct ActivationQuantConfig {
    /// Quantization data type
    pub dtype: QuantDtype,
    /// Quantization scheme (typically per-tensor for activations)
    pub scheme: QuantScheme,
    /// Calibration method
    pub calibration_method: CalibrationMethod,
    /// Whether to quantize input activations
    pub quantize_inputs: bool,
    /// Whether to quantize output activations
    pub quantize_outputs: bool,
}

impl Default for ActivationQuantConfig {
    fn default() -> Self {
        Self {
            dtype: QuantDtype::Int8,
            scheme: QuantScheme::PerTensor,
            calibration_method: CalibrationMethod::Histogram { num_bins: 2048 },
            quantize_inputs: true,
            quantize_outputs: true,
        }
    }
}

/// Full PTQ configuration
#[derive(Debug, Clone)]
pub struct PtqConfig {
    /// Weight quantization settings
    pub weight_config: WeightQuantConfig,
    /// Activation quantization settings
    pub activation_config: ActivationQuantConfig,
    /// Number of calibration batches
    pub num_calibration_batches: usize,
    /// Whether to perform error analysis
    pub analyze_errors: bool,
    /// Target error threshold (MSE)
    pub error_threshold: Option<f64>,
    /// Whether to use per-channel for the first layer
    pub first_layer_per_channel: bool,
    /// Whether to use per-channel for the last layer
    pub last_layer_per_channel: bool,
}

impl Default for PtqConfig {
    fn default() -> Self {
        Self {
            weight_config: WeightQuantConfig::default(),
            activation_config: ActivationQuantConfig::default(),
            num_calibration_batches: 100,
            analyze_errors: true,
            error_threshold: None,
            first_layer_per_channel: true,
            last_layer_per_channel: true,
        }
    }
}

impl PtqConfig {
    /// Create a config optimized for accuracy
    pub fn accuracy_focused() -> Self {
        Self {
            weight_config: WeightQuantConfig {
                dtype: QuantDtype::Int8,
                scheme: QuantScheme::PerChannel { axis: 0 },
                skip_bias: true,
                excluded_layers: Vec::new(),
            },
            activation_config: ActivationQuantConfig {
                dtype: QuantDtype::Int8,
                scheme: QuantScheme::PerTensor,
                calibration_method: CalibrationMethod::Entropy {
                    num_bins: 2048,
                    num_quantiles: 256,
                },
                quantize_inputs: true,
                quantize_outputs: true,
            },
            num_calibration_batches: 200,
            analyze_errors: true,
            error_threshold: Some(0.001),
            first_layer_per_channel: true,
            last_layer_per_channel: true,
        }
    }

    /// Create a config optimized for speed
    pub fn speed_focused() -> Self {
        Self {
            weight_config: WeightQuantConfig {
                dtype: QuantDtype::Int8,
                scheme: QuantScheme::PerTensor,
                skip_bias: true,
                excluded_layers: Vec::new(),
            },
            activation_config: ActivationQuantConfig {
                dtype: QuantDtype::Int8,
                scheme: QuantScheme::PerTensor,
                calibration_method: CalibrationMethod::MinMax,
                quantize_inputs: true,
                quantize_outputs: true,
            },
            num_calibration_batches: 50,
            analyze_errors: false,
            error_threshold: None,
            first_layer_per_channel: false,
            last_layer_per_channel: false,
        }
    }

    /// Create a config for INT4 quantization (aggressive)
    pub fn int4_aggressive() -> Self {
        Self {
            weight_config: WeightQuantConfig {
                dtype: QuantDtype::Int4,
                scheme: QuantScheme::PerGroup {
                    axis: 0,
                    group_size: 128,
                },
                skip_bias: true,
                excluded_layers: Vec::new(),
            },
            activation_config: ActivationQuantConfig {
                dtype: QuantDtype::Int8, // Keep activations at INT8
                scheme: QuantScheme::PerTensor,
                calibration_method: CalibrationMethod::Histogram { num_bins: 2048 },
                quantize_inputs: true,
                quantize_outputs: true,
            },
            num_calibration_batches: 100,
            analyze_errors: true,
            error_threshold: Some(0.01),
            first_layer_per_channel: true,
            last_layer_per_channel: true,
        }
    }
}

// ============================================================================
// Layer Information
// ============================================================================

/// Information about a quantizable layer
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name/identifier
    pub name: String,
    /// Layer type (e.g., "conv", "linear", "matmul")
    pub layer_type: String,
    /// Weight tensor shape
    pub weight_shape: Vec<usize>,
    /// Whether this layer has bias
    pub has_bias: bool,
    /// Input activation shape (batch, ...)
    pub input_shape: Vec<usize>,
    /// Output activation shape (batch, ...)
    pub output_shape: Vec<usize>,
}

/// Quantization status for a layer
#[derive(Debug, Clone)]
pub struct LayerQuantStatus {
    /// Layer info
    pub info: LayerInfo,
    /// Weight quantization params (first channel, for reference)
    pub weight_params: Option<QuantParams>,
    /// Per-channel weight params (if per-channel quantization)
    pub weight_per_channel_params: Option<PerChannelQuantParams>,
    /// Input activation quantization params
    pub input_params: Option<QuantParams>,
    /// Output activation quantization params
    pub output_params: Option<QuantParams>,
    /// Quantization error for weights
    pub weight_error: Option<QuantError>,
    /// Whether this layer is excluded
    pub excluded: bool,
}

// ============================================================================
// PTQ Engine
// ============================================================================

/// Post-Training Quantization Engine
///
/// Manages the complete PTQ workflow including calibration and quantization.
pub struct PtqEngine {
    /// Configuration
    config: PtqConfig,
    /// Layer information
    layers: Vec<LayerInfo>,
    /// Weight calibrators (per-layer)
    weight_calibrators: HashMap<String, PerChannelCalibrator>,
    /// Activation calibrators (per-layer, per-input/output)
    activation_calibrators: HashMap<String, CalibrationCollector>,
    /// Computed quantization parameters
    quant_params: HashMap<String, QuantParams>,
    /// Per-channel quantization parameters
    per_channel_params: HashMap<String, PerChannelQuantParams>,
    /// Quantization errors
    quant_errors: HashMap<String, QuantError>,
    /// Number of calibration samples seen
    calibration_samples: usize,
    /// Whether calibration is complete
    calibration_complete: bool,
}

impl PtqEngine {
    /// Create a new PTQ engine
    pub fn new(config: PtqConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
            weight_calibrators: HashMap::new(),
            activation_calibrators: HashMap::new(),
            quant_params: HashMap::new(),
            per_channel_params: HashMap::new(),
            quant_errors: HashMap::new(),
            calibration_samples: 0,
            calibration_complete: false,
        }
    }

    /// Register a layer for quantization
    pub fn register_layer(&mut self, info: LayerInfo) {
        let name = info.name.clone();

        // Create weight calibrator if needed
        if !self.config.weight_config.excluded_layers.contains(&name) {
            match &self.config.weight_config.scheme {
                QuantScheme::PerChannel { axis } => {
                    let num_channels = if *axis as usize >= info.weight_shape.len() {
                        1
                    } else {
                        info.weight_shape[*axis as usize]
                    };
                    let calibrator = PerChannelCalibrator::new(
                        num_channels,
                        *axis,
                        CalibrationMethod::MinMax, // MinMax for weights
                    );
                    self.weight_calibrators.insert(name.clone(), calibrator);
                }
                _ => {
                    let collector = CalibrationCollector::new(CalibrationMethod::MinMax);
                    self.activation_calibrators
                        .insert(format!("{}_weight", name), collector);
                }
            }
        }

        // Create activation calibrators
        if self.config.activation_config.quantize_inputs {
            let collector =
                CalibrationCollector::new(self.config.activation_config.calibration_method);
            self.activation_calibrators
                .insert(format!("{}_input", name), collector);
        }

        if self.config.activation_config.quantize_outputs {
            let collector =
                CalibrationCollector::new(self.config.activation_config.calibration_method);
            self.activation_calibrators
                .insert(format!("{}_output", name), collector);
        }

        self.layers.push(info);
    }

    /// Collect calibration data for weights
    pub fn calibrate_weights(&mut self, layer_name: &str, weights: &[f32]) {
        if let Some(calibrator) = self.weight_calibrators.get_mut(layer_name) {
            // For per-channel, we need to reshape
            // For now, treat as single channel
            calibrator.collect_channel(0, weights);
        } else if let Some(collector) = self
            .activation_calibrators
            .get_mut(&format!("{}_weight", layer_name))
        {
            collector.collect(weights);
        }
    }

    /// Collect calibration data for per-channel weights
    pub fn calibrate_weights_per_channel(
        &mut self,
        layer_name: &str,
        channel: usize,
        weights: &[f32],
    ) {
        if let Some(calibrator) = self.weight_calibrators.get_mut(layer_name) {
            calibrator.collect_channel(channel, weights);
        }
    }

    /// Collect calibration data for input activations
    pub fn calibrate_input(&mut self, layer_name: &str, activations: &[f32]) {
        let key = format!("{}_input", layer_name);
        if let Some(collector) = self.activation_calibrators.get_mut(&key) {
            collector.collect(activations);
        }
    }

    /// Collect calibration data for output activations
    pub fn calibrate_output(&mut self, layer_name: &str, activations: &[f32]) {
        let key = format!("{}_output", layer_name);
        if let Some(collector) = self.activation_calibrators.get_mut(&key) {
            collector.collect(activations);
        }
    }

    /// Increment calibration sample count
    pub fn increment_calibration_batch(&mut self) {
        self.calibration_samples += 1;
    }

    /// Check if calibration is complete
    pub fn is_calibration_complete(&self) -> bool {
        self.calibration_samples >= self.config.num_calibration_batches
    }

    /// Compute quantization parameters from calibration data
    pub fn compute_quant_params(&mut self) {
        // Compute weight params (per-channel)
        for (name, calibrator) in &self.weight_calibrators {
            let params = calibrator.compute_params(self.config.weight_config.dtype);
            self.per_channel_params.insert(name.clone(), params);
        }

        // Compute activation params
        for (key, collector) in &self.activation_calibrators {
            if key.ends_with("_weight") {
                // Per-tensor weight quantization
                let params = collector
                    .compute_params(self.config.weight_config.dtype, QuantScheme::PerTensor);
                self.quant_params.insert(key.clone(), params);
            } else {
                // Activation quantization
                let params = collector.compute_params(
                    self.config.activation_config.dtype,
                    self.config.activation_config.scheme,
                );
                self.quant_params.insert(key.clone(), params);
            }
        }

        self.calibration_complete = true;
    }

    /// Get quantization parameters for a layer's weights
    pub fn get_weight_params(&self, layer_name: &str) -> Option<&PerChannelQuantParams> {
        self.per_channel_params.get(layer_name)
    }

    /// Get quantization parameters for a layer's input
    pub fn get_input_params(&self, layer_name: &str) -> Option<&QuantParams> {
        self.quant_params.get(&format!("{}_input", layer_name))
    }

    /// Get quantization parameters for a layer's output
    pub fn get_output_params(&self, layer_name: &str) -> Option<&QuantParams> {
        self.quant_params.get(&format!("{}_output", layer_name))
    }

    /// Quantize a weight tensor
    pub fn quantize_weights(
        &mut self,
        layer_name: &str,
        weights: &[f32],
        shape: &[usize],
    ) -> Option<QuantizedTensor> {
        if !self.calibration_complete {
            return None;
        }

        // Check if excluded
        if self
            .config
            .weight_config
            .excluded_layers
            .contains(&layer_name.to_string())
        {
            return None;
        }

        // Get per-channel params
        if let Some(per_channel_params) = self.per_channel_params.get(layer_name) {
            if per_channel_params.num_channels == 0 {
                return None;
            }

            // Use first channel param as representative (for per-tensor case)
            let first_param = per_channel_params.params_for_channel(0);

            // Quantize based on dtype
            let data = match per_channel_params.dtype {
                QuantDtype::Int8 | QuantDtype::UInt8 => {
                    // Per-channel quantization
                    let axis = per_channel_params.axis as usize;
                    let num_channels = if axis < shape.len() { shape[axis] } else { 1 };
                    let channel_size: usize = if axis + 1 < shape.len() {
                        shape[axis + 1..].iter().product()
                    } else {
                        1
                    };

                    let mut quantized = Vec::with_capacity(weights.len());
                    for c in 0..num_channels {
                        let param = if c < per_channel_params.num_channels {
                            per_channel_params.params_for_channel(c)
                        } else {
                            first_param.clone()
                        };
                        let start = c * channel_size;
                        let end = start + channel_size;
                        for &v in &weights[start..end.min(weights.len())] {
                            quantized.push(param.quantize(v) as u8);
                        }
                    }
                    quantized
                }
                QuantDtype::Int4 | QuantDtype::UInt4 => {
                    // Pack INT4 values
                    let mut packed = Vec::with_capacity(weights.len().div_ceil(2));
                    for chunk in weights.chunks(2) {
                        let lo = first_param.quantize(chunk[0]);
                        let hi = if chunk.len() > 1 {
                            first_param.quantize(chunk[1])
                        } else {
                            0
                        };
                        packed.push(super::quantize::pack_int4(lo as i8, hi as i8));
                    }
                    packed
                }
                _ => return None,
            };

            // Compute quantization error if enabled
            if self.config.analyze_errors {
                let mut analyzer = QuantErrorAnalyzer::new();
                for &v in weights.iter() {
                    let q = first_param.quantize(v);
                    let dq = first_param.dequantize(q);
                    let was_clipped = q <= first_param.range.0 || q >= first_param.range.1;
                    analyzer.add(v, dq, was_clipped);
                }
                let error = analyzer.compute();
                self.quant_errors.insert(layer_name.to_string(), error);
            }

            Some(QuantizedTensor {
                data,
                params: first_param,
                shape: shape.to_vec(),
                name: layer_name.to_string(),
            })
        } else {
            None
        }
    }

    /// Get quantization status for all layers
    pub fn get_layer_status(&self) -> Vec<LayerQuantStatus> {
        self.layers
            .iter()
            .map(|info| {
                let name = &info.name;
                let excluded = self.config.weight_config.excluded_layers.contains(name);

                let per_channel = self.per_channel_params.get(name).cloned();
                let weight_params = per_channel.as_ref().map(|p| p.params_for_channel(0));

                LayerQuantStatus {
                    info: info.clone(),
                    weight_params,
                    weight_per_channel_params: per_channel,
                    input_params: self.quant_params.get(&format!("{}_input", name)).cloned(),
                    output_params: self.quant_params.get(&format!("{}_output", name)).cloned(),
                    weight_error: self.quant_errors.get(name).cloned(),
                    excluded,
                }
            })
            .collect()
    }

    /// Get overall quantization error summary
    pub fn get_error_summary(&self) -> PtqErrorSummary {
        let errors: Vec<_> = self.quant_errors.values().collect();

        if errors.is_empty() {
            return PtqErrorSummary::default();
        }

        let total_mse: f64 = errors.iter().map(|e| e.mse).sum();
        let total_mae: f64 = errors.iter().map(|e| e.mae).sum();
        let max_error = errors.iter().map(|e| e.max_error).fold(0.0, f64::max);
        let avg_snr = errors.iter().map(|e| e.snr_db).sum::<f64>() / errors.len() as f64;

        PtqErrorSummary {
            avg_mse: total_mse / errors.len() as f64,
            avg_mae: total_mae / errors.len() as f64,
            max_error,
            avg_snr_db: avg_snr,
            num_layers: errors.len(),
            layers_above_threshold: self.count_layers_above_threshold(),
        }
    }

    fn count_layers_above_threshold(&self) -> usize {
        if let Some(threshold) = self.config.error_threshold {
            self.quant_errors
                .values()
                .filter(|e| e.mse > threshold)
                .count()
        } else {
            0
        }
    }

    /// Get configuration
    pub fn config(&self) -> &PtqConfig {
        &self.config
    }

    /// Get calibration progress
    pub fn calibration_progress(&self) -> f32 {
        self.calibration_samples as f32 / self.config.num_calibration_batches as f32
    }
}

/// Summary of PTQ errors across all layers
#[derive(Debug, Clone, Default)]
pub struct PtqErrorSummary {
    /// Average MSE across layers
    pub avg_mse: f64,
    /// Average MAE across layers
    pub avg_mae: f64,
    /// Maximum error across all layers
    pub max_error: f64,
    /// Average SNR in dB
    pub avg_snr_db: f64,
    /// Number of quantized layers
    pub num_layers: usize,
    /// Number of layers above error threshold
    pub layers_above_threshold: usize,
}

// ============================================================================
// Quantized Module
// ============================================================================

/// A fully quantized module ready for deployment
#[derive(Debug)]
pub struct QuantizedModule {
    /// Quantized weight tensors
    pub weights: HashMap<String, QuantizedTensor>,
    /// Activation quantization parameters
    pub activation_params: HashMap<String, QuantParams>,
    /// Layer ordering for execution
    pub layer_order: Vec<String>,
    /// Configuration used
    pub config: PtqConfig,
    /// Error summary
    pub error_summary: PtqErrorSummary,
}

impl QuantizedModule {
    /// Get total quantized model size in bytes
    pub fn size_bytes(&self) -> usize {
        self.weights.values().map(|t| t.data.len()).sum()
    }

    /// Get compression ratio compared to FP32
    pub fn compression_ratio(&self) -> f32 {
        let fp32_size: usize = self
            .weights
            .values()
            .map(|t| t.shape.iter().product::<usize>() * 4)
            .sum();
        let quant_size = self.size_bytes();
        if quant_size > 0 {
            fp32_size as f32 / quant_size as f32
        } else {
            1.0
        }
    }

    /// Check if model meets error threshold
    pub fn meets_error_threshold(&self) -> bool {
        self.error_summary.layers_above_threshold == 0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptq_config_default() {
        let config = PtqConfig::default();
        assert_eq!(config.weight_config.dtype, QuantDtype::Int8);
        assert_eq!(config.activation_config.dtype, QuantDtype::Int8);
        assert_eq!(config.num_calibration_batches, 100);
    }

    #[test]
    fn test_ptq_config_presets() {
        let accuracy = PtqConfig::accuracy_focused();
        assert_eq!(accuracy.num_calibration_batches, 200);
        assert!(accuracy.analyze_errors);

        let speed = PtqConfig::speed_focused();
        assert_eq!(speed.num_calibration_batches, 50);
        assert!(!speed.analyze_errors);

        let int4 = PtqConfig::int4_aggressive();
        assert_eq!(int4.weight_config.dtype, QuantDtype::Int4);
    }

    #[test]
    fn test_ptq_engine_layer_registration() {
        let config = PtqConfig::default();
        let mut engine = PtqEngine::new(config);

        let layer = LayerInfo {
            name: "conv1".to_string(),
            layer_type: "conv".to_string(),
            weight_shape: vec![64, 3, 3, 3],
            has_bias: true,
            input_shape: vec![1, 3, 224, 224],
            output_shape: vec![1, 64, 224, 224],
        };

        engine.register_layer(layer);
        assert_eq!(engine.layers.len(), 1);
    }

    #[test]
    fn test_ptq_calibration_workflow() {
        let config = PtqConfig::default();
        let mut engine = PtqEngine::new(config);

        let layer = LayerInfo {
            name: "linear1".to_string(),
            layer_type: "linear".to_string(),
            weight_shape: vec![128, 64],
            has_bias: true,
            input_shape: vec![1, 64],
            output_shape: vec![1, 128],
        };
        engine.register_layer(layer);

        // Simulate calibration
        let activations: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        for _ in 0..100 {
            engine.calibrate_input("linear1", &activations);
            engine.calibrate_output("linear1", &activations);
            engine.increment_calibration_batch();
        }

        assert!(engine.is_calibration_complete());

        engine.compute_quant_params();
        assert!(engine.get_input_params("linear1").is_some());
        assert!(engine.get_output_params("linear1").is_some());
    }

    #[test]
    fn test_calibration_progress() {
        let mut config = PtqConfig::default();
        config.num_calibration_batches = 10;
        let mut engine = PtqEngine::new(config);

        assert_eq!(engine.calibration_progress(), 0.0);

        for _ in 0..5 {
            engine.increment_calibration_batch();
        }
        assert_eq!(engine.calibration_progress(), 0.5);

        for _ in 0..5 {
            engine.increment_calibration_batch();
        }
        assert_eq!(engine.calibration_progress(), 1.0);
    }

    #[test]
    fn test_quantized_module() {
        let weights = HashMap::new();
        let activation_params = HashMap::new();

        let module = QuantizedModule {
            weights,
            activation_params,
            layer_order: vec!["layer1".to_string()],
            config: PtqConfig::default(),
            error_summary: PtqErrorSummary::default(),
        };

        assert_eq!(module.size_bytes(), 0);
        assert_eq!(module.compression_ratio(), 1.0);
        assert!(module.meets_error_threshold());
    }

    #[test]
    fn test_error_summary() {
        let config = PtqConfig::default();
        let engine = PtqEngine::new(config);
        let summary = engine.get_error_summary();

        assert_eq!(summary.num_layers, 0);
        assert_eq!(summary.avg_mse, 0.0);
    }
}
