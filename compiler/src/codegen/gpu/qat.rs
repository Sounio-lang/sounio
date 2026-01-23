//! Quantization-Aware Training (QAT) Framework
//!
//! Provides end-to-end QAT workflow:
//! - Fake quantization layers with straight-through estimator
//! - Online min/max tracking during training
//! - Learnable scale/zero-point (optional)
//! - Integration with autodiff tape system

use super::quantize::{QuantDtype, QuantScheme};

/// QAT configuration
#[derive(Clone, Debug)]
pub struct QatConfig {
    /// Weight quantization dtype
    pub weight_dtype: QuantDtype,
    /// Activation quantization dtype
    pub activation_dtype: QuantDtype,
    /// Weight quantization scheme
    pub weight_scheme: QuantScheme,
    /// Activation quantization scheme
    pub activation_scheme: QuantScheme,
    /// Enable learnable scale parameters
    pub learnable_scale: bool,
    /// Enable learnable zero-point
    pub learnable_zero_point: bool,
    /// Number of warmup batches before enabling fake quantization
    pub warmup_batches: usize,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// Enable per-channel weight quantization
    pub per_channel_weights: bool,
}

impl Default for QatConfig {
    fn default() -> Self {
        Self {
            weight_dtype: QuantDtype::Int8,
            activation_dtype: QuantDtype::Int8,
            weight_scheme: QuantScheme::Symmetric,
            activation_scheme: QuantScheme::Symmetric,
            learnable_scale: true,
            learnable_zero_point: false, // Usually keep zero-point fixed
            warmup_batches: 1000,
            symmetric: true,
            per_channel_weights: true,
        }
    }
}

impl QatConfig {
    /// Create INT8 symmetric configuration (most common)
    pub fn int8_symmetric() -> Self {
        Self::default()
    }

    /// Create INT8 asymmetric configuration
    pub fn int8_asymmetric() -> Self {
        Self {
            weight_scheme: QuantScheme::Asymmetric,
            activation_scheme: QuantScheme::Asymmetric,
            symmetric: false,
            ..Self::default()
        }
    }

    /// Create INT4 configuration for aggressive quantization
    pub fn int4() -> Self {
        Self {
            weight_dtype: QuantDtype::Int4,
            activation_dtype: QuantDtype::Int8, // Activations stay INT8
            ..Self::default()
        }
    }
}

/// Quantization state for a single tensor (weights or activations)
#[derive(Clone, Debug)]
pub struct QatQuantState {
    /// Current scale
    pub scale: f32,
    /// Current zero point
    pub zero_point: i32,
    /// Log scale for learnable (for gradient stability)
    pub log_scale: f32,
    /// Running min for online calibration
    pub running_min: f32,
    /// Running max for online calibration
    pub running_max: f32,
    /// Exponential moving average momentum
    pub momentum: f32,
    /// Whether parameters are initialized
    pub initialized: bool,
    /// Quantization dtype
    pub dtype: QuantDtype,
    /// Quantization scheme
    pub scheme: QuantScheme,
}

impl QatQuantState {
    /// Create new state for the given dtype and scheme
    pub fn new(dtype: QuantDtype, scheme: QuantScheme) -> Self {
        Self {
            scale: 1.0,
            zero_point: 0,
            log_scale: 0.0,
            running_min: f32::MAX,
            running_max: f32::MIN,
            momentum: 0.01,
            initialized: false,
            dtype,
            scheme,
        }
    }

    /// Update running statistics with new batch
    pub fn observe(&mut self, values: &[f32]) {
        if values.is_empty() {
            return;
        }

        // Find batch min/max
        let batch_min = values.iter().copied().fold(f32::MAX, f32::min);
        let batch_max = values.iter().copied().fold(f32::MIN, f32::max);

        if !self.initialized {
            self.running_min = batch_min;
            self.running_max = batch_max;
            self.initialized = true;
        } else {
            // Exponential moving average
            self.running_min =
                self.momentum * batch_min + (1.0 - self.momentum) * self.running_min;
            self.running_max =
                self.momentum * batch_max + (1.0 - self.momentum) * self.running_max;
        }

        // Recompute scale/zero_point
        self.update_params();
    }

    /// Update scale and zero_point from running stats
    pub fn update_params(&mut self) {
        if !self.initialized || self.running_min >= self.running_max {
            return;
        }

        let (qmin, qmax) = self.dtype.range();

        match self.scheme {
            QuantScheme::Symmetric => {
                // Symmetric: zero_point = 0, scale = max(|min|, |max|) / qmax
                let abs_max = self.running_min.abs().max(self.running_max.abs());
                self.scale = if abs_max > 0.0 {
                    abs_max / (qmax as f32)
                } else {
                    1.0
                };
                self.zero_point = 0;
            }
            QuantScheme::Asymmetric => {
                // Asymmetric: full range mapping
                let range = self.running_max - self.running_min;
                if range > 0.0 {
                    self.scale = range / ((qmax - qmin) as f32);
                    self.zero_point =
                        (qmin as f32 - self.running_min / self.scale).round() as i32;
                    // Clamp zero_point to valid range
                    self.zero_point = self.zero_point.clamp(qmin, qmax);
                } else {
                    self.scale = 1.0;
                    self.zero_point = 0;
                }
            }
            _ => {
                // Per-channel handled separately
                self.scale = 1.0;
                self.zero_point = 0;
            }
        }

        self.log_scale = self.scale.ln();
    }

    /// Apply fake quantization to values
    ///
    /// Forward: output = dequant(quant(input))
    /// This simulates quantization during training.
    pub fn fake_quantize(&self, values: &[f32]) -> Vec<f32> {
        let (qmin, qmax) = self.dtype.range();
        let scale = if self.scale > 0.0 { self.scale } else { 1.0 };

        values
            .iter()
            .map(|&v| {
                // Quantize
                let q = (v / scale + self.zero_point as f32)
                    .round()
                    .clamp(qmin as f32, qmax as f32) as i32;
                // Dequantize
                (q - self.zero_point) as f32 * scale
            })
            .collect()
    }

    /// Get quantization parameters for export
    pub fn get_params(&self) -> (f32, i32) {
        (self.scale, self.zero_point)
    }
}

/// QAT layer type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QatLayerType {
    Linear,
    Conv2d,
    QuatLinear,
    QuatConv2d,
    BatchNorm,
    LayerNorm,
}

/// QAT-wrapped layer that applies fake quantization
#[derive(Clone, Debug)]
pub struct QatLayer {
    /// Layer name
    pub name: String,
    /// Layer type
    pub layer_type: QatLayerType,
    /// Weight quantization parameters
    pub weight_quant: QatQuantState,
    /// Input activation quantization
    pub input_quant: QatQuantState,
    /// Output activation quantization
    pub output_quant: QatQuantState,
    /// Training mode
    pub training: bool,
    /// Number of batches seen
    pub batches_seen: usize,
    /// Configuration
    pub config: QatConfig,
}

impl QatLayer {
    /// Create QAT-wrapped linear layer
    pub fn linear(name: &str, config: QatConfig) -> Self {
        Self {
            name: name.to_string(),
            layer_type: QatLayerType::Linear,
            weight_quant: QatQuantState::new(config.weight_dtype, config.weight_scheme),
            input_quant: QatQuantState::new(config.activation_dtype, config.activation_scheme),
            output_quant: QatQuantState::new(config.activation_dtype, config.activation_scheme),
            training: true,
            batches_seen: 0,
            config,
        }
    }

    /// Create QAT-wrapped quaternion linear layer
    pub fn quat_linear(name: &str, config: QatConfig) -> Self {
        Self {
            name: name.to_string(),
            layer_type: QatLayerType::QuatLinear,
            weight_quant: QatQuantState::new(config.weight_dtype, config.weight_scheme),
            input_quant: QatQuantState::new(config.activation_dtype, config.activation_scheme),
            output_quant: QatQuantState::new(config.activation_dtype, config.activation_scheme),
            training: true,
            batches_seen: 0,
            config,
        }
    }

    /// Check if in warmup phase
    pub fn in_warmup(&self) -> bool {
        self.batches_seen < self.config.warmup_batches
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Process a batch (call during forward pass)
    ///
    /// During warmup: collect statistics only
    /// After warmup: apply fake quantization
    pub fn forward_weights(&mut self, weights: &[f32]) -> Vec<f32> {
        self.batches_seen += 1;

        // Observe weight statistics
        self.weight_quant.observe(weights);

        // Apply fake quantization after warmup
        if self.in_warmup() || !self.training {
            weights.to_vec()
        } else {
            self.weight_quant.fake_quantize(weights)
        }
    }

    /// Process input activations
    pub fn forward_input(&mut self, inputs: &[f32]) -> Vec<f32> {
        // Observe input statistics
        self.input_quant.observe(inputs);

        // Apply fake quantization after warmup
        if self.in_warmup() || !self.training {
            inputs.to_vec()
        } else {
            self.input_quant.fake_quantize(inputs)
        }
    }

    /// Process output activations
    pub fn forward_output(&mut self, outputs: &[f32]) -> Vec<f32> {
        // Observe output statistics
        self.output_quant.observe(outputs);

        // Apply fake quantization after warmup
        if self.in_warmup() || !self.training {
            outputs.to_vec()
        } else {
            self.output_quant.fake_quantize(outputs)
        }
    }

    /// Export quantization parameters for deployment
    pub fn export_params(&self) -> QatExportedParams {
        QatExportedParams {
            name: self.name.clone(),
            layer_type: self.layer_type,
            weight_scale: self.weight_quant.scale,
            weight_zero_point: self.weight_quant.zero_point,
            input_scale: self.input_quant.scale,
            input_zero_point: self.input_quant.zero_point,
            output_scale: self.output_quant.scale,
            output_zero_point: self.output_quant.zero_point,
            dtype: self.config.weight_dtype,
        }
    }
}

/// Exported quantization parameters for a layer
#[derive(Clone, Debug)]
pub struct QatExportedParams {
    pub name: String,
    pub layer_type: QatLayerType,
    pub weight_scale: f32,
    pub weight_zero_point: i32,
    pub input_scale: f32,
    pub input_zero_point: i32,
    pub output_scale: f32,
    pub output_zero_point: i32,
    pub dtype: QuantDtype,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qat_config_default() {
        let config = QatConfig::default();
        assert_eq!(config.weight_dtype, QuantDtype::Int8);
        assert!(config.symmetric);
        assert_eq!(config.warmup_batches, 1000);
    }

    #[test]
    fn test_quant_state_observe() {
        let mut state = QatQuantState::new(QuantDtype::Int8, QuantScheme::Symmetric);
        assert!(!state.initialized);

        state.observe(&[1.0, 2.0, 3.0, 4.0]);
        assert!(state.initialized);
        assert!(state.running_max >= 4.0);
    }

    #[test]
    fn test_fake_quantize() {
        let mut state = QatQuantState::new(QuantDtype::Int8, QuantScheme::Symmetric);
        state.observe(&[-1.0, 0.0, 1.0]);

        let result = state.fake_quantize(&[0.5]);
        // Result should be close to 0.5 after quantize-dequantize
        assert!((result[0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_qat_layer_warmup() {
        let config = QatConfig {
            warmup_batches: 2,
            ..QatConfig::default()
        };
        let mut layer = QatLayer::linear("test", config);

        assert!(layer.in_warmup());
        layer.forward_weights(&[1.0, 2.0]);
        assert!(layer.in_warmup());
        layer.forward_weights(&[1.0, 2.0]);
        assert!(!layer.in_warmup()); // After 2 batches
    }
}
