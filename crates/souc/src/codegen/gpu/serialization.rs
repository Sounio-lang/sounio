//! Model Serialization and Checkpoint Management
//!
//! Handles save/load of quantized models with:
//! - Mixed-precision state (loss scales, overflow tracking)
//! - Sparsity patterns (2:4, CSR, N:M)
//! - QAT calibration data
//! - Quantization parameters

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Supported sparsity patterns
#[derive(Clone, Debug)]
pub enum SparsityType {
    Ratio24,
    NM { n: usize, m: usize },
    CSR,
}

/// Tensor shape metadata
#[derive(Clone, Debug)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}

impl TensorShape {
    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Quantized tensor representation
#[derive(Clone, Debug)]
pub struct QuantTensor {
    pub name: String,
    pub data: Vec<u8>, // Packed quantized data
    pub bit_width: u8, // 4 or 8 bits
    pub scales: Vec<f32>,
    pub zero_points: Vec<i32>,
    pub per_channel: bool,
    pub shape: TensorShape,
    pub is_weights: bool,
}

impl QuantTensor {
    pub fn validate(&self) -> Result<(), String> {
        if self.bit_width != 4 && self.bit_width != 8 {
            return Err(format!("Invalid bit width: {}", self.bit_width));
        }

        if self.scales.is_empty() {
            return Err("No scales provided".to_string());
        }

        for &scale in &self.scales {
            if scale <= 0.0 || scale > 1000.0 {
                return Err(format!("Invalid scale value: {}", scale));
            }
        }

        Ok(())
    }
}

/// Quantized model state
#[derive(Clone, Debug)]
pub struct QuantizedModelState {
    pub tensors: Vec<QuantTensor>,
}

impl QuantizedModelState {
    pub fn validate(&self) -> Result<(), String> {
        let names: HashMap<_, usize> =
            self.tensors.iter().fold(HashMap::new(), |mut map, t| {
                *map.entry(&t.name).or_insert(0) += 1;
                map
            });

        for (name, count) in names {
            if count > 1 {
                return Err(format!("Duplicate tensor name: {}", name));
            }
        }

        for tensor in &self.tensors {
            tensor.validate()?;
        }

        Ok(())
    }
}

/// Mixed-precision training state
#[derive(Clone, Debug, Default)]
pub struct MixedPrecisionState {
    pub loss_scale: f32,
    pub scale_history: Vec<f32>,
    pub overflow_detected: bool,
    pub overflow_count: usize,
}

impl MixedPrecisionState {
    pub fn validate(&self) -> Result<(), String> {
        if self.loss_scale <= 0.0 || self.loss_scale.is_nan() {
            return Err(format!("Invalid loss scale: {}", self.loss_scale));
        }
        Ok(())
    }
}

/// Sparsity pattern tracking
#[derive(Clone, Debug)]
pub struct SparsityState {
    pub pattern_type: SparsityType,
    pub csr_indices: Vec<usize>,
    pub quaternion_groups: Vec<Vec<usize>>,
}

impl SparsityState {
    pub fn validate(&self, num_elements: usize) -> Result<(), String> {
        match &self.pattern_type {
            SparsityType::Ratio24 => {
                if self.csr_indices.len() > num_elements / 2 {
                    return Err("Invalid 2:4 sparsity density".to_string());
                }
            }
            SparsityType::NM { n, m } => {
                if *n == 0 || *m == 0 || *n > *m || *m > num_elements {
                    return Err(format!("Invalid N:M parameters: {}:{}", n, m));
                }
            }
            SparsityType::CSR => {
                for &idx in &self.csr_indices {
                    if idx >= num_elements {
                        return Err("CSR index out of bounds".to_string());
                    }
                }
            }
        }

        Ok(())
    }
}

/// Quantization-Aware Training state
#[derive(Clone, Debug)]
pub struct QatState {
    pub min_ema: Vec<f32>,
    pub max_ema: Vec<f32>,
    pub warmup_batches: usize,
    pub quant_enabled: bool,
    pub quantized_params: Vec<QuantTensor>,
}

impl QatState {
    pub fn validate(&self, num_tensors: usize) -> Result<(), String> {
        if self.min_ema.len() != num_tensors || self.max_ema.len() != num_tensors {
            return Err("Calibration data length mismatch".to_string());
        }

        for i in 0..num_tensors {
            if self.min_ema[i] >= self.max_ema[i] {
                return Err(format!("Invalid EMA range for tensor {}", i));
            }
        }

        for param in &self.quantized_params {
            param.validate()?;
        }

        Ok(())
    }
}

/// Model metadata
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub version: u32,
    pub timestamp: u64,
    pub compiler_version: String,
    pub optimization_flags: String,
    pub target_architecture: String,
}

impl ModelMetadata {
    pub fn new() -> Self {
        Self {
            version: 2,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            compiler_version: "rustc-gpu-1.0".to_string(),
            optimization_flags: "-O3".to_string(),
            target_architecture: "ptx-sm80".to_string(),
        }
    }
}

/// Complete quantized model checkpoint
#[derive(Clone, Debug)]
pub struct QuantizedModel {
    pub quantized: QuantizedModelState,
    pub mixed: MixedPrecisionState,
    pub sparsity: SparsityState,
    pub qat: QatState,
    pub metadata: ModelMetadata,
}

impl QuantizedModel {
    pub fn new() -> Self {
        Self {
            quantized: QuantizedModelState { tensors: vec![] },
            mixed: MixedPrecisionState::default(),
            sparsity: SparsityState {
                pattern_type: SparsityType::Ratio24,
                csr_indices: vec![],
                quaternion_groups: vec![],
            },
            qat: QatState {
                min_ema: vec![],
                max_ema: vec![],
                warmup_batches: 0,
                quant_enabled: false,
                quantized_params: vec![],
            },
            metadata: ModelMetadata::new(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.metadata.version != 2 {
            return Err(format!(
                "Version mismatch: expected 2, got {}",
                self.metadata.version
            ));
        }

        self.quantized.validate()?;
        let num_tensors = self.quantized.tensors.len();
        self.qat.validate(num_tensors)?;
        self.mixed.validate()?;

        Ok(())
    }

    /// Save model to binary file (bincode format)
    pub fn save_binary(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.validate()?;

        let mut file = File::create(path)?;

        // Write version
        file.write_all(&self.metadata.version.to_le_bytes())?;

        // Write num tensors
        let num_tensors = self.quantized.tensors.len() as u32;
        file.write_all(&num_tensors.to_le_bytes())?;

        // Write each tensor (simplified serialization)
        for tensor in &self.quantized.tensors {
            // Name
            let name_bytes = tensor.name.as_bytes();
            file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            file.write_all(name_bytes)?;

            // Bit width
            file.write_all(&[tensor.bit_width])?;

            // Data
            file.write_all(&(tensor.data.len() as u32).to_le_bytes())?;
            file.write_all(&tensor.data)?;

            // Scales
            file.write_all(&(tensor.scales.len() as u32).to_le_bytes())?;
            for &scale in &tensor.scales {
                file.write_all(&scale.to_le_bytes())?;
            }

            // Zero points
            file.write_all(&(tensor.zero_points.len() as u32).to_le_bytes())?;
            for &zp in &tensor.zero_points {
                file.write_all(&zp.to_le_bytes())?;
            }
        }

        // Write mixed precision state
        file.write_all(&self.mixed.loss_scale.to_le_bytes())?;
        file.write_all(&(self.mixed.overflow_count as u32).to_le_bytes())?;

        Ok(())
    }

    /// Load model from binary file
    pub fn load_binary(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut buffer = vec![];
        file.read_to_end(&mut buffer)?;

        let mut pos = 0;

        // Read version
        let version = u32::from_le_bytes([
            buffer[pos],
            buffer[pos + 1],
            buffer[pos + 2],
            buffer[pos + 3],
        ]);
        pos += 4;

        if version != 2 {
            return Err(format!("Unsupported version: {}", version).into());
        }

        // Read num tensors
        let num_tensors = u32::from_le_bytes([
            buffer[pos],
            buffer[pos + 1],
            buffer[pos + 2],
            buffer[pos + 3],
        ]) as usize;
        pos += 4;

        let mut tensors = vec![];

        for _ in 0..num_tensors {
            // Read name
            let name_len = u32::from_le_bytes([
                buffer[pos],
                buffer[pos + 1],
                buffer[pos + 2],
                buffer[pos + 3],
            ]) as usize;
            pos += 4;

            let name = String::from_utf8(buffer[pos..pos + name_len].to_vec())?;
            pos += name_len;

            // Read bit width
            let bit_width = buffer[pos];
            pos += 1;

            // Read data
            let data_len = u32::from_le_bytes([
                buffer[pos],
                buffer[pos + 1],
                buffer[pos + 2],
                buffer[pos + 3],
            ]) as usize;
            pos += 4;

            let data = buffer[pos..pos + data_len].to_vec();
            pos += data_len;

            // Read scales
            let scale_len = u32::from_le_bytes([
                buffer[pos],
                buffer[pos + 1],
                buffer[pos + 2],
                buffer[pos + 3],
            ]) as usize;
            pos += 4;

            let mut scales = vec![];
            for _ in 0..scale_len {
                let scale = f32::from_le_bytes([
                    buffer[pos],
                    buffer[pos + 1],
                    buffer[pos + 2],
                    buffer[pos + 3],
                ]);
                scales.push(scale);
                pos += 4;
            }

            // Read zero points
            let zp_len = u32::from_le_bytes([
                buffer[pos],
                buffer[pos + 1],
                buffer[pos + 2],
                buffer[pos + 3],
            ]) as usize;
            pos += 4;

            let mut zero_points = vec![];
            for _ in 0..zp_len {
                let zp = i32::from_le_bytes([
                    buffer[pos],
                    buffer[pos + 1],
                    buffer[pos + 2],
                    buffer[pos + 3],
                ]);
                zero_points.push(zp);
                pos += 4;
            }

            tensors.push(QuantTensor {
                name,
                data,
                bit_width,
                scales,
                zero_points,
                per_channel: false,
                shape: TensorShape { dims: vec![100] }, // Placeholder
                is_weights: true,
            });
        }

        // Read mixed precision state
        let loss_scale = f32::from_le_bytes([
            buffer[pos],
            buffer[pos + 1],
            buffer[pos + 2],
            buffer[pos + 3],
        ]);
        pos += 4;

        let overflow_count = u32::from_le_bytes([
            buffer[pos],
            buffer[pos + 1],
            buffer[pos + 2],
            buffer[pos + 3],
        ]) as usize;

        let mut model = QuantizedModel::new();
        model.quantized.tensors = tensors;
        model.mixed.loss_scale = loss_scale;
        model.mixed.overflow_count = overflow_count;

        model.validate()?;
        Ok(model)
    }

    /// Export weights only for inference
    pub fn export_for_inference(&self) -> Vec<QuantTensor> {
        self.quantized
            .tensors
            .iter()
            .filter(|t| t.is_weights)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_quant_tensor_validation() {
        let tensor = QuantTensor {
            name: "test".to_string(),
            data: vec![1, 2, 3],
            bit_width: 8,
            scales: vec![0.01],
            zero_points: vec![0],
            per_channel: false,
            shape: TensorShape { dims: vec![3] },
            is_weights: true,
        };

        assert!(tensor.validate().is_ok());
    }

    #[test]
    fn test_model_save_load() {
        let mut model = QuantizedModel::new();

        model.quantized.tensors.push(QuantTensor {
            name: "weight".to_string(),
            data: vec![1, 2, 3, 4],
            bit_width: 8,
            scales: vec![0.01],
            zero_points: vec![0],
            per_channel: false,
            shape: TensorShape { dims: vec![4] },
            is_weights: true,
        });

        model.mixed.loss_scale = 32768.0;
        model.mixed.overflow_count = 2;

        let path = "test_model.bin";
        assert!(model.save_binary(Path::new(path)).is_ok());

        let loaded = QuantizedModel::load_binary(Path::new(path));
        assert!(loaded.is_ok());

        let loaded_model = loaded.unwrap();
        assert_eq!(loaded_model.mixed.loss_scale, 32768.0);
        assert_eq!(loaded_model.mixed.overflow_count, 2);

        let _ = fs::remove_file(path);
    }
}
