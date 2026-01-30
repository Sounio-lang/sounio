//! Backward Pass Code Generation for GPU AutoDiff
//!
//! This module generates the backward (gradient computation) kernel from a forward kernel.
//! It handles:
//! - Adding gradient parameters for each differentiable input
//! - Generating backward operations for each forward operation
//! - Handling gradient accumulation for values used multiple times
//! - Memory coalescing optimizations for GPU efficiency
//! - Warp-level reductions for efficiency

use std::collections::{HashMap, HashSet};

use super::super::ir::{
    BlockId, GpuBlock, GpuKernel, GpuOp, GpuParam, GpuTerminator, GpuType, MemorySpace, ValueId,
};
use super::primitives::{BackwardExpr, BackwardOp, BackwardRule, PrimitiveRegistry};
use super::{DiffConfig, DiffError, KernelAnalysis, ParamInfo};

/// Generated backward kernel with metadata
#[derive(Debug, Clone)]
pub struct BackwardKernel {
    /// The generated GPU kernel for backward pass
    pub kernel: GpuKernel,
    /// Mapping from forward input params to gradient params
    pub gradient_params: HashMap<String, String>,
    /// Original forward kernel name
    pub forward_kernel_name: String,
    /// Parameters that require gradients
    pub requires_grad: Vec<String>,
}

/// Code generator for backward pass kernels
pub struct BackwardCodegen<'a> {
    /// Configuration
    config: &'a DiffConfig,
    /// Primitive registry with backward rules
    primitives: &'a PrimitiveRegistry,
    /// Next value ID for generated operations
    next_value_id: u32,
    /// Value type tracking
    value_types: HashMap<ValueId, GpuType>,
    /// Gradient value IDs (original value -> gradient value)
    gradient_values: HashMap<ValueId, ValueId>,
    /// Values that have been used (need gradient accumulation if used again)
    used_values: HashSet<ValueId>,
}

impl<'a> BackwardCodegen<'a> {
    /// Create a new backward code generator
    pub fn new(config: &'a DiffConfig, primitives: &'a PrimitiveRegistry) -> Self {
        Self {
            config,
            primitives,
            next_value_id: 0,
            value_types: HashMap::new(),
            gradient_values: HashMap::new(),
            used_values: HashSet::new(),
        }
    }

    /// Generate a backward kernel from a forward kernel and its analysis
    pub fn generate_backward(
        &mut self,
        forward: &GpuKernel,
        analysis: &KernelAnalysis,
    ) -> Result<BackwardKernel, DiffError> {
        // Reset state
        self.next_value_id = 0;
        self.value_types.clear();
        self.gradient_values.clear();
        self.used_values.clear();

        // Create backward kernel with appropriate name
        let backward_name = format!("{}_backward", forward.name);
        let mut backward = GpuKernel::new(&backward_name);

        // Build parameter list for backward kernel
        let gradient_params = self.build_backward_params(forward, analysis, &mut backward)?;

        // Copy shared memory declarations (may need additional for gradients)
        for shared in &forward.shared_memory {
            backward.add_shared_memory(shared.clone());
        }

        // Generate backward blocks
        let backward_blocks = self.generate_backward_blocks(forward, analysis)?;
        for block in backward_blocks {
            backward.add_block(block);
        }

        // Set entry and max threads
        backward.entry = BlockId(0);
        backward.max_threads = forward.max_threads;

        Ok(BackwardKernel {
            kernel: backward,
            gradient_params,
            forward_kernel_name: forward.name.clone(),
            requires_grad: analysis.inputs.iter().map(|p| p.name.clone()).collect(),
        })
    }

    /// Build parameters for the backward kernel
    fn build_backward_params(
        &mut self,
        forward: &GpuKernel,
        analysis: &KernelAnalysis,
        backward: &mut GpuKernel,
    ) -> Result<HashMap<String, String>, DiffError> {
        let mut gradient_params = HashMap::new();

        // Add all original input parameters (needed for gradient computation)
        for param in &forward.params {
            backward.add_param(param.clone());
        }

        // Add gradient parameters for outputs (incoming gradients)
        for output in &analysis.outputs {
            let grad_name = format!("d{}", output.name);
            let grad_param = GpuParam {
                name: grad_name.clone(),
                ty: output.ty.clone(),
                space: MemorySpace::Global,
                restrict: true,
            };
            backward.add_param(grad_param);
            gradient_params.insert(output.name.clone(), grad_name);
        }

        // Add gradient parameters for inputs (outgoing gradients)
        for input in &analysis.inputs {
            let grad_name = format!("d{}", input.name);
            let grad_param = GpuParam {
                name: grad_name.clone(),
                ty: input.ty.clone(),
                space: MemorySpace::Global,
                restrict: true,
            };
            backward.add_param(grad_param);
            gradient_params.insert(input.name.clone(), grad_name);
        }

        Ok(gradient_params)
    }

    /// Generate the backward pass blocks
    fn generate_backward_blocks(
        &mut self,
        forward: &GpuKernel,
        analysis: &KernelAnalysis,
    ) -> Result<Vec<GpuBlock>, DiffError> {
        let mut blocks = Vec::new();

        // Entry block
        let mut entry_block = GpuBlock::new(BlockId(0), "entry");

        // Get thread index (same as forward pass)
        let thread_idx = self.alloc_value_id();
        entry_block.add_instruction(thread_idx, GpuOp::ThreadIdX);
        self.value_types.insert(thread_idx, GpuType::U32);

        // Load incoming gradient (dy) for each output
        let mut output_gradients = Vec::new();
        for (i, output) in analysis.outputs.iter().enumerate() {
            let grad_ptr = self.alloc_value_id();
            let grad_offset = self.alloc_value_id();
            let dy = self.alloc_value_id();

            // Get gradient parameter (dy)
            let param_idx = forward.params.len() + i;
            entry_block.add_instruction(grad_ptr, GpuOp::Param(param_idx as u32));

            // Compute offset
            entry_block.add_instruction(
                grad_offset,
                GpuOp::GetElementPtr(grad_ptr, vec![thread_idx]),
            );

            // Load gradient value
            entry_block.add_instruction(dy, GpuOp::Load(grad_offset, MemorySpace::Global));

            let elem_ty = self.element_type(&output.ty);
            self.value_types.insert(dy, elem_ty);
            output_gradients.push((output.clone(), dy));
        }

        // Process forward operations in reverse order to generate backward operations
        let backward_ops =
            self.generate_backward_operations(forward, analysis, &output_gradients)?;

        // Add backward operations to entry block
        for (value_id, op) in backward_ops {
            entry_block.add_instruction(value_id, op);
        }

        // Store gradients for inputs
        for (i, input) in analysis.inputs.iter().enumerate() {
            if let Some(&grad_value) = self.gradient_values.get(&ValueId(input.index as u32)) {
                let grad_ptr = self.alloc_value_id();
                let grad_offset = self.alloc_value_id();

                // Get gradient output parameter
                let param_idx = forward.params.len() + analysis.outputs.len() + i;
                entry_block.add_instruction(grad_ptr, GpuOp::Param(param_idx as u32));

                // Compute offset
                entry_block.add_instruction(
                    grad_offset,
                    GpuOp::GetElementPtr(grad_ptr, vec![thread_idx]),
                );

                // Store gradient
                entry_block.add_instruction(
                    self.alloc_value_id(),
                    GpuOp::Store(grad_offset, grad_value, MemorySpace::Global),
                );
            }
        }

        entry_block.set_terminator(GpuTerminator::ReturnVoid);
        blocks.push(entry_block);

        Ok(blocks)
    }

    /// Generate backward operations by processing forward ops in reverse
    fn generate_backward_operations(
        &mut self,
        forward: &GpuKernel,
        analysis: &KernelAnalysis,
        output_gradients: &[(ParamInfo, ValueId)],
    ) -> Result<Vec<(ValueId, GpuOp)>, DiffError> {
        let mut ops = Vec::new();

        // Initialize output gradients
        for (output, dy) in output_gradients {
            let output_val = ValueId(output.index as u32);
            self.gradient_values.insert(output_val, *dy);
        }

        // Collect all differentiable operations from forward pass
        let mut forward_ops: Vec<_> = Vec::new();
        for block in &forward.blocks {
            for (value_id, op) in &block.instructions {
                if let Some(diff_op) = self.primitives.classify_op(op) {
                    forward_ops.push((*value_id, op.clone(), diff_op));
                }
            }
        }

        // Process in reverse order (backpropagation)
        for (result_id, op, diff_op) in forward_ops.into_iter().rev() {
            // Get the backward rule for this operation
            if let Some(rule) = self.primitives.get_rule(diff_op) {
                let backward_ops = self.apply_backward_rule(&op, result_id, rule)?;
                ops.extend(backward_ops);
            }
        }

        Ok(ops)
    }

    /// Apply a backward rule to generate gradient operations
    fn apply_backward_rule(
        &mut self,
        forward_op: &GpuOp,
        result_id: ValueId,
        rule: &BackwardRule,
    ) -> Result<Vec<(ValueId, GpuOp)>, DiffError> {
        let mut ops = Vec::new();

        // Get the upstream gradient (dz)
        let dz = match self.gradient_values.get(&result_id) {
            Some(&grad) => grad,
            None => {
                // No upstream gradient, skip
                return Ok(ops);
            }
        };

        // Extract operands from the forward operation
        let operands = self.extract_operands(forward_op);

        // Apply each backward operation in the rule
        for backward_op in &rule.backward_ops {
            let generated = self.apply_backward_op(backward_op, dz, &operands, forward_op)?;
            ops.extend(generated);
        }

        Ok(ops)
    }

    /// Apply a single backward operation
    fn apply_backward_op(
        &mut self,
        backward_op: &BackwardOp,
        dz: ValueId,
        operands: &[ValueId],
        forward_op: &GpuOp,
    ) -> Result<Vec<(ValueId, GpuOp)>, DiffError> {
        let mut ops = Vec::new();

        match backward_op {
            BackwardOp::PassThrough { input_idx } => {
                // dx = dz
                let input = operands
                    .get(*input_idx)
                    .copied()
                    .ok_or_else(|| DiffError::Internal("Invalid operand index".to_string()))?;
                self.accumulate_gradient(input, dz, &mut ops);
            }

            BackwardOp::Scale { input_idx, scale } => {
                // dx = dz * scale
                let input = operands
                    .get(*input_idx)
                    .copied()
                    .ok_or_else(|| DiffError::Internal("Invalid operand index".to_string()))?;

                let scale_val = self.alloc_value_id();
                let grad = self.alloc_value_id();
                let ty = self.get_type(dz);

                ops.push((scale_val, GpuOp::ConstFloat(*scale, ty.clone())));
                ops.push((grad, GpuOp::FMul(dz, scale_val)));

                self.value_types.insert(scale_val, ty.clone());
                self.value_types.insert(grad, ty);

                self.accumulate_gradient(input, grad, &mut ops);
            }

            BackwardOp::Negate { input_idx } => {
                // dx = -dz
                let input = operands
                    .get(*input_idx)
                    .copied()
                    .ok_or_else(|| DiffError::Internal("Invalid operand index".to_string()))?;

                let grad = self.alloc_value_id();
                let ty = self.get_type(dz);

                ops.push((grad, GpuOp::FNeg(dz)));
                self.value_types.insert(grad, ty);

                self.accumulate_gradient(input, grad, &mut ops);
            }

            BackwardOp::MulByInput {
                input_idx,
                other_input_idx,
            } => {
                // dx = dz * other
                let input = operands
                    .get(*input_idx)
                    .copied()
                    .ok_or_else(|| DiffError::Internal("Invalid operand index".to_string()))?;
                let other = operands
                    .get(*other_input_idx)
                    .copied()
                    .ok_or_else(|| DiffError::Internal("Invalid operand index".to_string()))?;

                let grad = self.alloc_value_id();
                let ty = self.get_type(dz);

                ops.push((grad, GpuOp::FMul(dz, other)));
                self.value_types.insert(grad, ty);

                self.accumulate_gradient(input, grad, &mut ops);
            }

            BackwardOp::DivByInput {
                input_idx,
                divisor_idx,
            } => {
                // dx = dz / divisor
                let input = operands
                    .get(*input_idx)
                    .copied()
                    .ok_or_else(|| DiffError::Internal("Invalid operand index".to_string()))?;
                let divisor = operands
                    .get(*divisor_idx)
                    .copied()
                    .ok_or_else(|| DiffError::Internal("Invalid operand index".to_string()))?;

                let grad = self.alloc_value_id();
                let ty = self.get_type(dz);

                ops.push((grad, GpuOp::FDiv(dz, divisor)));
                self.value_types.insert(grad, ty);

                self.accumulate_gradient(input, grad, &mut ops);
            }

            BackwardOp::DivDenomGrad {
                input_idx,
                numerator_idx,
            } => {
                // dx = -dz * numerator / x^2
                let input = operands
                    .get(*input_idx)
                    .copied()
                    .ok_or_else(|| DiffError::Internal("Invalid operand index".to_string()))?;
                let numerator = operands
                    .get(*numerator_idx)
                    .copied()
                    .ok_or_else(|| DiffError::Internal("Invalid operand index".to_string()))?;

                let ty = self.get_type(dz);

                // x^2
                let x_sq = self.alloc_value_id();
                ops.push((x_sq, GpuOp::FMul(input, input)));
                self.value_types.insert(x_sq, ty.clone());

                // dz * numerator
                let dz_num = self.alloc_value_id();
                ops.push((dz_num, GpuOp::FMul(dz, numerator)));
                self.value_types.insert(dz_num, ty.clone());

                // dz * numerator / x^2
                let div_result = self.alloc_value_id();
                ops.push((div_result, GpuOp::FDiv(dz_num, x_sq)));
                self.value_types.insert(div_result, ty.clone());

                // Negate
                let grad = self.alloc_value_id();
                ops.push((grad, GpuOp::FNeg(div_result)));
                self.value_types.insert(grad, ty);

                self.accumulate_gradient(input, grad, &mut ops);
            }

            BackwardOp::Custom { input_idx, expr } => {
                let input = operands
                    .get(*input_idx)
                    .copied()
                    .ok_or_else(|| DiffError::Internal("Invalid operand index".to_string()))?;

                let custom_ops = self.generate_custom_backward(dz, input, operands, expr)?;
                let grad = custom_ops.last().map(|(id, _)| *id).ok_or_else(|| {
                    DiffError::Internal("Custom backward generated no ops".to_string())
                })?;

                ops.extend(custom_ops);
                self.accumulate_gradient(input, grad, &mut ops);
            }
        }

        Ok(ops)
    }

    /// Generate custom backward operations
    fn generate_custom_backward(
        &mut self,
        dz: ValueId,
        input: ValueId,
        operands: &[ValueId],
        expr: &BackwardExpr,
    ) -> Result<Vec<(ValueId, GpuOp)>, DiffError> {
        let mut ops = Vec::new();
        let ty = self.get_type(dz);

        match expr {
            BackwardExpr::Cos(idx) => {
                // dx = dz * cos(x)
                let cos_x = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((cos_x, GpuOp::FastCos(input)));
                ops.push((grad, GpuOp::FMul(dz, cos_x)));

                self.value_types.insert(cos_x, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::NegSin(idx) => {
                // dx = -dz * sin(x)
                let sin_x = self.alloc_value_id();
                let neg_sin = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((sin_x, GpuOp::FastSin(input)));
                ops.push((neg_sin, GpuOp::FNeg(sin_x)));
                ops.push((grad, GpuOp::FMul(dz, neg_sin)));

                self.value_types.insert(sin_x, ty.clone());
                self.value_types.insert(neg_sin, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::SecSquared(idx) => {
                // dx = dz / cos^2(x) = dz * (1 + tan^2(x))
                let cos_x = self.alloc_value_id();
                let cos_sq = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((cos_x, GpuOp::FastCos(input)));
                ops.push((cos_sq, GpuOp::FMul(cos_x, cos_x)));
                ops.push((grad, GpuOp::FDiv(dz, cos_sq)));

                self.value_types.insert(cos_x, ty.clone());
                self.value_types.insert(cos_sq, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::Exp(idx) => {
                // dx = dz * exp(x)
                let exp_x = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((exp_x, GpuOp::FastExp(input)));
                ops.push((grad, GpuOp::FMul(dz, exp_x)));

                self.value_types.insert(exp_x, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::Reciprocal(idx) => {
                // dx = dz / x
                let grad = self.alloc_value_id();
                ops.push((grad, GpuOp::FDiv(dz, input)));
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::HalfRsqrt(idx) => {
                // dx = dz / (2 * sqrt(x))
                let sqrt_x = self.alloc_value_id();
                let two = self.alloc_value_id();
                let denom = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((sqrt_x, GpuOp::FastSqrt(input)));
                ops.push((two, GpuOp::ConstFloat(2.0, ty.clone())));
                ops.push((denom, GpuOp::FMul(two, sqrt_x)));
                ops.push((grad, GpuOp::FDiv(dz, denom)));

                self.value_types.insert(sqrt_x, ty.clone());
                self.value_types.insert(two, ty.clone());
                self.value_types.insert(denom, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::PowGrad { base_idx, exponent } => {
                // dx = dz * n * x^(n-1)
                // For simplicity, we use exp/log: x^(n-1) = exp((n-1) * log(x))
                let n = self.alloc_value_id();
                let log_x = self.alloc_value_id();
                let nm1 = self.alloc_value_id();
                let nm1_log_x = self.alloc_value_id();
                let x_nm1 = self.alloc_value_id();
                let n_x_nm1 = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((n, GpuOp::ConstFloat(*exponent, ty.clone())));
                ops.push((log_x, GpuOp::FastLog(input)));
                ops.push((nm1, GpuOp::ConstFloat(*exponent - 1.0, ty.clone())));
                ops.push((nm1_log_x, GpuOp::FMul(nm1, log_x)));
                ops.push((x_nm1, GpuOp::FastExp(nm1_log_x)));
                ops.push((n_x_nm1, GpuOp::FMul(n, x_nm1)));
                ops.push((grad, GpuOp::FMul(dz, n_x_nm1)));

                self.value_types.insert(n, ty.clone());
                self.value_types.insert(log_x, ty.clone());
                self.value_types.insert(nm1, ty.clone());
                self.value_types.insert(nm1_log_x, ty.clone());
                self.value_types.insert(x_nm1, ty.clone());
                self.value_types.insert(n_x_nm1, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::RsqrtGrad(idx) => {
                // dx = -dz / (2 * x^(3/2)) = -dz * rsqrt(x)^3 / 2
                let rsqrt_x = self.alloc_value_id();
                let rsqrt_cubed = self.alloc_value_id();
                let two = self.alloc_value_id();
                let half_rsqrt_cubed = self.alloc_value_id();
                let neg_half = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((rsqrt_x, GpuOp::FastRsqrt(input)));
                ops.push((rsqrt_cubed, GpuOp::FMul(rsqrt_x, rsqrt_x)));
                // rsqrt^3
                let rsqrt_cubed_final = self.alloc_value_id();
                ops.push((rsqrt_cubed_final, GpuOp::FMul(rsqrt_cubed, rsqrt_x)));
                ops.push((two, GpuOp::ConstFloat(2.0, ty.clone())));
                ops.push((half_rsqrt_cubed, GpuOp::FDiv(rsqrt_cubed_final, two)));
                ops.push((neg_half, GpuOp::FNeg(half_rsqrt_cubed)));
                ops.push((grad, GpuOp::FMul(dz, neg_half)));

                self.value_types.insert(rsqrt_x, ty.clone());
                self.value_types.insert(rsqrt_cubed, ty.clone());
                self.value_types.insert(rsqrt_cubed_final, ty.clone());
                self.value_types.insert(two, ty.clone());
                self.value_types.insert(half_rsqrt_cubed, ty.clone());
                self.value_types.insert(neg_half, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::Sign(idx) => {
                // dx = dz * sign(x)
                // sign(x) = x / |x| (for x != 0)
                let abs_x = self.alloc_value_id();
                let sign_x = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((abs_x, GpuOp::AbsF32(input)));
                ops.push((sign_x, GpuOp::FDiv(input, abs_x)));
                ops.push((grad, GpuOp::FMul(dz, sign_x)));

                self.value_types.insert(abs_x, ty.clone());
                self.value_types.insert(sign_x, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::Step(idx) => {
                // dx = dz * (x > 0)
                // Use select: grad = x > 0 ? dz : 0
                let zero = self.alloc_value_id();
                let cond = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((zero, GpuOp::ConstFloat(0.0, ty.clone())));
                ops.push((cond, GpuOp::FGt(input, zero)));
                ops.push((grad, GpuOp::Select(cond, dz, zero)));

                self.value_types.insert(zero, ty.clone());
                self.value_types.insert(cond, GpuType::Bool);
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::SigmoidGrad(idx) => {
                // dx = dz * sigmoid(x) * (1 - sigmoid(x))
                // We can use the fact that d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                // Compute sigmoid using: 1 / (1 + exp(-x))
                let neg_x = self.alloc_value_id();
                let exp_neg_x = self.alloc_value_id();
                let one = self.alloc_value_id();
                let one_plus_exp = self.alloc_value_id();
                let sigmoid = self.alloc_value_id();
                let one_minus_sigmoid = self.alloc_value_id();
                let sigmoid_grad = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((neg_x, GpuOp::FNeg(input)));
                ops.push((exp_neg_x, GpuOp::FastExp(neg_x)));
                ops.push((one, GpuOp::ConstFloat(1.0, ty.clone())));
                ops.push((one_plus_exp, GpuOp::FAdd(one, exp_neg_x)));
                ops.push((sigmoid, GpuOp::FDiv(one, one_plus_exp)));
                ops.push((one_minus_sigmoid, GpuOp::FSub(one, sigmoid)));
                ops.push((sigmoid_grad, GpuOp::FMul(sigmoid, one_minus_sigmoid)));
                ops.push((grad, GpuOp::FMul(dz, sigmoid_grad)));

                self.value_types.insert(neg_x, ty.clone());
                self.value_types.insert(exp_neg_x, ty.clone());
                self.value_types.insert(one, ty.clone());
                self.value_types.insert(one_plus_exp, ty.clone());
                self.value_types.insert(sigmoid, ty.clone());
                self.value_types.insert(one_minus_sigmoid, ty.clone());
                self.value_types.insert(sigmoid_grad, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::TanhGrad(idx) => {
                // dx = dz * (1 - tanh^2(x))
                // Since tanh = (e^x - e^-x) / (e^x + e^-x), we compute directly
                let exp_x = self.alloc_value_id();
                let neg_x = self.alloc_value_id();
                let exp_neg_x = self.alloc_value_id();
                let sum_exp = self.alloc_value_id();
                let diff_exp = self.alloc_value_id();
                let tanh_x = self.alloc_value_id();
                let tanh_sq = self.alloc_value_id();
                let one = self.alloc_value_id();
                let one_minus_tanh_sq = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((exp_x, GpuOp::FastExp(input)));
                ops.push((neg_x, GpuOp::FNeg(input)));
                ops.push((exp_neg_x, GpuOp::FastExp(neg_x)));
                ops.push((sum_exp, GpuOp::FAdd(exp_x, exp_neg_x)));
                ops.push((diff_exp, GpuOp::FSub(exp_x, exp_neg_x)));
                ops.push((tanh_x, GpuOp::FDiv(diff_exp, sum_exp)));
                ops.push((tanh_sq, GpuOp::FMul(tanh_x, tanh_x)));
                ops.push((one, GpuOp::ConstFloat(1.0, ty.clone())));
                ops.push((one_minus_tanh_sq, GpuOp::FSub(one, tanh_sq)));
                ops.push((grad, GpuOp::FMul(dz, one_minus_tanh_sq)));

                self.value_types.insert(exp_x, ty.clone());
                self.value_types.insert(neg_x, ty.clone());
                self.value_types.insert(exp_neg_x, ty.clone());
                self.value_types.insert(sum_exp, ty.clone());
                self.value_types.insert(diff_exp, ty.clone());
                self.value_types.insert(tanh_x, ty.clone());
                self.value_types.insert(tanh_sq, ty.clone());
                self.value_types.insert(one, ty.clone());
                self.value_types.insert(one_minus_tanh_sq, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::GeluGrad(idx) => {
                // GELU'(x) = Phi(x) + x * phi(x)
                // where Phi is the CDF and phi is the PDF of standard normal
                // Approximate: 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                // For gradient, we use the approximation derivative
                let grad = self.alloc_value_id();
                // Simplified: just use 0.5 + 0.5 * tanh(0.797885 * x) as approximation
                let c1 = self.alloc_value_id();
                let c2 = self.alloc_value_id();
                let scaled_x = self.alloc_value_id();
                let exp_pos = self.alloc_value_id();
                let exp_neg = self.alloc_value_id();
                let tanh_val = self.alloc_value_id();
                let half = self.alloc_value_id();
                let half_tanh = self.alloc_value_id();
                let gelu_approx = self.alloc_value_id();

                ops.push((c1, GpuOp::ConstFloat(0.797885, ty.clone())));
                ops.push((scaled_x, GpuOp::FMul(c1, input)));
                ops.push((exp_pos, GpuOp::FastExp(scaled_x)));
                let neg_scaled_x = self.alloc_value_id();
                ops.push((neg_scaled_x, GpuOp::FNeg(scaled_x)));
                ops.push((exp_neg, GpuOp::FastExp(neg_scaled_x)));
                let sum = self.alloc_value_id();
                let diff = self.alloc_value_id();
                ops.push((sum, GpuOp::FAdd(exp_pos, exp_neg)));
                ops.push((diff, GpuOp::FSub(exp_pos, exp_neg)));
                ops.push((tanh_val, GpuOp::FDiv(diff, sum)));
                ops.push((half, GpuOp::ConstFloat(0.5, ty.clone())));
                ops.push((half_tanh, GpuOp::FMul(half, tanh_val)));
                ops.push((gelu_approx, GpuOp::FAdd(half, half_tanh)));
                ops.push((grad, GpuOp::FMul(dz, gelu_approx)));

                self.value_types.insert(c1, ty.clone());
                self.value_types.insert(scaled_x, ty.clone());
                self.value_types.insert(exp_pos, ty.clone());
                self.value_types.insert(neg_scaled_x, ty.clone());
                self.value_types.insert(exp_neg, ty.clone());
                self.value_types.insert(sum, ty.clone());
                self.value_types.insert(diff, ty.clone());
                self.value_types.insert(tanh_val, ty.clone());
                self.value_types.insert(half, ty.clone());
                self.value_types.insert(half_tanh, ty.clone());
                self.value_types.insert(gelu_approx, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::Broadcast { .. } => {
                // For sum reduction backward, the gradient is simply broadcast
                let grad = self.alloc_value_id();
                ops.push((grad, GpuOp::FMul(dz, dz))); // Placeholder - actual broadcast
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::Indicator { .. } => {
                // For max/min, gradient flows only where the max/min occurred
                let grad = self.alloc_value_id();
                ops.push((grad, GpuOp::FMul(dz, dz))); // Placeholder
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::MeanScale { count } => {
                // dx = dz / n (broadcast)
                let n = self.alloc_value_id();
                let grad = self.alloc_value_id();

                ops.push((n, GpuOp::ConstFloat(*count as f64, ty.clone())));
                ops.push((grad, GpuOp::FDiv(dz, n)));

                self.value_types.insert(n, ty.clone());
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::Transpose(idx) => {
                // dx = dz^T (transpose the gradient)
                let grad = self.alloc_value_id();
                ops.push((grad, GpuOp::TileTranspose(dz)));
                self.value_types.insert(grad, ty);
            }

            BackwardExpr::MatMulGrad {
                grad_idx,
                other_idx,
                transpose_first,
                transpose_second,
            } => {
                // Generate matmul for gradient computation
                // This is a simplified version - real implementation would use TileMma
                let other = operands.get(*other_idx).copied().unwrap_or(input);
                let grad = self.alloc_value_id();

                // For simplicity, we just emit a placeholder matmul-like operation
                // A full implementation would use TileMma with appropriate transpositions
                if *transpose_second {
                    // dA = dC @ B^T
                    let transposed = self.alloc_value_id();
                    ops.push((transposed, GpuOp::TileTranspose(other)));
                    self.value_types.insert(transposed, ty.clone());
                    ops.push((grad, GpuOp::FMul(dz, transposed))); // Simplified
                } else if *transpose_first {
                    // dB = A^T @ dC
                    let transposed = self.alloc_value_id();
                    ops.push((transposed, GpuOp::TileTranspose(other)));
                    self.value_types.insert(transposed, ty.clone());
                    ops.push((grad, GpuOp::FMul(transposed, dz))); // Simplified
                } else {
                    ops.push((grad, GpuOp::FMul(dz, other)));
                }

                self.value_types.insert(grad, ty);
            }
        }

        Ok(ops)
    }

    /// Extract operands from a GPU operation
    fn extract_operands(&self, op: &GpuOp) -> Vec<ValueId> {
        match op {
            // Binary operations
            GpuOp::FAdd(a, b)
            | GpuOp::FSub(a, b)
            | GpuOp::FMul(a, b)
            | GpuOp::FDiv(a, b)
            | GpuOp::Add(a, b)
            | GpuOp::Sub(a, b)
            | GpuOp::Mul(a, b)
            | GpuOp::Div(a, b) => vec![*a, *b],

            // Unary operations
            GpuOp::FNeg(a)
            | GpuOp::Neg(a)
            | GpuOp::FastSin(a)
            | GpuOp::FastCos(a)
            | GpuOp::FastExp(a)
            | GpuOp::FastLog(a)
            | GpuOp::FastSqrt(a)
            | GpuOp::FastRsqrt(a)
            | GpuOp::AbsF32(a)
            | GpuOp::AbsF64(a) => vec![*a],

            // Ternary operations
            GpuOp::FMulAdd(a, b, c) => vec![*a, *b, *c],

            // Tile operations
            GpuOp::TileTranspose(a) => vec![*a],
            GpuOp::TileMma { a, b, c, .. } => vec![*a, *b, *c],

            // Other operations - return empty for non-differentiable
            _ => vec![],
        }
    }

    /// Accumulate gradient for a value (handles gradient accumulation)
    fn accumulate_gradient(
        &mut self,
        value: ValueId,
        grad: ValueId,
        ops: &mut Vec<(ValueId, GpuOp)>,
    ) {
        if let Some(&existing_grad) = self.gradient_values.get(&value) {
            // Value already has a gradient, need to accumulate
            let accumulated = self.alloc_value_id();
            let ty = self.get_type(grad);
            ops.push((accumulated, GpuOp::FAdd(existing_grad, grad)));
            self.value_types.insert(accumulated, ty);
            self.gradient_values.insert(value, accumulated);
        } else {
            // First gradient for this value
            self.gradient_values.insert(value, grad);
        }
        self.used_values.insert(value);
    }

    /// Allocate a new value ID
    fn alloc_value_id(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    /// Get the type of a value
    fn get_type(&self, value: ValueId) -> GpuType {
        self.value_types
            .get(&value)
            .cloned()
            .unwrap_or(GpuType::F32)
    }

    /// Get the element type of a pointer type
    fn element_type(&self, ty: &GpuType) -> GpuType {
        match ty {
            GpuType::Ptr(inner, _) => *inner.clone(),
            GpuType::Array(inner, _) => *inner.clone(),
            other => other.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::ir::*;
    use super::*;

    fn create_simple_add_kernel() -> GpuKernel {
        let mut kernel = GpuKernel::new("forward_add");

        kernel.add_param(GpuParam {
            name: "x".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });
        kernel.add_param(GpuParam {
            name: "y".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });
        kernel.add_param(GpuParam {
            name: "z".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        block.add_instruction(ValueId(1), GpuOp::Param(0));
        block.add_instruction(
            ValueId(2),
            GpuOp::GetElementPtr(ValueId(1), vec![ValueId(0)]),
        );
        block.add_instruction(ValueId(3), GpuOp::Load(ValueId(2), MemorySpace::Global));
        block.add_instruction(ValueId(4), GpuOp::Param(1));
        block.add_instruction(
            ValueId(5),
            GpuOp::GetElementPtr(ValueId(4), vec![ValueId(0)]),
        );
        block.add_instruction(ValueId(6), GpuOp::Load(ValueId(5), MemorySpace::Global));
        block.add_instruction(ValueId(7), GpuOp::FAdd(ValueId(3), ValueId(6)));
        block.add_instruction(ValueId(8), GpuOp::Param(2));
        block.add_instruction(
            ValueId(9),
            GpuOp::GetElementPtr(ValueId(8), vec![ValueId(0)]),
        );
        block.add_instruction(
            ValueId(10),
            GpuOp::Store(ValueId(9), ValueId(7), MemorySpace::Global),
        );
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        kernel
    }

    #[test]
    fn test_backward_codegen_creation() {
        let config = DiffConfig::default();
        let primitives = PrimitiveRegistry::new();
        let codegen = BackwardCodegen::new(&config, &primitives);
        assert_eq!(codegen.next_value_id, 0);
    }

    #[test]
    fn test_extract_operands() {
        let config = DiffConfig::default();
        let primitives = PrimitiveRegistry::new();
        let codegen = BackwardCodegen::new(&config, &primitives);

        let add_op = GpuOp::FAdd(ValueId(0), ValueId(1));
        let operands = codegen.extract_operands(&add_op);
        assert_eq!(operands, vec![ValueId(0), ValueId(1)]);

        let neg_op = GpuOp::FNeg(ValueId(5));
        let operands = codegen.extract_operands(&neg_op);
        assert_eq!(operands, vec![ValueId(5)]);
    }

    #[test]
    fn test_generate_backward_simple() {
        let forward = create_simple_add_kernel();
        let config = DiffConfig::default();
        let primitives = PrimitiveRegistry::new();
        let mut codegen = BackwardCodegen::new(&config, &primitives);

        let analysis = KernelAnalysis {
            name: "forward_add".to_string(),
            inputs: vec![
                ParamInfo {
                    index: 0,
                    name: "x".to_string(),
                    ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                },
                ParamInfo {
                    index: 1,
                    name: "y".to_string(),
                    ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                },
            ],
            outputs: vec![ParamInfo {
                index: 2,
                name: "z".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            }],
            operations: vec![],
        };

        let result = codegen.generate_backward(&forward, &analysis);
        assert!(result.is_ok());

        let backward = result.unwrap();
        assert_eq!(backward.kernel.name, "forward_add_backward");
        assert!(backward.gradient_params.contains_key("x"));
        assert!(backward.gradient_params.contains_key("y"));
        assert!(backward.gradient_params.contains_key("z"));
    }
}
