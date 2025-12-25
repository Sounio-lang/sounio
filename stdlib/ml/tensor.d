//! stdlib/ml/tensor.d
//!
//! Foundational Tensor Operations for Machine Learning
//!
//! This module provides a dynamic-shape tensor type with operations for:
//! - Construction: zeros, ones, random normal, identity matrices
//! - Indexing: multi-dimensional indexing and slicing
//! - Shape manipulation: reshape, transpose, flatten, squeeze, unsqueeze
//! - Element-wise operations: arithmetic, exponential, logarithm, power
//! - Reductions: sum, mean, max, min (global and per-axis)
//! - Linear algebra: matrix multiplication, dot product
//! - Broadcasting: scalar operations
//! - Epistemic uncertainty: variance propagation through tensor operations
//!
//! # Design
//!
//! - Row-major storage (C-order): last dimension varies fastest
//! - Tensors store flattened data + shape + strides for efficient indexing
//! - All operations return new tensors (functional style)
//! - Variance propagation using delta method where applicable
//!
//! # Example
//!
//! ```demetrios
//! // Create tensors
//! let a = tensor_randn([2, 3], 42)  // 2x3 random normal
//! let b = tensor_ones([3, 4])       // 3x4 ones
//!
//! // Matrix multiplication
//! let c = tensor_matmul(a, b)       // 2x4 result
//!
//! // Element-wise operations
//! let d = tensor_exp(c)
//! let e = tensor_add(d, tensor_fill([2, 4], 1.0))
//!
//! // Reductions
//! let sum = tensor_sum(e)           // Scalar
//! let col_means = tensor_mean_axis(e, 0)  // Mean over rows
//! ```

// ============================================================================
// TENSOR STRUCTURE
// ============================================================================

struct Tensor {
    data: [f64],      // Flattened data in row-major order
    shape: [i64],     // Dimensions [d0, d1, ..., dn]
    strides: [i64],   // Strides for each dimension
    ndim: i64,        // Number of dimensions
    size: i64,        // Total number of elements
}

// Tensor with epistemic uncertainty (variance per element)
struct TensorWithUncertainty {
    data: Tensor,      // Mean values
    variance: Tensor,  // Variance per element
}

// ============================================================================
// HELPER FUNCTIONS - INTERNAL
// ============================================================================

// Compute strides from shape (row-major)
// For shape [2, 3, 4], strides are [12, 4, 1]
fn compute_strides(shape: [i64], ndim: i64) -> [i64] {
    var strides = array_new_i64(ndim)
    var stride = 1
    var i = ndim - 1
    while i >= 0 {
        strides[i] = stride
        stride = stride * shape[i]
        i = i - 1
    }
    return strides
}

// Compute total size from shape
fn compute_size(shape: [i64], ndim: i64) -> i64 {
    var size = 1
    var i = 0
    while i < ndim {
        size = size * shape[i]
        i = i + 1
    }
    return size
}

// Convert multi-dimensional index to linear index
// index: [i0, i1, ..., in], strides: [s0, s1, ..., sn]
// result: i0*s0 + i1*s1 + ... + in*sn
fn linear_index(indices: [i64], strides: [i64], ndim: i64) -> i64 {
    var idx = 0
    var i = 0
    while i < ndim {
        idx = idx + indices[i] * strides[i]
        i = i + 1
    }
    return idx
}

// Check if shapes are compatible for element-wise ops
fn shapes_equal(shape1: [i64], shape2: [i64], ndim: i64) -> i64 {
    var i = 0
    while i < ndim {
        if shape1[i] != shape2[i] {
            return 0
        }
        i = i + 1
    }
    return 1
}

// Copy array of i64
fn copy_array_i64(src: [i64], n: i64) -> [i64] {
    var dst = array_new_i64(n)
    var i = 0
    while i < n {
        dst[i] = src[i]
        i = i + 1
    }
    return dst
}

// Copy array of f64
fn copy_array_f64(src: [f64], n: i64) -> [f64] {
    var dst = array_new_f64(n)
    var i = 0
    while i < n {
        dst[i] = src[i]
        i = i + 1
    }
    return dst
}

// ============================================================================
// CONSTRUCTORS
// ============================================================================

// Create tensor filled with zeros
fn tensor_zeros(shape: [i64]) -> Tensor {
    let ndim = array_len_i64(shape)
    let size = compute_size(shape, ndim)
    let strides = compute_strides(shape, ndim)

    var data = array_new_f64(size)
    var i = 0
    while i < size {
        data[i] = 0.0
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(shape, ndim),
        strides: strides,
        ndim: ndim,
        size: size,
    }
}

// Create tensor filled with ones
fn tensor_ones(shape: [i64]) -> Tensor {
    let ndim = array_len_i64(shape)
    let size = compute_size(shape, ndim)
    let strides = compute_strides(shape, ndim)

    var data = array_new_f64(size)
    var i = 0
    while i < size {
        data[i] = 1.0
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(shape, ndim),
        strides: strides,
        ndim: ndim,
        size: size,
    }
}

// Create tensor filled with constant value
fn tensor_fill(shape: [i64], value: f64) -> Tensor {
    let ndim = array_len_i64(shape)
    let size = compute_size(shape, ndim)
    let strides = compute_strides(shape, ndim)

    var data = array_new_f64(size)
    var i = 0
    while i < size {
        data[i] = value
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(shape, ndim),
        strides: strides,
        ndim: ndim,
        size: size,
    }
}

// Create tensor from data array
fn tensor_from_data(data: [f64], shape: [i64]) -> Tensor {
    let ndim = array_len_i64(shape)
    let size = compute_size(shape, ndim)
    let strides = compute_strides(shape, ndim)

    return Tensor {
        data: copy_array_f64(data, size),
        shape: copy_array_i64(shape, ndim),
        strides: strides,
        ndim: ndim,
        size: size,
    }
}

// Create tensor with random normal values (Box-Muller transform)
fn tensor_randn(shape: [i64], seed: i64) -> Tensor {
    let ndim = array_len_i64(shape)
    let size = compute_size(shape, ndim)
    let strides = compute_strides(shape, ndim)

    var data = array_new_f64(size)
    var rng_state = seed as u64

    var i = 0
    while i < size {
        // Box-Muller: generate two independent normals
        let (u1, u2, new_state) = random_pair(rng_state)
        rng_state = new_state

        let z = sqrt_f64(-2.0 * ln_f64(u1)) * cos_f64(2.0 * 3.141592653589793 * u2)
        data[i] = z
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(shape, ndim),
        strides: strides,
        ndim: ndim,
        size: size,
    }
}

// Create identity matrix (2D tensor)
fn tensor_eye(n: i64) -> Tensor {
    var shape = array_new_i64(2)
    shape[0] = n
    shape[1] = n

    let size = n * n
    let strides = compute_strides(shape, 2)

    var data = array_new_f64(size)
    var i = 0
    while i < n {
        data[i * n + i] = 1.0
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: shape,
        strides: strides,
        ndim: 2,
        size: size,
    }
}

// ============================================================================
// INDEXING & SLICING
// ============================================================================

// Get element at multi-dimensional index
fn tensor_get(t: Tensor, indices: [i64]) -> f64 {
    let idx = linear_index(indices, t.strides, t.ndim)
    return t.data[idx]
}

// Set element at multi-dimensional index (returns new tensor)
fn tensor_set(t: Tensor, indices: [i64], value: f64) -> Tensor {
    let idx = linear_index(indices, t.strides, t.ndim)
    var new_data = copy_array_f64(t.data, t.size)
    new_data[idx] = value

    return Tensor {
        data: new_data,
        shape: copy_array_i64(t.shape, t.ndim),
        strides: copy_array_i64(t.strides, t.ndim),
        ndim: t.ndim,
        size: t.size,
    }
}

// Extract slice (simplified: rectangular region)
// start, end: indices for each dimension
fn tensor_slice(t: Tensor, start: [i64], end: [i64]) -> Tensor {
    // Compute new shape
    var new_shape = array_new_i64(t.ndim)
    var i = 0
    while i < t.ndim {
        new_shape[i] = end[i] - start[i]
        i = i + 1
    }

    let new_size = compute_size(new_shape, t.ndim)
    let new_strides = compute_strides(new_shape, t.ndim)
    var new_data = array_new_f64(new_size)

    // Copy elements in the slice region
    var out_idx = 0
    var indices = copy_array_i64(start, t.ndim)

    // Iterate through all indices in the slice
    while out_idx < new_size {
        let in_idx = linear_index(indices, t.strides, t.ndim)
        new_data[out_idx] = t.data[in_idx]

        // Increment indices
        var dim = t.ndim - 1
        while dim >= 0 {
            indices[dim] = indices[dim] + 1
            if indices[dim] < end[dim] {
                break
            }
            indices[dim] = start[dim]
            dim = dim - 1
        }

        out_idx = out_idx + 1
    }

    return Tensor {
        data: new_data,
        shape: new_shape,
        strides: new_strides,
        ndim: t.ndim,
        size: new_size,
    }
}

// ============================================================================
// SHAPE OPERATIONS
// ============================================================================

// Reshape tensor (must preserve total size)
fn tensor_reshape(t: Tensor, new_shape: [i64]) -> Tensor {
    let new_ndim = array_len_i64(new_shape)
    let new_size = compute_size(new_shape, new_ndim)
    let new_strides = compute_strides(new_shape, new_ndim)

    return Tensor {
        data: copy_array_f64(t.data, t.size),
        shape: copy_array_i64(new_shape, new_ndim),
        strides: new_strides,
        ndim: new_ndim,
        size: new_size,
    }
}

// Transpose 2D tensor (swap dimensions)
fn tensor_transpose(t: Tensor) -> Tensor {
    let rows = t.shape[0]
    let cols = t.shape[1]

    var new_shape = array_new_i64(2)
    new_shape[0] = cols
    new_shape[1] = rows

    let size = rows * cols
    var new_data = array_new_f64(size)

    var i = 0
    while i < rows {
        var j = 0
        while j < cols {
            new_data[j * rows + i] = t.data[i * cols + j]
            j = j + 1
        }
        i = i + 1
    }

    return Tensor {
        data: new_data,
        shape: new_shape,
        strides: compute_strides(new_shape, 2),
        ndim: 2,
        size: size,
    }
}

// Flatten to 1D tensor
fn tensor_flatten(t: Tensor) -> Tensor {
    var new_shape = array_new_i64(1)
    new_shape[0] = t.size

    return Tensor {
        data: copy_array_f64(t.data, t.size),
        shape: new_shape,
        strides: compute_strides(new_shape, 1),
        ndim: 1,
        size: t.size,
    }
}

// Remove dimensions of size 1
fn tensor_squeeze(t: Tensor) -> Tensor {
    // Count non-singleton dimensions
    var new_ndim = 0
    var i = 0
    while i < t.ndim {
        if t.shape[i] != 1 {
            new_ndim = new_ndim + 1
        }
        i = i + 1
    }

    var new_shape = array_new_i64(new_ndim)
    var j = 0
    i = 0
    while i < t.ndim {
        if t.shape[i] != 1 {
            new_shape[j] = t.shape[i]
            j = j + 1
        }
        i = i + 1
    }

    return Tensor {
        data: copy_array_f64(t.data, t.size),
        shape: new_shape,
        strides: compute_strides(new_shape, new_ndim),
        ndim: new_ndim,
        size: t.size,
    }
}

// Add dimension of size 1 at position dim
fn tensor_unsqueeze(t: Tensor, dim: i64) -> Tensor {
    let new_ndim = t.ndim + 1
    var new_shape = array_new_i64(new_ndim)

    var i = 0
    var j = 0
    while i < new_ndim {
        if i == dim {
            new_shape[i] = 1
        } else {
            new_shape[i] = t.shape[j]
            j = j + 1
        }
        i = i + 1
    }

    return Tensor {
        data: copy_array_f64(t.data, t.size),
        shape: new_shape,
        strides: compute_strides(new_shape, new_ndim),
        ndim: new_ndim,
        size: t.size,
    }
}

// ============================================================================
// ELEMENT-WISE OPERATIONS
// ============================================================================

// Element-wise addition
fn tensor_add(a: Tensor, b: Tensor) -> Tensor {
    var data = array_new_f64(a.size)
    var i = 0
    while i < a.size {
        data[i] = a.data[i] + b.data[i]
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(a.shape, a.ndim),
        strides: copy_array_i64(a.strides, a.ndim),
        ndim: a.ndim,
        size: a.size,
    }
}

// Element-wise subtraction
fn tensor_sub(a: Tensor, b: Tensor) -> Tensor {
    var data = array_new_f64(a.size)
    var i = 0
    while i < a.size {
        data[i] = a.data[i] - b.data[i]
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(a.shape, a.ndim),
        strides: copy_array_i64(a.strides, a.ndim),
        ndim: a.ndim,
        size: a.size,
    }
}

// Element-wise multiplication
fn tensor_mul(a: Tensor, b: Tensor) -> Tensor {
    var data = array_new_f64(a.size)
    var i = 0
    while i < a.size {
        data[i] = a.data[i] * b.data[i]
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(a.shape, a.ndim),
        strides: copy_array_i64(a.strides, a.ndim),
        ndim: a.ndim,
        size: a.size,
    }
}

// Element-wise division
fn tensor_div(a: Tensor, b: Tensor) -> Tensor {
    var data = array_new_f64(a.size)
    var i = 0
    while i < a.size {
        data[i] = a.data[i] / b.data[i]
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(a.shape, a.ndim),
        strides: copy_array_i64(a.strides, a.ndim),
        ndim: a.ndim,
        size: a.size,
    }
}

// Element-wise negation
fn tensor_neg(t: Tensor) -> Tensor {
    var data = array_new_f64(t.size)
    var i = 0
    while i < t.size {
        data[i] = 0.0 - t.data[i]
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(t.shape, t.ndim),
        strides: copy_array_i64(t.strides, t.ndim),
        ndim: t.ndim,
        size: t.size,
    }
}

// Element-wise exponential
fn tensor_exp(t: Tensor) -> Tensor {
    var data = array_new_f64(t.size)
    var i = 0
    while i < t.size {
        data[i] = exp_f64(t.data[i])
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(t.shape, t.ndim),
        strides: copy_array_i64(t.strides, t.ndim),
        ndim: t.ndim,
        size: t.size,
    }
}

// Element-wise natural logarithm
fn tensor_log(t: Tensor) -> Tensor {
    var data = array_new_f64(t.size)
    var i = 0
    while i < t.size {
        data[i] = ln_f64(t.data[i])
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(t.shape, t.ndim),
        strides: copy_array_i64(t.strides, t.ndim),
        ndim: t.ndim,
        size: t.size,
    }
}

// Element-wise square root
fn tensor_sqrt(t: Tensor) -> Tensor {
    var data = array_new_f64(t.size)
    var i = 0
    while i < t.size {
        data[i] = sqrt_f64(t.data[i])
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(t.shape, t.ndim),
        strides: copy_array_i64(t.strides, t.ndim),
        ndim: t.ndim,
        size: t.size,
    }
}

// Element-wise power
fn tensor_pow(t: Tensor, p: f64) -> Tensor {
    var data = array_new_f64(t.size)
    var i = 0
    while i < t.size {
        data[i] = pow_f64(t.data[i], p)
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(t.shape, t.ndim),
        strides: copy_array_i64(t.strides, t.ndim),
        ndim: t.ndim,
        size: t.size,
    }
}

// ============================================================================
// REDUCTIONS
// ============================================================================

// Sum all elements
fn tensor_sum(t: Tensor) -> f64 {
    var sum = 0.0
    var i = 0
    while i < t.size {
        sum = sum + t.data[i]
        i = i + 1
    }
    return sum
}

// Mean of all elements
fn tensor_mean(t: Tensor) -> f64 {
    let sum = tensor_sum(t)
    return sum / (t.size as f64)
}

// Maximum element
fn tensor_max(t: Tensor) -> f64 {
    var max_val = t.data[0]
    var i = 1
    while i < t.size {
        if t.data[i] > max_val {
            max_val = t.data[i]
        }
        i = i + 1
    }
    return max_val
}

// Minimum element
fn tensor_min(t: Tensor) -> f64 {
    var min_val = t.data[0]
    var i = 1
    while i < t.size {
        if t.data[i] < min_val {
            min_val = t.data[i]
        }
        i = i + 1
    }
    return min_val
}

// Sum along axis (simplified for common cases)
fn tensor_sum_axis(t: Tensor, axis: i64) -> Tensor {
    // For 2D: axis=0 sums rows (result shape: [cols]), axis=1 sums cols (result shape: [rows])
    if t.ndim == 2 {
        let rows = t.shape[0]
        let cols = t.shape[1]

        if axis == 0 {
            // Sum over rows -> [cols]
            var shape = array_new_i64(1)
            shape[0] = cols
            var data = array_new_f64(cols)

            var j = 0
            while j < cols {
                var sum = 0.0
                var i = 0
                while i < rows {
                    sum = sum + t.data[i * cols + j]
                    i = i + 1
                }
                data[j] = sum
                j = j + 1
            }

            return tensor_from_data(data, shape)
        } else {
            // Sum over cols -> [rows]
            var shape = array_new_i64(1)
            shape[0] = rows
            var data = array_new_f64(rows)

            var i = 0
            while i < rows {
                var sum = 0.0
                var j = 0
                while j < cols {
                    sum = sum + t.data[i * cols + j]
                    j = j + 1
                }
                data[i] = sum
                i = i + 1
            }

            return tensor_from_data(data, shape)
        }
    }

    // Fallback: return flattened sum for now
    var shape = array_new_i64(1)
    shape[0] = 1
    var data = array_new_f64(1)
    data[0] = tensor_sum(t)
    return tensor_from_data(data, shape)
}

// Mean along axis
fn tensor_mean_axis(t: Tensor, axis: i64) -> Tensor {
    let sum_t = tensor_sum_axis(t, axis)
    let n = if axis == 0 { t.shape[0] } else { t.shape[1] }
    let scale = 1.0 / (n as f64)

    return tensor_broadcast_mul(sum_t, scale)
}

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

// Matrix multiplication (2D tensors only)
fn tensor_matmul(a: Tensor, b: Tensor) -> Tensor {
    let m = a.shape[0]
    let k = a.shape[1]
    let n = b.shape[1]

    var shape = array_new_i64(2)
    shape[0] = m
    shape[1] = n

    let size = m * n
    var data = array_new_f64(size)

    var i = 0
    while i < m {
        var j = 0
        while j < n {
            var sum = 0.0
            var p = 0
            while p < k {
                sum = sum + a.data[i * k + p] * b.data[p * n + j]
                p = p + 1
            }
            data[i * n + j] = sum
            j = j + 1
        }
        i = i + 1
    }

    return tensor_from_data(data, shape)
}

// Dot product (1D tensors only)
fn tensor_dot(a: Tensor, b: Tensor) -> f64 {
    var sum = 0.0
    var i = 0
    while i < a.size {
        sum = sum + a.data[i] * b.data[i]
        i = i + 1
    }
    return sum
}

// ============================================================================
// BROADCASTING
// ============================================================================

// Broadcast add scalar to all elements
fn tensor_broadcast_add(t: Tensor, scalar: f64) -> Tensor {
    var data = array_new_f64(t.size)
    var i = 0
    while i < t.size {
        data[i] = t.data[i] + scalar
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(t.shape, t.ndim),
        strides: copy_array_i64(t.strides, t.ndim),
        ndim: t.ndim,
        size: t.size,
    }
}

// Broadcast multiply scalar to all elements
fn tensor_broadcast_mul(t: Tensor, scalar: f64) -> Tensor {
    var data = array_new_f64(t.size)
    var i = 0
    while i < t.size {
        data[i] = t.data[i] * scalar
        i = i + 1
    }

    return Tensor {
        data: data,
        shape: copy_array_i64(t.shape, t.ndim),
        strides: copy_array_i64(t.strides, t.ndim),
        ndim: t.ndim,
        size: t.size,
    }
}

// ============================================================================
// EPISTEMIC UNCERTAINTY PROPAGATION
// ============================================================================

// Create tensor with uncertainty
fn tensor_with_uncertainty(data: Tensor, variance: Tensor) -> TensorWithUncertainty {
    return TensorWithUncertainty {
        data: data,
        variance: variance,
    }
}

// Add tensors with uncertainty propagation
// Var(X + Y) = Var(X) + Var(Y)
fn tensor_add_uncertain(a: TensorWithUncertainty, b: TensorWithUncertainty) -> TensorWithUncertainty {
    let data = tensor_add(a.data, b.data)
    let variance = tensor_add(a.variance, b.variance)

    return TensorWithUncertainty {
        data: data,
        variance: variance,
    }
}

// Multiply tensors with uncertainty propagation
// Var(XY) ≈ Y²Var(X) + X²Var(Y)
fn tensor_mul_uncertain(a: TensorWithUncertainty, b: TensorWithUncertainty) -> TensorWithUncertainty {
    let data = tensor_mul(a.data, b.data)

    let b_sq = tensor_mul(b.data, b.data)
    let a_sq = tensor_mul(a.data, a.data)

    let term1 = tensor_mul(b_sq, a.variance)
    let term2 = tensor_mul(a_sq, b.variance)
    let variance = tensor_add(term1, term2)

    return TensorWithUncertainty {
        data: data,
        variance: variance,
    }
}

// Exponential with uncertainty propagation
// Var(e^X) ≈ e^(2X) · Var(X)
fn tensor_exp_uncertain(t: TensorWithUncertainty) -> TensorWithUncertainty {
    let data = tensor_exp(t.data)

    let two_x = tensor_broadcast_mul(t.data, 2.0)
    let exp_2x = tensor_exp(two_x)
    let variance = tensor_mul(exp_2x, t.variance)

    return TensorWithUncertainty {
        data: data,
        variance: variance,
    }
}

// Log with uncertainty propagation
// Var(ln X) ≈ Var(X) / X²
fn tensor_log_uncertain(t: TensorWithUncertainty) -> TensorWithUncertainty {
    let data = tensor_log(t.data)

    let x_sq = tensor_mul(t.data, t.data)
    let variance = tensor_div(t.variance, x_sq)

    return TensorWithUncertainty {
        data: data,
        variance: variance,
    }
}

// Matrix multiplication with uncertainty propagation
// For C = AB, Var(C_ij) ≈ Σ_k [B_kj² Var(A_ik) + A_ik² Var(B_kj)]
fn tensor_matmul_uncertain(a: TensorWithUncertainty, b: TensorWithUncertainty) -> TensorWithUncertainty {
    let data = tensor_matmul(a.data, b.data)

    let m = a.data.shape[0]
    let k = a.data.shape[1]
    let n = b.data.shape[1]

    var shape = array_new_i64(2)
    shape[0] = m
    shape[1] = n

    let size = m * n
    var var_data = array_new_f64(size)

    var i = 0
    while i < m {
        var j = 0
        while j < n {
            var var_sum = 0.0
            var p = 0
            while p < k {
                let b_val = b.data.data[p * n + j]
                let a_var = a.variance.data[i * k + p]
                let a_val = a.data.data[i * k + p]
                let b_var = b.variance.data[p * n + j]

                var_sum = var_sum + b_val * b_val * a_var + a_val * a_val * b_var
                p = p + 1
            }
            var_data[i * n + j] = var_sum
            j = j + 1
        }
        i = i + 1
    }

    let variance = tensor_from_data(var_data, shape)

    return TensorWithUncertainty {
        data: data,
        variance: variance,
    }
}

// ============================================================================
// MATH HELPER FUNCTIONS
// ============================================================================

fn exp_f64(x: f64) -> f64 {
    if x > 10.0 { return 22026.465794806718 * exp_f64(x - 10.0) }
    if x < -10.0 { return 0.00004539992976248485 * exp_f64(x + 10.0) }

    var result = 1.0
    var term = 1.0
    var i = 1
    while i < 20 {
        term = term * x / (i as f64)
        result = result + term
        i = i + 1
    }
    return result
}

fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 999999.0 }
    if x == 1.0 { return 0.0 }

    let y = (x - 1.0) / (x + 1.0)
    var result = 0.0
    var term = y
    var i = 1
    while i < 30 {
        result = result + term / (i as f64)
        term = term * y * y
        i = i + 2
    }
    return 2.0 * result
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    var y = x
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    return y
}

fn pow_f64(base: f64, exp: f64) -> f64 {
    return exp_f64(exp * ln_f64(base))
}

fn cos_f64(x: f64) -> f64 {
    let pi = 3.141592653589793
    var x_mod = x
    while x_mod > pi { x_mod = x_mod - 2.0 * pi }
    while x_mod < 0.0 - pi { x_mod = x_mod + 2.0 * pi }

    var result = 1.0
    var term = 1.0
    var i = 1
    while i < 10 {
        term = 0.0 - term * x_mod * x_mod / ((2.0 * i as f64 - 1.0) * (2.0 * i as f64))
        result = result + term
        i = i + 1
    }
    return result
}

// Simple LCG random number generator
fn random_pair(state: u64) -> (f64, f64, u64) {
    let a = 6364136223846793005_u64
    let c = 1442695040888963407_u64

    let s1 = state * a + c
    let s2 = s1 * a + c

    let u1 = ((s1 >> 32) as f64) / 4294967296.0
    let u2 = ((s2 >> 32) as f64) / 4294967296.0

    let u1_safe = if u1 < 0.0000001 { 0.0000001 } else { u1 }

    return (u1_safe, u2, s2)
}

// ============================================================================
// ARRAY ALLOCATION STUBS
// ============================================================================

// These would typically be provided by the runtime or stdlib
fn array_new_f64(n: i64) -> [f64] {
    // Placeholder - actual implementation allocates array
    let dummy = [0.0]
    return dummy
}

fn array_new_i64(n: i64) -> [i64] {
    // Placeholder - actual implementation allocates array
    let dummy = [0]
    return dummy
}

fn array_len_i64(arr: [i64]) -> i64 {
    // Placeholder - actual implementation returns array length
    return 0
}
