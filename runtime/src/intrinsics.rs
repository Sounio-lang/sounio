//! Runtime intrinsics for Sounio
//!
//! These functions are called by compiled Sounio code via FFI.

use std::ffi::CStr;
use std::os::raw::c_char;

// ============================================================================
// Print intrinsics
// ============================================================================

/// Print a string to stdout
#[no_mangle]
pub extern "C" fn sounio_print_str(s: *const c_char) {
    if s.is_null() {
        return;
    }

    unsafe {
        let cstr = CStr::from_ptr(s);
        if let Ok(s) = cstr.to_str() {
            print!("{}", s);
        }
    }
}

/// Print an integer
#[no_mangle]
pub extern "C" fn sounio_print_i64(value: i64) {
    print!("{}", value);
}

/// Print a float
#[no_mangle]
pub extern "C" fn sounio_print_f64(value: f64) {
    print!("{}", value);
}

/// Print a boolean
#[no_mangle]
pub extern "C" fn sounio_print_bool(value: bool) {
    print!("{}", value);
}

/// Print newline
#[no_mangle]
pub extern "C" fn sounio_println() {
    println!();
}

// ============================================================================
// Memory intrinsics
// ============================================================================

/// Allocate memory
#[no_mangle]
pub extern "C" fn sounio_alloc(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        return align as *mut u8;
    }

    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
        std::alloc::alloc(layout)
    }
}

/// Allocate zeroed memory
#[no_mangle]
pub extern "C" fn sounio_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        return align as *mut u8;
    }

    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
        std::alloc::alloc_zeroed(layout)
    }
}

/// Deallocate memory
#[no_mangle]
pub extern "C" fn sounio_dealloc(ptr: *mut u8, size: usize, align: usize) {
    if size == 0 || ptr.is_null() {
        return;
    }

    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
        std::alloc::dealloc(ptr, layout);
    }
}

/// Reallocate memory
#[no_mangle]
pub extern "C" fn sounio_realloc(ptr: *mut u8, old_size: usize, align: usize, new_size: usize) -> *mut u8 {
    if ptr.is_null() || old_size == 0 {
        return sounio_alloc(new_size, align);
    }

    if new_size == 0 {
        sounio_dealloc(ptr, old_size, align);
        return std::ptr::null_mut();
    }

    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(old_size, align);
        std::alloc::realloc(ptr, layout, new_size)
    }
}

// ============================================================================
// Panic / Assert intrinsics
// ============================================================================

/// Panic with message
#[no_mangle]
pub extern "C" fn sounio_panic(msg: *const c_char) -> ! {
    let message = if msg.is_null() {
        "panic".to_string()
    } else {
        unsafe {
            CStr::from_ptr(msg)
                .to_str()
                .unwrap_or("panic")
                .to_string()
        }
    };

    panic!("{}", message);
}

/// Assert with message
#[no_mangle]
pub extern "C" fn sounio_assert(condition: bool, msg: *const c_char) {
    if !condition {
        sounio_panic(msg);
    }
}

// ============================================================================
// Math intrinsics
// ============================================================================

#[no_mangle]
pub extern "C" fn sounio_sqrt_f64(x: f64) -> f64 { x.sqrt() }

#[no_mangle]
pub extern "C" fn sounio_sin_f64(x: f64) -> f64 { x.sin() }

#[no_mangle]
pub extern "C" fn sounio_cos_f64(x: f64) -> f64 { x.cos() }

#[no_mangle]
pub extern "C" fn sounio_tan_f64(x: f64) -> f64 { x.tan() }

#[no_mangle]
pub extern "C" fn sounio_asin_f64(x: f64) -> f64 { x.asin() }

#[no_mangle]
pub extern "C" fn sounio_acos_f64(x: f64) -> f64 { x.acos() }

#[no_mangle]
pub extern "C" fn sounio_atan_f64(x: f64) -> f64 { x.atan() }

#[no_mangle]
pub extern "C" fn sounio_atan2_f64(y: f64, x: f64) -> f64 { y.atan2(x) }

#[no_mangle]
pub extern "C" fn sounio_exp_f64(x: f64) -> f64 { x.exp() }

#[no_mangle]
pub extern "C" fn sounio_ln_f64(x: f64) -> f64 { x.ln() }

#[no_mangle]
pub extern "C" fn sounio_log10_f64(x: f64) -> f64 { x.log10() }

#[no_mangle]
pub extern "C" fn sounio_log2_f64(x: f64) -> f64 { x.log2() }

#[no_mangle]
pub extern "C" fn sounio_pow_f64(base: f64, exp: f64) -> f64 { base.powf(exp) }

#[no_mangle]
pub extern "C" fn sounio_abs_f64(x: f64) -> f64 { x.abs() }

#[no_mangle]
pub extern "C" fn sounio_floor_f64(x: f64) -> f64 { x.floor() }

#[no_mangle]
pub extern "C" fn sounio_ceil_f64(x: f64) -> f64 { x.ceil() }

#[no_mangle]
pub extern "C" fn sounio_round_f64(x: f64) -> f64 { x.round() }

#[no_mangle]
pub extern "C" fn sounio_trunc_f64(x: f64) -> f64 { x.trunc() }

#[no_mangle]
pub extern "C" fn sounio_min_f64(a: f64, b: f64) -> f64 { a.min(b) }

#[no_mangle]
pub extern "C" fn sounio_max_f64(a: f64, b: f64) -> f64 { a.max(b) }

#[no_mangle]
pub extern "C" fn sounio_clamp_f64(x: f64, min: f64, max: f64) -> f64 { x.clamp(min, max) }

// ============================================================================
// Knowledge/Uncertainty intrinsics
// ============================================================================

/// Knowledge value with uncertainty
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Knowledge {
    pub value: f64,
    pub uncertainty: f64,
    pub confidence: f64,
}

#[no_mangle]
pub extern "C" fn sounio_knowledge_new(value: f64, uncertainty: f64, confidence: f64) -> Knowledge {
    Knowledge {
        value,
        uncertainty: uncertainty.abs(),
        confidence: confidence.clamp(0.0, 1.0),
    }
}

#[no_mangle]
pub extern "C" fn sounio_knowledge_exact(value: f64) -> Knowledge {
    Knowledge {
        value,
        uncertainty: 0.0,
        confidence: 1.0,
    }
}

/// Add two Knowledge values (uncertainty propagation)
#[no_mangle]
pub extern "C" fn sounio_knowledge_add(a: Knowledge, b: Knowledge) -> Knowledge {
    Knowledge {
        value: a.value + b.value,
        // Quadrature sum for independent uncertainties
        uncertainty: (a.uncertainty.powi(2) + b.uncertainty.powi(2)).sqrt(),
        // Minimum confidence
        confidence: a.confidence.min(b.confidence),
    }
}

/// Subtract two Knowledge values
#[no_mangle]
pub extern "C" fn sounio_knowledge_sub(a: Knowledge, b: Knowledge) -> Knowledge {
    Knowledge {
        value: a.value - b.value,
        uncertainty: (a.uncertainty.powi(2) + b.uncertainty.powi(2)).sqrt(),
        confidence: a.confidence.min(b.confidence),
    }
}

/// Multiply two Knowledge values
#[no_mangle]
pub extern "C" fn sounio_knowledge_mul(a: Knowledge, b: Knowledge) -> Knowledge {
    let value = a.value * b.value;
    // Relative uncertainty propagation
    let rel_a = if a.value != 0.0 { a.uncertainty / a.value.abs() } else { 0.0 };
    let rel_b = if b.value != 0.0 { b.uncertainty / b.value.abs() } else { 0.0 };
    let rel_result = (rel_a.powi(2) + rel_b.powi(2)).sqrt();

    Knowledge {
        value,
        uncertainty: value.abs() * rel_result,
        confidence: a.confidence.min(b.confidence),
    }
}

/// Divide two Knowledge values
#[no_mangle]
pub extern "C" fn sounio_knowledge_div(a: Knowledge, b: Knowledge) -> Knowledge {
    if b.value == 0.0 {
        return Knowledge {
            value: f64::NAN,
            uncertainty: f64::INFINITY,
            confidence: 0.0,
        };
    }

    let value = a.value / b.value;
    let rel_a = if a.value != 0.0 { a.uncertainty / a.value.abs() } else { 0.0 };
    let rel_b = b.uncertainty / b.value.abs();
    let rel_result = (rel_a.powi(2) + rel_b.powi(2)).sqrt();

    Knowledge {
        value,
        uncertainty: value.abs() * rel_result,
        confidence: a.confidence.min(b.confidence),
    }
}

/// Power function for Knowledge
#[no_mangle]
pub extern "C" fn sounio_knowledge_pow(k: Knowledge, n: f64) -> Knowledge {
    let value = k.value.powf(n);
    let rel_uncertainty = n.abs() * (k.uncertainty / k.value.abs());

    Knowledge {
        value,
        uncertainty: value.abs() * rel_uncertainty,
        confidence: k.confidence,
    }
}

/// Print Knowledge value
#[no_mangle]
pub extern "C" fn sounio_knowledge_print(k: Knowledge) {
    let ci_width = k.uncertainty * 1.96; // 95% CI for Gaussian
    print!("{:.4} +/- {:.4} ({}% CI)", k.value, ci_width, (k.confidence * 100.0) as i32);
}

/// Get 95% confidence interval lower bound
#[no_mangle]
pub extern "C" fn sounio_knowledge_ci_lower(k: Knowledge) -> f64 {
    k.value - k.uncertainty * 1.96
}

/// Get 95% confidence interval upper bound
#[no_mangle]
pub extern "C" fn sounio_knowledge_ci_upper(k: Knowledge) -> f64 {
    k.value + k.uncertainty * 1.96
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_add() {
        let a = sounio_knowledge_new(10.0, 0.5, 0.95);
        let b = sounio_knowledge_new(20.0, 0.3, 0.95);
        let result = sounio_knowledge_add(a, b);

        assert!((result.value - 30.0).abs() < 1e-10);
        assert!((result.uncertainty - (0.5_f64.powi(2) + 0.3_f64.powi(2)).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_knowledge_mul() {
        let a = sounio_knowledge_new(10.0, 1.0, 0.95); // 10% relative uncertainty
        let b = sounio_knowledge_new(5.0, 0.5, 0.95);  // 10% relative uncertainty
        let result = sounio_knowledge_mul(a, b);

        assert!((result.value - 50.0).abs() < 1e-10);
        // Relative uncertainty should be ~14.14% (sqrt(0.1^2 + 0.1^2))
    }

    #[test]
    fn test_math_intrinsics() {
        assert!((sounio_sqrt_f64(4.0) - 2.0).abs() < 1e-10);
        assert!((sounio_sin_f64(0.0)).abs() < 1e-10);
        assert!((sounio_cos_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((sounio_exp_f64(0.0) - 1.0).abs() < 1e-10);
        assert!((sounio_ln_f64(1.0)).abs() < 1e-10);
    }
}
