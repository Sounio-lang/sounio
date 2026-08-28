//! C FFI surface for calling the ODEP prover from Sounio.
//!
//! Sounio's `with ZD, Witness` functions can invoke this symbol via
//! the standard C ABI to obtain a JSON envelope of the unlearn proof.

use crate::{prove, UnlearnClaim, PrimaSpec, KERNEL_DIM};
use std::slice;

/// Generate an ODEP witness envelope.
///
/// # Safety
/// * `w_pre` and `w_post` must each point to at least `KERNEL_DIM` f64s.
/// * `out_buf` must point to at least `out_cap` writable bytes.
/// * Returns the number of bytes written to `out_buf` on success, or a
///   negative error code on failure.
#[no_mangle]
pub unsafe extern "C" fn odep_prove(
    w_pre: *const f64,
    w_post: *const f64,
    c1: f64,
    c2: f64,
    prim_i: usize,
    prim_j: usize,
    out_buf: *mut u8,
    out_cap: usize,
) -> i32 {
    if w_pre.is_null() || w_post.is_null() || out_buf.is_null() {
        return -1;
    }
    let pre_slice = slice::from_raw_parts(w_pre, KERNEL_DIM);
    let post_slice = slice::from_raw_parts(w_post, KERNEL_DIM);

    let mut w_pre_arr = [0.0_f64; KERNEL_DIM];
    let mut w_post_arr = [0.0_f64; KERNEL_DIM];
    w_pre_arr.copy_from_slice(pre_slice);
    w_post_arr.copy_from_slice(post_slice);

    let claim = UnlearnClaim {
        w_pre: w_pre_arr,
        w_post: w_post_arr,
        c1,
        c2,
        prim_a_i: prim_i,
        prim_a_j: prim_j,
    };

    match prove(&claim, "GDPR-Art17") {
        Ok(env) => {
            let json = match serde_json::to_vec(&env) {
                Ok(v) => v,
                Err(_) => return -2,
            };
            if json.len() > out_cap {
                return -3;
            }
            std::ptr::copy_nonoverlapping(json.as_ptr(), out_buf, json.len());
            json.len() as i32
        }
        Err(_) => -4,
    }
}

// Silence the unused-import warning if `PrimaSpec` ever goes unused in
// the future; it is part of the envelope schema and may be consumed
// by callers.
#[allow(dead_code)]
fn _use_prima_spec(_: PrimaSpec) {}
