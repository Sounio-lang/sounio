// gen_ptx — generate epistemic GEMM PTX and print to stdout.
//
// Usage:
//   GEMM_M=4096 GEMM_N=4096 GEMM_K=4096 SM_MAJOR=8 \
//     cargo run --release --features gpu --bin gen_ptx
//
// Output is fenced with --- PTX BEGIN --- / --- PTX END ---
// for extraction by scripts/cuda_gemm_dispatch.py.

use sounio::codegen::gpu::epistemic_gemm::{
    generate_epistemic_gemm_ptx_sm, EpistemicGemmConfig, MatrixPrecision,
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let m  = env_usize("GEMM_M", 4096);
    let n  = env_usize("GEMM_N", m);
    let k  = env_usize("GEMM_K", m);
    let sm = env_u32("SM_MAJOR", 8);

    let config = EpistemicGemmConfig {
        m, k, n,
        alpha: 1.0,
        beta: 0.0,
        precision: MatrixPrecision::F32,
        use_tensor_cores: false,
        confidence_threshold: None,
    };

    let ptx = generate_epistemic_gemm_ptx_sm(&config, "epistemic_gemm", sm, 0);

    eprintln!("[gen_ptx] {}×{}×{} SM{} → {} bytes PTX", m, n, k, sm, ptx.len());
    println!("--- PTX BEGIN ---");
    println!("{}", ptx);
    println!("--- PTX END ---");
}
