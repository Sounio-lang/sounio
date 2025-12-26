//! Runtime Support for Sounio
//!
//! This module provides runtime representations and support code for
//! Sounio programs. Key features:
//!
//! - Epistemic type runtime representations (Full/Compact/Erased modes)
//! - GPU memory layouts for vectorized epistemic operations
//! - GPU kernel execution runtime (cudarc-based)
//! - Runtime support for confidence tracking
//! - Provenance chain management
//! - ODE solvers (Euler, RK4, Dormand-Prince)
//! - Stiff ODE solvers (BDF, LSODA, Rosenbrock)
//! - Uncertainty propagation (uncertain<T>)
//! - Probabilistic programming (sample/observe/infer)
//! - Tensor with compile-time shape verification
//! - Einstein notation (einsum)
//! - Symbolic computation (differentiation, integration, simplification)
//! - Causal inference (do-calculus, counterfactuals)
//! - Model discovery (SINDy-like sparse regression)
//! - PDE solvers (heat, wave, advection, diffusion-reaction)
//! - GPU kernels for scientific primitives
//! - Async/await runtime for concurrent execution

pub mod async_runtime;
pub mod causal;
pub mod discover;
pub mod einsum;
pub mod epistemic;
pub mod gpu_epistemic;
pub mod gpu_executor;
pub mod gpu_scientific;
pub mod io;
pub mod ode;
pub mod pde;
pub mod prob;
pub mod stiff;
pub mod symbolic;
pub mod tensor;
pub mod uncertain;

pub use causal::{
    ATEResult, CausalModel, CounterfactualResult, DAG, Evidence, counterfactual, estimate_ate,
    find_backdoor_adjustment, satisfies_backdoor,
};
pub use discover::{
    DiscoveredModel, DiscoveredODE, FunctionLibrary, LibraryTerm, SINDyOptions,
    compute_derivatives, discover_ode, dynamics_library, polynomial_library, sindy,
};
pub use einsum::{
    EinsumError, EinsumExpr, diagonal, einsum, sum_axis, tensordot, trace, transpose,
};
pub use epistemic::{
    CompactKnowledge, EpistemicMode, EpistemicRuntime, ErasedKnowledge, FullKnowledge,
    RuntimeConfidence, RuntimeProvenance,
};
pub use gpu_epistemic::{AoSKnowledge, GpuEpistemicArray, GpuMemoryLayout, SoAKnowledge};
pub use gpu_scientific::{
    CompiledKernel, DataType, DeviceCapabilities, GPUBackend, GPUBuffer, KernelManager,
    KernelParams, KernelType, LaunchConfig, MatMulKernel, MonteCarloKernel, ODEKernel, ODEMethod,
    ReductionKernel, ReductionOp, StencilKernel, StencilType, TensorContractionKernel,
};
pub use ode::{
    ODESolution, SolverMethod, SolverOptions, solve, solve_euler, solve_rk4, solve_rk45,
};
pub use pde::{
    BoundaryCondition, Domain1D, Domain2D, PDESolution1D, PDESolution2D, advection_equation_1d,
    diffusion_reaction_1d, heat_equation_1d_crank_nicolson, heat_equation_1d_explicit,
    heat_equation_2d_explicit, wave_equation_1d,
};
pub use prob::{Distribution, InferenceMethod, ProbRuntime};
pub use stiff::{
    BDFCoefficients, BDFConfig, BDFSolver, LSODAConfig, LSODASolver, MethodType, NewtonConfig,
    SolverStats, StiffSolution, implicit_euler, rosenbrock, solve_lsoda, solve_stiff,
};
pub use symbolic::{
    Expr, compile_expr, differentiate, evaluate, expand, integrate, simplify, solve_polynomial,
    substitute,
};
pub use tensor::{
    Dim, Shape, ShapeError, Tensor, verify_elementwise, verify_matmul, verify_reshape,
};
pub use uncertain::{Uncertain, UncertainOps};

// Async runtime
pub use async_runtime::{
    AsyncContext, AsyncState, SounioFuture, SounioRuntime, SounioValue, TaskHandle,
    TaskId, block_on, init_runtime, runtime, spawn,
};

// GPU Executor (cudarc-based kernel execution runtime)
pub use gpu_executor::{
    DeviceInfo as GpuDeviceInfo, EpistemicBuffer, ExecutorBuffer, GpuExecutor, GpuExecutorError,
    KernelLauncher, KernelParam,
};
