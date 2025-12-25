//! PDE (Partial Differential Equations) Solver Runtime
//!
//! This module provides native PDE solvers for Sounio, enabling
//! simulation of spatially-distributed dynamical systems.
//!
//! # Supported PDEs
//!
//! - Heat equation: ∂u/∂t = α∇²u
//! - Wave equation: ∂²u/∂t² = c²∇²u
//! - Advection equation: ∂u/∂t + v·∇u = 0
//! - Diffusion-reaction: ∂u/∂t = D∇²u + f(u)
//!
//! # Methods
//!
//! - Finite Difference Method (FDM)
//! - Method of Lines (MOL)
//! - Crank-Nicolson (implicit)
//! - ADI (Alternating Direction Implicit)
//!
//! # Example
//!
//! ```d
//! pde HeatEquation {
//!     params: { alpha: f64 }
//!     domain: [0, 1]
//!
//!     ∂u/∂t = alpha * ∂²u/∂x²
//!
//!     boundary: {
//!         x = 0: u = 0,
//!         x = 1: u = 0,
//!     }
//! }
//! ```

/// Domain specification for PDE
#[derive(Debug, Clone)]
pub struct Domain1D {
    /// Left boundary
    pub x_min: f64,
    /// Right boundary
    pub x_max: f64,
    /// Number of spatial grid points
    pub nx: usize,
}

impl Domain1D {
    pub fn new(x_min: f64, x_max: f64, nx: usize) -> Self {
        Self { x_min, x_max, nx }
    }

    /// Grid spacing
    pub fn dx(&self) -> f64 {
        (self.x_max - self.x_min) / (self.nx - 1) as f64
    }

    /// Get x coordinate at index i
    pub fn x(&self, i: usize) -> f64 {
        self.x_min + i as f64 * self.dx()
    }

    /// Generate grid points
    pub fn grid(&self) -> Vec<f64> {
        (0..self.nx).map(|i| self.x(i)).collect()
    }
}

/// 2D Domain specification
#[derive(Debug, Clone)]
pub struct Domain2D {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Domain2D {
    pub fn new(x_range: (f64, f64), y_range: (f64, f64), nx: usize, ny: usize) -> Self {
        Self {
            x_min: x_range.0,
            x_max: x_range.1,
            y_min: y_range.0,
            y_max: y_range.1,
            nx,
            ny,
        }
    }

    pub fn dx(&self) -> f64 {
        (self.x_max - self.x_min) / (self.nx - 1) as f64
    }

    pub fn dy(&self) -> f64 {
        (self.y_max - self.y_min) / (self.ny - 1) as f64
    }

    pub fn x(&self, i: usize) -> f64 {
        self.x_min + i as f64 * self.dx()
    }

    pub fn y(&self, j: usize) -> f64 {
        self.y_min + j as f64 * self.dy()
    }

    /// Total number of grid points
    pub fn size(&self) -> usize {
        self.nx * self.ny
    }
}

/// Boundary condition type
#[derive(Debug, Clone)]
pub enum BoundaryCondition {
    /// Dirichlet: u = value
    Dirichlet(f64),
    /// Neumann: ∂u/∂n = value
    Neumann(f64),
    /// Periodic
    Periodic,
    /// Robin: a*u + b*∂u/∂n = c
    Robin { a: f64, b: f64, c: f64 },
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        BoundaryCondition::Dirichlet(0.0)
    }
}

/// Boundary conditions for 1D domain
#[derive(Debug, Clone, Default)]
pub struct Boundary1D {
    pub left: BoundaryCondition,
    pub right: BoundaryCondition,
}

/// Boundary conditions for 2D domain
#[derive(Debug, Clone, Default)]
pub struct Boundary2D {
    pub left: BoundaryCondition,   // x = x_min
    pub right: BoundaryCondition,  // x = x_max
    pub bottom: BoundaryCondition, // y = y_min
    pub top: BoundaryCondition,    // y = y_max
}

/// PDE solution containing time evolution
#[derive(Debug, Clone)]
pub struct PDESolution1D {
    /// Time points
    pub t: Vec<f64>,
    /// Spatial grid
    pub x: Vec<f64>,
    /// Solution at each time: u[time_idx][space_idx]
    pub u: Vec<Vec<f64>>,
    /// Domain
    pub domain: Domain1D,
}

impl PDESolution1D {
    /// Get solution at specific time
    pub fn at_time(&self, t_query: f64) -> Option<Vec<f64>> {
        // Find closest time index
        let idx = self
            .t
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - t_query).abs())
                    .partial_cmp(&((*b - t_query).abs()))
                    .unwrap()
            })
            .map(|(i, _)| i)?;

        Some(self.u[idx].clone())
    }

    /// Get solution at specific point over time
    pub fn at_point(&self, x_query: f64) -> Option<Vec<f64>> {
        // Find closest spatial index
        let idx = self
            .x
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - x_query).abs())
                    .partial_cmp(&((*b - x_query).abs()))
                    .unwrap()
            })
            .map(|(i, _)| i)?;

        Some(self.u.iter().map(|ut| ut[idx]).collect())
    }

    /// Final solution
    pub fn final_state(&self) -> Option<&Vec<f64>> {
        self.u.last()
    }
}

/// 2D PDE solution
#[derive(Debug, Clone)]
pub struct PDESolution2D {
    pub t: Vec<f64>,
    /// Solution at each time: u[time_idx] is a flattened 2D array (row-major)
    pub u: Vec<Vec<f64>>,
    pub domain: Domain2D,
}

impl PDESolution2D {
    /// Get value at (i, j) for time index t_idx
    pub fn get(&self, t_idx: usize, i: usize, j: usize) -> Option<f64> {
        let idx = i * self.domain.ny + j;
        self.u.get(t_idx).and_then(|ut| ut.get(idx).copied())
    }

    /// Final solution as 2D array
    pub fn final_state_2d(&self) -> Option<Vec<Vec<f64>>> {
        let ut = self.u.last()?;
        let mut result = vec![vec![0.0; self.domain.ny]; self.domain.nx];
        for i in 0..self.domain.nx {
            for j in 0..self.domain.ny {
                result[i][j] = ut[i * self.domain.ny + j];
            }
        }
        Some(result)
    }
}

/// PDE solver options
#[derive(Debug, Clone)]
pub struct PDEOptions {
    /// Time step (0 for automatic based on CFL)
    pub dt: f64,
    /// CFL number for stability (default 0.4)
    pub cfl: f64,
    /// Store every nth time step
    pub save_every: usize,
    /// Maximum iterations (for implicit methods)
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for PDEOptions {
    fn default() -> Self {
        Self {
            dt: 0.0,
            cfl: 0.4,
            save_every: 1,
            max_iter: 1000,
            tolerance: 1e-8,
        }
    }
}

// ============================================================================
// 1D Heat Equation: ∂u/∂t = α ∂²u/∂x²
// ============================================================================

/// Solve 1D heat equation using explicit FTCS (Forward Time Central Space)
pub fn heat_equation_1d_explicit<F>(
    domain: &Domain1D,
    boundary: &Boundary1D,
    alpha: f64,
    initial: F,
    t_final: f64,
    options: &PDEOptions,
) -> PDESolution1D
where
    F: Fn(f64) -> f64,
{
    let dx = domain.dx();
    let nx = domain.nx;

    // CFL condition: dt <= dx² / (2α)
    let dt_max = dx * dx / (2.0 * alpha);
    let dt = if options.dt > 0.0 {
        options.dt.min(dt_max * options.cfl)
    } else {
        dt_max * options.cfl
    };

    let r = alpha * dt / (dx * dx);

    // Initialize
    let x_grid = domain.grid();
    let mut u: Vec<f64> = x_grid.iter().map(|&x| initial(x)).collect();
    let mut u_new = vec![0.0; nx];

    let mut solution = PDESolution1D {
        t: vec![0.0],
        x: x_grid.clone(),
        u: vec![u.clone()],
        domain: domain.clone(),
    };

    let mut t = 0.0;
    let mut step = 0;

    while t < t_final {
        // Apply boundary conditions
        apply_boundary_1d(&mut u, boundary, dx);

        // FTCS scheme: u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
        for i in 1..(nx - 1) {
            u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
        }

        // Copy boundary values
        u_new[0] = u[0];
        u_new[nx - 1] = u[nx - 1];

        std::mem::swap(&mut u, &mut u_new);
        t += dt;
        step += 1;

        if step % options.save_every == 0 {
            solution.t.push(t);
            solution.u.push(u.clone());
        }
    }

    // Ensure final state is saved
    if solution.t.last() != Some(&t) {
        solution.t.push(t);
        solution.u.push(u);
    }

    solution
}

/// Solve 1D heat equation using Crank-Nicolson (implicit, unconditionally stable)
pub fn heat_equation_1d_crank_nicolson<F>(
    domain: &Domain1D,
    boundary: &Boundary1D,
    alpha: f64,
    initial: F,
    t_final: f64,
    options: &PDEOptions,
) -> PDESolution1D
where
    F: Fn(f64) -> f64,
{
    let dx = domain.dx();
    let nx = domain.nx;

    // Can use larger time step since unconditionally stable
    let dt = if options.dt > 0.0 {
        options.dt
    } else {
        dx * dx / alpha * 2.0 // Larger than explicit
    };

    let r = alpha * dt / (dx * dx);

    // Initialize
    let x_grid = domain.grid();
    let mut u: Vec<f64> = x_grid.iter().map(|&x| initial(x)).collect();

    let mut solution = PDESolution1D {
        t: vec![0.0],
        x: x_grid.clone(),
        u: vec![u.clone()],
        domain: domain.clone(),
    };

    // Tridiagonal system coefficients
    // (1 + r) * u_new[i] - r/2 * (u_new[i-1] + u_new[i+1]) =
    // (1 - r) * u[i] + r/2 * (u[i-1] + u[i+1])

    let mut t = 0.0;
    let mut step = 0;

    while t < t_final {
        apply_boundary_1d(&mut u, boundary, dx);

        // Build RHS
        let mut rhs = vec![0.0; nx];
        for i in 1..(nx - 1) {
            rhs[i] = (1.0 - r) * u[i] + r / 2.0 * (u[i - 1] + u[i + 1]);
        }

        // Boundary conditions in RHS
        rhs[0] = u[0];
        rhs[nx - 1] = u[nx - 1];

        // Solve tridiagonal system
        let a = -r / 2.0; // sub-diagonal
        let b = 1.0 + r; // diagonal
        let c = -r / 2.0; // super-diagonal

        u = solve_tridiagonal(a, b, c, &rhs, boundary);

        t += dt;
        step += 1;

        if step % options.save_every == 0 {
            solution.t.push(t);
            solution.u.push(u.clone());
        }
    }

    if solution.t.last() != Some(&t) {
        solution.t.push(t);
        solution.u.push(u);
    }

    solution
}

/// Apply 1D boundary conditions
fn apply_boundary_1d(u: &mut [f64], boundary: &Boundary1D, dx: f64) {
    let n = u.len();

    match &boundary.left {
        BoundaryCondition::Dirichlet(val) => u[0] = *val,
        BoundaryCondition::Neumann(val) => u[0] = u[1] - val * dx,
        BoundaryCondition::Periodic => u[0] = u[n - 2],
        BoundaryCondition::Robin { a, b, c } => {
            // a*u + b*∂u/∂n = c, ∂u/∂n ≈ (u[1] - u[0])/dx (outward normal is -x)
            u[0] = (c * dx + b * u[1]) / (a * dx + b);
        }
    }

    match &boundary.right {
        BoundaryCondition::Dirichlet(val) => u[n - 1] = *val,
        BoundaryCondition::Neumann(val) => u[n - 1] = u[n - 2] + val * dx,
        BoundaryCondition::Periodic => u[n - 1] = u[1],
        BoundaryCondition::Robin { a, b, c } => {
            u[n - 1] = (c * dx + b * u[n - 2]) / (a * dx + b);
        }
    }
}

/// Solve tridiagonal system using Thomas algorithm
fn solve_tridiagonal(
    a: f64, // sub-diagonal (constant)
    b: f64, // diagonal (constant)
    c: f64, // super-diagonal (constant)
    rhs: &[f64],
    boundary: &Boundary1D,
) -> Vec<f64> {
    let n = rhs.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut x = vec![0.0; n];

    // Handle boundary - first row
    match &boundary.left {
        BoundaryCondition::Dirichlet(_) => {
            c_prime[0] = 0.0;
            d_prime[0] = rhs[0];
        }
        _ => {
            c_prime[0] = c / b;
            d_prime[0] = rhs[0] / b;
        }
    }

    // Forward sweep
    for i in 1..(n - 1) {
        let denom = b - a * c_prime[i - 1];
        c_prime[i] = c / denom;
        d_prime[i] = (rhs[i] - a * d_prime[i - 1]) / denom;
    }

    // Last row (boundary)
    match &boundary.right {
        BoundaryCondition::Dirichlet(_) => {
            d_prime[n - 1] = rhs[n - 1];
        }
        _ => {
            let denom = b - a * c_prime[n - 2];
            d_prime[n - 1] = (rhs[n - 1] - a * d_prime[n - 2]) / denom;
        }
    }

    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for i in (0..(n - 1)).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    x
}

// ============================================================================
// 1D Wave Equation: ∂²u/∂t² = c² ∂²u/∂x²
// ============================================================================

/// Solve 1D wave equation using explicit CTCS scheme
pub fn wave_equation_1d<F, G>(
    domain: &Domain1D,
    boundary: &Boundary1D,
    c: f64,        // wave speed
    initial_u: F,  // u(x, 0)
    initial_ut: G, // ∂u/∂t(x, 0)
    t_final: f64,
    options: &PDEOptions,
) -> PDESolution1D
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let dx = domain.dx();
    let nx = domain.nx;

    // CFL condition: dt <= dx / c
    let dt_max = dx / c;
    let dt = if options.dt > 0.0 {
        options.dt.min(dt_max * options.cfl)
    } else {
        dt_max * options.cfl
    };

    let r = c * dt / dx;
    let r2 = r * r;

    // Initialize
    let x_grid = domain.grid();
    let mut u_prev: Vec<f64> = x_grid.iter().map(|&x| initial_u(x)).collect();

    // First step using initial velocity
    // u[1] = u[0] + dt * ut[0] + (dt²/2) * c² * uxx[0]
    let mut u: Vec<f64> = vec![0.0; nx];
    for i in 1..(nx - 1) {
        let uxx = (u_prev[i + 1] - 2.0 * u_prev[i] + u_prev[i - 1]) / (dx * dx);
        u[i] = u_prev[i] + dt * initial_ut(x_grid[i]) + 0.5 * dt * dt * c * c * uxx;
    }
    apply_boundary_1d(&mut u, boundary, dx);

    let mut u_next = vec![0.0; nx];

    let mut solution = PDESolution1D {
        t: vec![0.0, dt],
        x: x_grid.clone(),
        u: vec![u_prev.clone(), u.clone()],
        domain: domain.clone(),
    };

    let mut t = dt;
    let mut step = 1;

    while t < t_final {
        // CTCS: u_next[i] = 2*u[i] - u_prev[i] + r² * (u[i+1] - 2*u[i] + u[i-1])
        for i in 1..(nx - 1) {
            u_next[i] = 2.0 * u[i] - u_prev[i] + r2 * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
        }

        apply_boundary_1d(&mut u_next, boundary, dx);

        // Rotate arrays
        std::mem::swap(&mut u_prev, &mut u);
        std::mem::swap(&mut u, &mut u_next);

        t += dt;
        step += 1;

        if step % options.save_every == 0 {
            solution.t.push(t);
            solution.u.push(u.clone());
        }
    }

    if solution.t.last() != Some(&t) {
        solution.t.push(t);
        solution.u.push(u);
    }

    solution
}

// ============================================================================
// 1D Advection Equation: ∂u/∂t + v ∂u/∂x = 0
// ============================================================================

/// Solve 1D advection equation using upwind scheme
pub fn advection_equation_1d<F>(
    domain: &Domain1D,
    boundary: &Boundary1D,
    velocity: f64,
    initial: F,
    t_final: f64,
    options: &PDEOptions,
) -> PDESolution1D
where
    F: Fn(f64) -> f64,
{
    let dx = domain.dx();
    let nx = domain.nx;

    // CFL condition: dt <= dx / |v|
    let dt_max = dx / velocity.abs();
    let dt = if options.dt > 0.0 {
        options.dt.min(dt_max * options.cfl)
    } else {
        dt_max * options.cfl
    };

    let cfl = velocity * dt / dx;

    let x_grid = domain.grid();
    let mut u: Vec<f64> = x_grid.iter().map(|&x| initial(x)).collect();
    let mut u_new = vec![0.0; nx];

    let mut solution = PDESolution1D {
        t: vec![0.0],
        x: x_grid.clone(),
        u: vec![u.clone()],
        domain: domain.clone(),
    };

    let mut t = 0.0;
    let mut step = 0;

    while t < t_final {
        apply_boundary_1d(&mut u, boundary, dx);

        // Upwind scheme
        if velocity > 0.0 {
            // Information flows from left to right
            for i in 1..nx {
                u_new[i] = u[i] - cfl * (u[i] - u[i - 1]);
            }
            u_new[0] = u[0]; // Inflow boundary
        } else {
            // Information flows from right to left
            for i in 0..(nx - 1) {
                u_new[i] = u[i] - cfl * (u[i + 1] - u[i]);
            }
            u_new[nx - 1] = u[nx - 1]; // Inflow boundary
        }

        std::mem::swap(&mut u, &mut u_new);
        t += dt;
        step += 1;

        if step % options.save_every == 0 {
            solution.t.push(t);
            solution.u.push(u.clone());
        }
    }

    if solution.t.last() != Some(&t) {
        solution.t.push(t);
        solution.u.push(u);
    }

    solution
}

// ============================================================================
// 2D Heat Equation: ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)
// ============================================================================

/// Solve 2D heat equation using explicit FTCS
pub fn heat_equation_2d_explicit<F>(
    domain: &Domain2D,
    boundary: &Boundary2D,
    alpha: f64,
    initial: F,
    t_final: f64,
    options: &PDEOptions,
) -> PDESolution2D
where
    F: Fn(f64, f64) -> f64,
{
    let dx = domain.dx();
    let dy = domain.dy();
    let nx = domain.nx;
    let ny = domain.ny;

    // CFL: dt <= 1/(2α) * (1/dx² + 1/dy²)^(-1)
    let dt_max = 0.5 / alpha / (1.0 / (dx * dx) + 1.0 / (dy * dy));
    let dt = if options.dt > 0.0 {
        options.dt.min(dt_max * options.cfl)
    } else {
        dt_max * options.cfl
    };

    let rx = alpha * dt / (dx * dx);
    let ry = alpha * dt / (dy * dy);

    // Initialize (flattened row-major)
    let mut u: Vec<f64> = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            u.push(initial(domain.x(i), domain.y(j)));
        }
    }
    let mut u_new = vec![0.0; nx * ny];

    let idx = |i: usize, j: usize| i * ny + j;

    let mut solution = PDESolution2D {
        t: vec![0.0],
        u: vec![u.clone()],
        domain: domain.clone(),
    };

    let mut t = 0.0;
    let mut step = 0;

    while t < t_final {
        // Apply boundary conditions
        apply_boundary_2d(&mut u, domain, boundary);

        // Interior points
        for i in 1..(nx - 1) {
            for j in 1..(ny - 1) {
                let uxx = u[idx(i + 1, j)] - 2.0 * u[idx(i, j)] + u[idx(i - 1, j)];
                let uyy = u[idx(i, j + 1)] - 2.0 * u[idx(i, j)] + u[idx(i, j - 1)];
                u_new[idx(i, j)] = u[idx(i, j)] + rx * uxx + ry * uyy;
            }
        }

        // Copy boundaries
        for i in 0..nx {
            u_new[idx(i, 0)] = u[idx(i, 0)];
            u_new[idx(i, ny - 1)] = u[idx(i, ny - 1)];
        }
        for j in 0..ny {
            u_new[idx(0, j)] = u[idx(0, j)];
            u_new[idx(nx - 1, j)] = u[idx(nx - 1, j)];
        }

        std::mem::swap(&mut u, &mut u_new);
        t += dt;
        step += 1;

        if step % options.save_every == 0 {
            solution.t.push(t);
            solution.u.push(u.clone());
        }
    }

    if solution.t.last() != Some(&t) {
        solution.t.push(t);
        solution.u.push(u);
    }

    solution
}

/// Apply 2D boundary conditions
fn apply_boundary_2d(u: &mut [f64], domain: &Domain2D, boundary: &Boundary2D) {
    let nx = domain.nx;
    let ny = domain.ny;
    let idx = |i: usize, j: usize| i * ny + j;

    // Left boundary (x = x_min)
    for j in 0..ny {
        match &boundary.left {
            BoundaryCondition::Dirichlet(val) => u[idx(0, j)] = *val,
            BoundaryCondition::Neumann(val) => u[idx(0, j)] = u[idx(1, j)] - val * domain.dx(),
            BoundaryCondition::Periodic => u[idx(0, j)] = u[idx(nx - 2, j)],
            _ => {}
        }
    }

    // Right boundary (x = x_max)
    for j in 0..ny {
        match &boundary.right {
            BoundaryCondition::Dirichlet(val) => u[idx(nx - 1, j)] = *val,
            BoundaryCondition::Neumann(val) => {
                u[idx(nx - 1, j)] = u[idx(nx - 2, j)] + val * domain.dx()
            }
            BoundaryCondition::Periodic => u[idx(nx - 1, j)] = u[idx(1, j)],
            _ => {}
        }
    }

    // Bottom boundary (y = y_min)
    for i in 0..nx {
        match &boundary.bottom {
            BoundaryCondition::Dirichlet(val) => u[idx(i, 0)] = *val,
            BoundaryCondition::Neumann(val) => u[idx(i, 0)] = u[idx(i, 1)] - val * domain.dy(),
            BoundaryCondition::Periodic => u[idx(i, 0)] = u[idx(i, ny - 2)],
            _ => {}
        }
    }

    // Top boundary (y = y_max)
    for i in 0..nx {
        match &boundary.top {
            BoundaryCondition::Dirichlet(val) => u[idx(i, ny - 1)] = *val,
            BoundaryCondition::Neumann(val) => {
                u[idx(i, ny - 1)] = u[idx(i, ny - 2)] + val * domain.dy()
            }
            BoundaryCondition::Periodic => u[idx(i, ny - 1)] = u[idx(i, 1)],
            _ => {}
        }
    }
}

// ============================================================================
// Diffusion-Reaction: ∂u/∂t = D∇²u + f(u)
// ============================================================================

/// Solve 1D diffusion-reaction equation
pub fn diffusion_reaction_1d<F, R>(
    domain: &Domain1D,
    boundary: &Boundary1D,
    diffusion: f64,
    reaction: R,
    initial: F,
    t_final: f64,
    options: &PDEOptions,
) -> PDESolution1D
where
    F: Fn(f64) -> f64,
    R: Fn(f64) -> f64,
{
    let dx = domain.dx();
    let nx = domain.nx;

    let dt_max = dx * dx / (2.0 * diffusion);
    let dt = if options.dt > 0.0 {
        options.dt.min(dt_max * options.cfl)
    } else {
        dt_max * options.cfl
    };

    let r = diffusion * dt / (dx * dx);

    let x_grid = domain.grid();
    let mut u: Vec<f64> = x_grid.iter().map(|&x| initial(x)).collect();
    let mut u_new = vec![0.0; nx];

    let mut solution = PDESolution1D {
        t: vec![0.0],
        x: x_grid.clone(),
        u: vec![u.clone()],
        domain: domain.clone(),
    };

    let mut t = 0.0;
    let mut step = 0;

    while t < t_final {
        apply_boundary_1d(&mut u, boundary, dx);

        for i in 1..(nx - 1) {
            let diffusion_term = r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
            let reaction_term = dt * reaction(u[i]);
            u_new[i] = u[i] + diffusion_term + reaction_term;
        }

        u_new[0] = u[0];
        u_new[nx - 1] = u[nx - 1];

        std::mem::swap(&mut u, &mut u_new);
        t += dt;
        step += 1;

        if step % options.save_every == 0 {
            solution.t.push(t);
            solution.u.push(u.clone());
        }
    }

    if solution.t.last() != Some(&t) {
        solution.t.push(t);
        solution.u.push(u);
    }

    solution
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_domain_1d() {
        let domain = Domain1D::new(0.0, 1.0, 11);
        assert_eq!(domain.dx(), 0.1);
        assert_eq!(domain.x(0), 0.0);
        assert_eq!(domain.x(10), 1.0);
    }

    #[test]
    fn test_heat_equation_explicit() {
        // Heat equation with zero boundary conditions
        // Initial: u(x,0) = sin(πx)
        // Exact: u(x,t) = exp(-π²αt) * sin(πx)

        let domain = Domain1D::new(0.0, 1.0, 51);
        let boundary = Boundary1D {
            left: BoundaryCondition::Dirichlet(0.0),
            right: BoundaryCondition::Dirichlet(0.0),
        };
        let alpha = 0.01;
        let t_final = 0.5;

        let solution = heat_equation_1d_explicit(
            &domain,
            &boundary,
            alpha,
            |x| (PI * x).sin(),
            t_final,
            &PDEOptions::default(),
        );

        // Check that solution decays
        let initial_max: f64 = solution.u[0].iter().cloned().fold(0.0, f64::max);
        let final_max: f64 = solution
            .final_state()
            .unwrap()
            .iter()
            .cloned()
            .fold(0.0, f64::max);

        assert!(final_max < initial_max, "Solution should decay");

        // Check approximate exponential decay
        let expected_decay = (-PI * PI * alpha * t_final).exp();
        let actual_decay = final_max / initial_max;
        assert!(
            (actual_decay - expected_decay).abs() < 0.1,
            "Decay rate: expected {}, got {}",
            expected_decay,
            actual_decay
        );
    }

    #[test]
    fn test_heat_equation_crank_nicolson() {
        let domain = Domain1D::new(0.0, 1.0, 51);
        let boundary = Boundary1D {
            left: BoundaryCondition::Dirichlet(0.0),
            right: BoundaryCondition::Dirichlet(0.0),
        };

        let solution = heat_equation_1d_crank_nicolson(
            &domain,
            &boundary,
            0.01,
            |x| (PI * x).sin(),
            0.5,
            &PDEOptions::default(),
        );

        assert!(solution.u.len() > 1);
        let final_max: f64 = solution
            .final_state()
            .unwrap()
            .iter()
            .cloned()
            .fold(0.0, f64::max);
        assert!(final_max < 1.0);
    }

    #[test]
    fn test_wave_equation() {
        // Standing wave with fixed ends
        let domain = Domain1D::new(0.0, 1.0, 51);
        let boundary = Boundary1D {
            left: BoundaryCondition::Dirichlet(0.0),
            right: BoundaryCondition::Dirichlet(0.0),
        };

        let solution = wave_equation_1d(
            &domain,
            &boundary,
            1.0,
            |x| (PI * x).sin(), // Initial displacement
            |_| 0.0,            // Zero initial velocity
            2.0,
            &PDEOptions::default(),
        );

        assert!(solution.u.len() > 1);

        // Wave should oscillate (energy conserved approximately)
        let initial_energy: f64 = solution.u[0].iter().map(|u| u * u).sum();
        let final_energy: f64 = solution.final_state().unwrap().iter().map(|u| u * u).sum();

        // Energy should be approximately conserved
        assert!(
            (final_energy / initial_energy - 1.0).abs() < 0.2,
            "Energy should be approximately conserved"
        );
    }

    #[test]
    fn test_advection() {
        // Advect a Gaussian pulse
        let domain = Domain1D::new(0.0, 2.0, 101);
        let boundary = Boundary1D {
            left: BoundaryCondition::Dirichlet(0.0),
            right: BoundaryCondition::Dirichlet(0.0),
        };

        let solution = advection_equation_1d(
            &domain,
            &boundary,
            1.0,                                         // velocity
            |x| (-((x - 0.5) * (x - 0.5)) / 0.01).exp(), // Gaussian at x=0.5
            0.5,
            &PDEOptions::default(),
        );

        // Peak should have moved to approximately x = 1.0
        let final_state = solution.final_state().unwrap();
        let max_idx = final_state
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let peak_x = domain.x(max_idx);
        assert!(
            (peak_x - 1.0).abs() < 0.2,
            "Peak should be near x=1.0, got {}",
            peak_x
        );
    }

    #[test]
    fn test_heat_2d() {
        let domain = Domain2D::new((0.0, 1.0), (0.0, 1.0), 21, 21);
        let boundary = Boundary2D {
            left: BoundaryCondition::Dirichlet(0.0),
            right: BoundaryCondition::Dirichlet(0.0),
            bottom: BoundaryCondition::Dirichlet(0.0),
            top: BoundaryCondition::Dirichlet(0.0),
        };

        let solution = heat_equation_2d_explicit(
            &domain,
            &boundary,
            0.01,
            |x, y| (PI * x).sin() * (PI * y).sin(),
            0.1,
            &PDEOptions {
                save_every: 10,
                ..Default::default()
            },
        );

        assert!(solution.u.len() > 1);

        // Solution should decay
        let initial_max: f64 = solution.u[0].iter().cloned().fold(0.0, f64::max);
        let final_max: f64 = solution
            .u
            .last()
            .unwrap()
            .iter()
            .cloned()
            .fold(0.0, f64::max);
        assert!(final_max < initial_max);
    }

    #[test]
    fn test_diffusion_reaction() {
        // Fisher-KPP equation: ∂u/∂t = D∇²u + u(1-u)
        let domain = Domain1D::new(0.0, 10.0, 101);
        let boundary = Boundary1D {
            left: BoundaryCondition::Dirichlet(1.0),
            right: BoundaryCondition::Neumann(0.0),
        };

        let solution = diffusion_reaction_1d(
            &domain,
            &boundary,
            0.5,                                 // Higher diffusion
            |u| u * (1.0 - u),                   // Logistic growth
            |x| if x < 2.0 { 0.9 } else { 0.1 }, // Step initial condition
            10.0,                                // Longer time
            &PDEOptions {
                save_every: 100,
                ..Default::default()
            },
        );

        assert!(solution.u.len() > 1);

        // Front should propagate - check that wave has moved
        let final_state = solution.final_state().unwrap();
        let initial_state = &solution.u[0];

        // Compare: solution should have evolved
        let diff: f64 = final_state
            .iter()
            .zip(initial_state.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff > 1.0, "Solution should evolve from initial state");
    }
}
