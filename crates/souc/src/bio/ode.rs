//! Declarative ODE model builder for biological/PBPK simulations.
//!
//! Bridges Sounio's biological domain to the native `runtime::ode` solver by
//! allowing researchers to declare compartmental models in terms of named
//! parameters rather than raw Butcher-tableau closures.
//!
//! # Example — two-compartment PBPK model
//!
//! ```text
//! let model = PBPKModel::new()
//!     .compartment("central",  0.5)   // initial concentration (mg/L)
//!     .compartment("tissue",   0.0)
//!     .rate("central", "tissue", 0.3) // k12  (1/h)
//!     .rate("tissue",  "central", 0.1) // k21
//!     .elimination("central", 0.2);   // kel  (1/h)
//!
//! let solution = model.solve((0.0, 24.0), SolverMethod::RK45);
//! ```

use crate::runtime::ode::{solve_rk45, ODESolution, SolverOptions};

/// A transfer rate between two named compartments.
#[derive(Debug, Clone)]
pub struct CompartmentRate {
    pub from: usize,
    pub to: usize,
    pub rate: f64,
}

/// A first-order elimination from one compartment.
#[derive(Debug, Clone)]
pub struct Elimination {
    pub compartment: usize,
    pub rate: f64,
}

/// Declarative multi-compartment PBPK / pharmacokinetic model.
///
/// Internally compiles down to a linear ODE:
///
/// ```text
/// dy_i/dt = Σ_j(k_ji * y_j) - (Σ_j(k_ij) + kel_i) * y_i
/// ```
#[derive(Debug, Clone)]
pub struct PBPKModel {
    pub names: Vec<String>,
    pub initial: Vec<f64>,
    pub rates: Vec<CompartmentRate>,
    pub eliminations: Vec<Elimination>,
}

impl PBPKModel {
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            initial: Vec::new(),
            rates: Vec::new(),
            eliminations: Vec::new(),
        }
    }

    /// Add a named compartment with an initial concentration/amount.
    pub fn compartment(mut self, name: &str, initial_value: f64) -> Self {
        self.names.push(name.to_string());
        self.initial.push(initial_value);
        self
    }

    /// Add a first-order inter-compartmental transfer rate (1/time units).
    pub fn rate(mut self, from: &str, to: &str, k: f64) -> Self {
        let fi = self.index_of(from);
        let ti = self.index_of(to);
        if let (Some(f), Some(t)) = (fi, ti) {
            self.rates.push(CompartmentRate { from: f, to: t, rate: k });
        }
        self
    }

    /// Add a first-order elimination from a compartment.
    pub fn elimination(mut self, compartment: &str, kel: f64) -> Self {
        if let Some(idx) = self.index_of(compartment) {
            self.eliminations.push(Elimination { compartment: idx, rate: kel });
        }
        self
    }

    /// Solve the model over `t_span = (t0, tf)` with RK45 adaptive stepping.
    pub fn solve(&self, t_span: (f64, f64), options: &SolverOptions) -> ODESolution {
        let n = self.initial.len();
        if n == 0 {
            return ODESolution::new();
        }

        let rates = self.rates.clone();
        let eliminations = self.eliminations.clone();

        let f = move |_t: f64, y: &[f64], dydt: &mut [f64]| {
            for i in 0..dydt.len() {
                dydt[i] = 0.0;
            }
            for r in &rates {
                let flux = r.rate * y[r.from];
                dydt[r.from] -= flux;
                dydt[r.to]   += flux;
            }
            for e in &eliminations {
                dydt[e.compartment] -= e.rate * y[e.compartment];
            }
        };

        solve_rk45(f, &self.initial, t_span, options)
    }

    /// Extract the mean and variance of the final-state concentrations as a
    /// simple statistical summary useful for downstream Bayesian inference.
    ///
    /// Returns `(mean, variance)` over all compartment values at `t_final`.
    pub fn final_state_stats(solution: &ODESolution) -> Option<(f64, f64)> {
        let state = solution.final_state()?;
        if state.is_empty() {
            return None;
        }
        let n = state.len() as f64;
        let mean = state.iter().sum::<f64>() / n;
        let var = state.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        Some((mean, var))
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }
}

impl Default for PBPKModel {
    fn default() -> Self {
        Self::new()
    }
}
