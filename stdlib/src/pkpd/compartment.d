//! Pharmacokinetic compartment models

use linalg::{Vector, Matrix}
use numerics::ode::{ODESystem, odeint, ODESolution}
use units::{mg, L, h, mg_L, L_h, h_inv}

/// Pharmacokinetic parameters
pub struct PKParameters {
    /// Clearance (volume/time)
    pub cl: f64: L_h,

    /// Volume of distribution (central)
    pub v1: f64: L,

    /// Volume of distribution (peripheral 1)
    pub v2: Option<f64: L>,

    /// Volume of distribution (peripheral 2)
    pub v3: Option<f64: L>,

    /// Intercompartmental clearance 1-2
    pub q2: Option<f64: L_h>,

    /// Intercompartmental clearance 1-3
    pub q3: Option<f64: L_h>,

    /// Absorption rate constant
    pub ka: Option<f64: h_inv>,

    /// Bioavailability
    pub f: f64,
}

impl PKParameters {
    /// Create 1-compartment IV parameters
    pub fn one_compartment(cl: f64: L_h, v: f64: L) -> Self {
        PKParameters {
            cl,
            v1: v,
            v2: None,
            v3: None,
            q2: None,
            q3: None,
            ka: None,
            f: 1.0,
        }
    }

    /// Create 2-compartment IV parameters
    pub fn two_compartment(
        cl: f64: L_h,
        v1: f64: L,
        v2: f64: L,
        q: f64: L_h
    ) -> Self {
        PKParameters {
            cl,
            v1,
            v2: Some(v2),
            v3: None,
            q2: Some(q),
            q3: None,
            ka: None,
            f: 1.0,
        }
    }

    /// Create 1-compartment oral parameters
    pub fn one_compartment_oral(
        cl: f64: L_h,
        v: f64: L,
        ka: f64: h_inv,
        f: f64
    ) -> Self {
        PKParameters {
            cl,
            v1: v,
            v2: None,
            v3: None,
            q2: None,
            q3: None,
            ka: Some(ka),
            f,
        }
    }

    /// Calculate elimination rate constant
    pub fn ke(&self) -> f64: h_inv {
        self.cl / self.v1
    }

    /// Calculate half-life
    pub fn half_life(&self) -> f64: h {
        0.693 / self.ke()
    }

    /// Number of compartments
    pub fn n_compartments(&self) -> usize {
        let mut n = 1;
        if self.v2.is_some() { n += 1; }
        if self.v3.is_some() { n += 1; }
        if self.ka.is_some() { n += 1; } // Absorption compartment
        n
    }
}

/// Dosing event
pub struct DoseEvent {
    /// Time of dose
    pub time: f64: h,

    /// Amount
    pub amount: f64: mg,

    /// Compartment (0 = depot/absorption, 1 = central)
    pub cmt: usize,

    /// Infusion duration (0 = bolus)
    pub duration: f64: h,

    /// Bioavailability override
    pub bioav: Option<f64>,
}

impl DoseEvent {
    /// IV bolus dose
    pub fn iv_bolus(time: f64: h, amount: f64: mg) -> Self {
        DoseEvent {
            time,
            amount,
            cmt: 1,
            duration: 0.0: h,
            bioav: Some(1.0),
        }
    }

    /// IV infusion
    pub fn iv_infusion(time: f64: h, amount: f64: mg, duration: f64: h) -> Self {
        DoseEvent {
            time,
            amount,
            cmt: 1,
            duration,
            bioav: Some(1.0),
        }
    }

    /// Oral dose
    pub fn oral(time: f64: h, amount: f64: mg) -> Self {
        DoseEvent {
            time,
            amount,
            cmt: 0,  // Depot compartment
            duration: 0.0: h,
            bioav: None,  // Use parameter F
        }
    }
}

/// Compartment model ODE system
struct CompartmentODE {
    params: PKParameters,
    infusion_rate: f64,
}

impl ODESystem for CompartmentODE {
    fn eval(&self, t: f64, y: &Vector<f64>, dydt: &!Vector<f64>) {
        let n = y.len();

        match n {
            // 1-compartment (IV or oral)
            1 => {
                // dA1/dt = infusion - CL * C1
                let c1 = y[0] / self.params.v1;
                dydt[0] = self.infusion_rate - self.params.cl * c1;
            }

            // 1-compartment oral (depot + central)
            2 if self.params.ka.is_some() => {
                let ka = self.params.ka.unwrap();
                let c1 = y[1] / self.params.v1;

                // dA_depot/dt = -ka * A_depot
                dydt[0] = -ka * y[0];

                // dA1/dt = ka * A_depot - CL * C1
                dydt[1] = ka * y[0] + self.infusion_rate - self.params.cl * c1;
            }

            // 2-compartment (IV)
            2 => {
                let v2 = self.params.v2.unwrap();
                let q = self.params.q2.unwrap();

                let c1 = y[0] / self.params.v1;
                let c2 = y[1] / v2;

                // dA1/dt = infusion - CL * C1 - Q * (C1 - C2)
                dydt[0] = self.infusion_rate - self.params.cl * c1 - q * (c1 - c2);

                // dA2/dt = Q * (C1 - C2)
                dydt[1] = q * (c1 - c2);
            }

            // 2-compartment oral
            3 if self.params.ka.is_some() => {
                let ka = self.params.ka.unwrap();
                let v2 = self.params.v2.unwrap();
                let q = self.params.q2.unwrap();

                let c1 = y[1] / self.params.v1;
                let c2 = y[2] / v2;

                dydt[0] = -ka * y[0];
                dydt[1] = ka * y[0] + self.infusion_rate - self.params.cl * c1 - q * (c1 - c2);
                dydt[2] = q * (c1 - c2);
            }

            // 3-compartment
            3 => {
                let v2 = self.params.v2.unwrap();
                let v3 = self.params.v3.unwrap();
                let q2 = self.params.q2.unwrap();
                let q3 = self.params.q3.unwrap();

                let c1 = y[0] / self.params.v1;
                let c2 = y[1] / v2;
                let c3 = y[2] / v3;

                dydt[0] = self.infusion_rate - self.params.cl * c1
                    - q2 * (c1 - c2) - q3 * (c1 - c3);
                dydt[1] = q2 * (c1 - c2);
                dydt[2] = q3 * (c1 - c3);
            }

            _ => panic!("unsupported compartment configuration"),
        }
    }

    fn dim(&self) -> usize {
        self.params.n_compartments()
    }
}

/// PK simulation result
pub struct PKResult {
    /// Time points
    pub time: Vector<f64>,

    /// Amounts in each compartment
    pub amounts: Matrix<f64>,

    /// Concentrations in each compartment
    pub concentrations: Matrix<f64>,

    /// AUC (area under curve) up to each time
    pub auc: Vector<f64>,

    /// Peak concentration
    pub cmax: f64,

    /// Time of peak
    pub tmax: f64,
}

/// Simulate PK model
pub fn simulate_pk(
    params: &PKParameters,
    doses: &[DoseEvent],
    times: &[f64],
) -> PKResult with Alloc {
    let n_cmt = params.n_compartments();
    let n_times = times.len();

    let mut y = Vector::zeros(n_cmt);
    let mut result_amounts = Matrix::zeros(n_times, n_cmt);
    let mut result_conc = Matrix::zeros(n_times, n_cmt);
    let mut auc = Vector::zeros(n_times);

    let mut t = 0.0;
    let mut time_idx = 0;
    let mut dose_idx = 0;
    let mut current_auc = 0.0;

    // Determine depot compartment index
    let depot_idx = if params.ka.is_some() { 0 } else { usize::MAX };
    let central_idx = if params.ka.is_some() { 1 } else { 0 };

    // Sort doses by time
    let mut sorted_doses = doses.to_vec();
    sorted_doses.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    // Process events
    while time_idx < n_times {
        let next_obs = times[time_idx];
        let next_dose = if dose_idx < sorted_doses.len() {
            Some(sorted_doses[dose_idx].time)
        } else {
            None
        };

        // Determine next event
        let next_event = match next_dose {
            Some(td) if td <= next_obs => td,
            _ => next_obs,
        };

        // Integrate to next event
        if next_event > t {
            let ode = CompartmentODE {
                params: params.clone(),
                infusion_rate: 0.0,  // TODO: handle infusions
            };

            let sol = odeint(ode, &y, (t, next_event), "RK45");
            y = sol.y.row(sol.y.nrows() - 1).to_vector();

            // Update AUC (trapezoidal)
            let c_prev = result_conc[(time_idx.saturating_sub(1), central_idx)];
            let c_curr = y[central_idx] / params.v1;
            current_auc += 0.5 * (c_prev + c_curr) * (next_event - t);
        }

        t = next_event;

        // Apply dose if this is a dose event
        if let Some(td) = next_dose {
            if td == t {
                let dose = &sorted_doses[dose_idx];
                let bioav = dose.bioav.unwrap_or(params.f);
                let amt = dose.amount * bioav;

                if dose.cmt == 0 && depot_idx != usize::MAX {
                    y[depot_idx] += amt;
                } else {
                    y[central_idx] += amt;
                }

                dose_idx += 1;
            }
        }

        // Record observation
        if t == times[time_idx] {
            for j in 0..n_cmt {
                result_amounts[(time_idx, j)] = y[j];
                let vol = if j == central_idx { params.v1 }
                    else if j == depot_idx { 1.0 }  // Depot has no concentration
                    else if j == 2 && params.v2.is_some() { params.v2.unwrap() }
                    else if j == 3 && params.v3.is_some() { params.v3.unwrap() }
                    else { 1.0 };
                result_conc[(time_idx, j)] = y[j] / vol;
            }
            auc[time_idx] = current_auc;
            time_idx += 1;
        }
    }

    // Calculate Cmax and Tmax
    let mut cmax = 0.0;
    let mut tmax = 0.0;
    for i in 0..n_times {
        let c = result_conc[(i, central_idx)];
        if c > cmax {
            cmax = c;
            tmax = times[i];
        }
    }

    PKResult {
        time: Vector::from_slice(times),
        amounts: result_amounts,
        concentrations: result_conc,
        auc,
        cmax,
        tmax,
    }
}

/// Population PK model with random effects
pub struct PopulationPK {
    /// Fixed effects (typical values)
    pub theta: Vector<f64>,

    /// Between-subject variability (omega matrix)
    pub omega: Matrix<f64>,

    /// Residual variability
    pub sigma: f64,

    /// Covariate model
    pub covariates: Option<CovariateModel>,
}

/// Covariate effects
pub struct CovariateModel {
    /// Covariate names
    pub names: Vec<string>,

    /// Reference values
    pub reference: Vector<f64>,

    /// Effect coefficients
    pub coefficients: Matrix<f64>,
}

impl PopulationPK {
    /// Sample individual parameters
    pub fn sample_individual(
        &self,
        covariates: Option<&Vector<f64>>,
        rng: &!Rng
    ) -> Vector<f64> with Prob {
        // Sample random effects
        let mvn = prob::MultivariateNormal::new(
            Vector::zeros(self.omega.nrows()),
            self.omega.clone()
        ).unwrap();
        let eta = mvn.sample(rng);

        // Calculate individual parameters
        let mut params = self.theta.clone();
        for i in 0..params.len() {
            // Log-normal distribution for positive parameters
            params[i] *= exp(eta[i]);
        }

        // Apply covariate effects
        if let (Some(cov_model), Some(cov_values)) = (&self.covariates, covariates) {
            for i in 0..params.len() {
                for j in 0..cov_values.len() {
                    let effect = cov_model.coefficients[(i, j)];
                    let cov_centered = cov_values[j] - cov_model.reference[j];
                    params[i] *= pow(cov_centered, effect);
                }
            }
        }

        params
    }
}

/// Emax pharmacodynamic model
pub struct EmaxModel {
    /// Maximum effect
    pub emax: f64,

    /// EC50 (concentration at half-max effect)
    pub ec50: f64,

    /// Hill coefficient
    pub hill: f64,

    /// Baseline effect
    pub e0: f64,
}

impl EmaxModel {
    pub fn new(emax: f64, ec50: f64) -> Self {
        EmaxModel {
            emax,
            ec50,
            hill: 1.0,
            e0: 0.0,
        }
    }

    pub fn sigmoid(emax: f64, ec50: f64, hill: f64) -> Self {
        EmaxModel { emax, ec50, hill, e0: 0.0 }
    }

    /// Calculate effect at given concentration
    pub fn effect(&self, concentration: f64) -> f64 {
        self.e0 + self.emax * pow(concentration, self.hill)
            / (pow(self.ec50, self.hill) + pow(concentration, self.hill))
    }

    /// Inverse: concentration for given effect
    pub fn concentration_for_effect(&self, effect: f64) -> f64 {
        let e_normalized = (effect - self.e0) / self.emax;
        self.ec50 * pow(e_normalized / (1.0 - e_normalized), 1.0 / self.hill)
    }
}
