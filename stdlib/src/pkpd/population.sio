//! Population pharmacokinetic modeling

use super::compartment::{PKParameters, simulate_pk, DoseEvent, PKResult}
use prob::distributions::{Distribution, Normal, LogNormal, MultivariateNormal}
use prob::mcmc::{MetropolisHastings, MCMCSample}
use linalg::{Vector, Matrix}
use units::{mg, L, h, mg_L, L_h, h_inv}

/// Individual in a population PK study
pub struct Individual {
    /// Subject ID
    pub id: string,
    
    /// Covariates (age, weight, sex, etc.)
    pub covariates: Vector<f64>,
    
    /// Dosing history
    pub doses: Vec<DoseEvent>,
    
    /// Observed concentrations
    pub observations: Vec<Observation>,
    
    /// Individual PK parameters (if estimated)
    pub pk_params: Option<PKParameters>,
}

/// Pharmacokinetic observation
pub struct Observation {
    /// Time of observation
    pub time: f64: h,
    
    /// Observed concentration
    pub concentration: f64: mg_L,
    
    /// Compartment (usually 1 for plasma)
    pub compartment: usize,
    
    /// Measurement error (if known)
    pub error: Option<f64>,
    
    /// Below limit of quantification flag
    pub bloq: bool,
}

/// Population PK model structure
pub struct PopulationPKModel {
    /// Fixed effects (typical values)
    pub theta: Vector<f64>,
    
    /// Between-subject variability (omega matrix)
    pub omega: Matrix<f64>,
    
    /// Residual error parameters
    pub sigma: Vector<f64>,
    
    /// Covariate model
    pub covariate_effects: CovariateModel,
    
    /// Parameter names
    pub param_names: Vec<string>,
    
    /// Covariate names
    pub covariate_names: Vec<string>,
}

/// Covariate effects on PK parameters
pub struct CovariateModel {
    /// Covariate effects matrix (param x covariate)
    pub effects: Matrix<f64>,
    
    /// Reference covariate values
    pub reference_values: Vector<f64>,
    
    /// Covariate transformation types
    pub transformations: Vec<CovariateTransform>,
}

/// Types of covariate transformations
pub enum CovariateTransform {
    /// Linear: theta * (cov - ref)
    Linear,
    
    /// Power: (cov / ref)^theta
    Power,
    
    /// Exponential: exp(theta * (cov - ref))
    Exponential,
    
    /// Categorical: theta if cov == category, 0 otherwise
    Categorical(f64),
}

impl PopulationPKModel {
    /// Create a new population PK model
    pub fn new(
        n_params: usize,
        n_covariates: usize,
        param_names: Vec<string>,
        covariate_names: Vec<string>,
    ) -> Self {
        PopulationPKModel {
            theta: Vector::zeros(n_params),
            omega: Matrix::eye(n_params) * 0.1, // 10% CV
            sigma: Vector::from_slice(&[0.1, 0.1]), // Proportional + additive error
            covariate_effects: CovariateModel {
                effects: Matrix::zeros(n_params, n_covariates),
                reference_values: Vector::zeros(n_covariates),
                transformations: vec![CovariateTransform::Power; n_covariates],
            },
            param_names,
            covariate_names,
        }
    }
    
    /// Sample individual parameters from population distribution
    pub fn sample_individual(
        &self,
        covariates: &Vector<f64>,
        rng: &mut impl rand::Rng,
    ) -> Vector<f64> with Prob {
        // Sample random effects
        let mvn = MultivariateNormal::new(
            Vector::zeros(self.theta.len()),
            self.omega.clone()
        ).unwrap();
        let eta = mvn.sample(rng);
        
        // Calculate individual parameters
        let mut params = self.theta.clone();
        
        // Apply random effects (log-normal distribution)
        for i in 0..params.len() {
            params[i] *= eta[i].exp();
        }
        
        // Apply covariate effects
        for i in 0..params.len() {
            for j in 0..covariates.len() {
                let effect = self.covariate_effects.effects[(i, j)];
                if effect != 0.0 {
                    let cov_val = covariates[j];
                    let ref_val = self.covariate_effects.reference_values[j];
                    
                    match self.covariate_effects.transformations[j] {
                        CovariateTransform::Linear => {
                            params[i] += effect * (cov_val - ref_val);
                        }
                        CovariateTransform::Power => {
                            params[i] *= (cov_val / ref_val).powf(effect);
                        }
                        CovariateTransform::Exponential => {
                            params[i] *= (effect * (cov_val - ref_val)).exp();
                        }
                        CovariateTransform::Categorical(category) => {
                            if (cov_val - category).abs() < 1e-6 {
                                params[i] *= effect.exp();
                            }
                        }
                    }
                }
            }
        }
        
        params
    }
    
    /// Predict concentrations for an individual
    pub fn predict_individual(
        &self,
        individual: &Individual,
        times: &[f64],
    ) -> PKResult with Alloc {
        // Get individual parameters
        let params = if let Some(ref pk_params) = individual.pk_params {
            // Use fitted individual parameters
            vec![pk_params.cl, pk_params.v1, 
                 pk_params.v2.unwrap_or(0.0), pk_params.q2.unwrap_or(0.0)]
        } else {
            // Sample from population
            let mut rng = rand::thread_rng();
            self.sample_individual(&individual.covariates, &mut rng)
        };
        
        // Create PK parameters structure
        let pk_params = PKParameters::two_compartment(
            params[0], params[1], params[2], params[3]
        );
        
        simulate_pk(&pk_params, &individual.doses, times)
    }
    
    /// Compute log-likelihood for an individual
    pub fn log_likelihood(&self, individual: &Individual) -> f64 {
        let mut log_lik = 0.0;
        
        // Get prediction times
        let obs_times: Vec<f64> = individual.observations.iter()
            .map(|obs| obs.time)
            .collect();
        
        let prediction = self.predict_individual(individual, &obs_times);
        
        // Compare predictions to observations
        for (i, obs) in individual.observations.iter().enumerate() {
            if !obs.bloq {
                let pred_conc = prediction.concentrations[(i, 0)]; // Central compartment
                let residual = obs.concentration - pred_conc;
                
                // Combined error model: sigma[0] * pred + sigma[1]
                let error_var = (self.sigma[0] * pred_conc).powi(2) + self.sigma[1].powi(2);
                
                // Normal likelihood
                log_lik += -0.5 * (residual.powi(2) / error_var + error_var.ln() + 
                                  (2.0 * std::math::PI).ln());
            }
        }
        
        log_lik
    }
    
    /// Fit population model using MCMC
    pub fn fit_mcmc(
        &mut self,
        individuals: &[Individual],
        n_samples: usize,
        rng: &mut impl rand::Rng,
    ) -> MCMCSample {
        // Define log-posterior
        let log_posterior = |params: &Vector<f64>| -> f64 {
            // Update model parameters
            let n_theta = self.theta.len();
            let n_omega = n_theta * (n_theta + 1) / 2;
            
            // Extract parameters
            for i in 0..n_theta {
                self.theta[i] = params[i];
            }
            
            // Extract omega (lower triangular)
            let mut idx = n_theta;
            for i in 0..n_theta {
                for j in 0..=i {
                    self.omega[(i, j)] = params[idx];
                    self.omega[(j, i)] = params[idx]; // Symmetry
                    idx += 1;
                }
            }
            
            // Extract sigma
            for i in 0..self.sigma.len() {
                self.sigma[i] = params[idx + i];
            }
            
            // Compute log-likelihood
            let mut log_lik = 0.0;
            for individual in individuals {
                log_lik += self.log_likelihood(individual);
            }
            
            // Add priors
            let mut log_prior = 0.0;
            
            // Prior on theta (normal)
            for &theta_i in &self.theta {
                log_prior += Normal::new(0.0, 10.0).log_pdf(theta_i);
            }
            
            // Prior on omega diagonal (inverse gamma)
            for i in 0..n_theta {
                if self.omega[(i, i)] > 0.0 {
                    log_prior += -2.0 * self.omega[(i, i)].ln() - 1.0 / self.omega[(i, i)];
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            
            // Prior on sigma (inverse gamma)
            for &sigma_i in &self.sigma {
                if sigma_i > 0.0 {
                    log_prior += -2.0 * sigma_i.ln() - 1.0 / sigma_i;
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            
            log_lik + log_prior
        };
        
        // Initial parameters
        let n_params = self.theta.len() + self.theta.len() * (self.theta.len() + 1) / 2 + 
                      self.sigma.len();
        let mut x0 = Vector::zeros(n_params);
        
        // Initialize with current values
        let mut idx = 0;
        for i in 0..self.theta.len() {
            x0[idx] = self.theta[i];
            idx += 1;
        }
        for i in 0..self.theta.len() {
            for j in 0..=i {
                x0[idx] = self.omega[(i, j)];
                idx += 1;
            }
        }
        for i in 0..self.sigma.len() {
            x0[idx] = self.sigma[i];
            idx += 1;
        }
        
        // Run MCMC
        let mut sampler = MetropolisHastings::new(n_params);
        sampler.sample(log_posterior, &x0, n_samples, rng)
    }
}

/// Non-compartmental analysis
pub struct NCA {
    /// Concentration-time data
    pub time: Vector<f64>,
    pub concentration: Vector<f64>,
    
    /// Dose information
    pub dose: f64,
    pub dose_time: f64,
}

impl NCA {
    pub fn new(time: Vector<f64>, concentration: Vector<f64>, dose: f64) -> Self {
        NCA {
            time,
            concentration,
            dose,
            dose_time: 0.0,
        }
    }
    
    /// Calculate area under the curve using trapezoidal rule
    pub fn auc(&self, t_start: f64, t_end: f64) -> f64 {
        let mut auc = 0.0;
        
        for i in 1..self.time.len() {
            let t1 = self.time[i-1];
            let t2 = self.time[i];
            let c1 = self.concentration[i-1];
            let c2 = self.concentration[i];
            
            if t2 > t_start && t1 < t_end {
                let t_left = t1.max(t_start);
                let t_right = t2.min(t_end);
                
                // Linear interpolation for concentrations at boundaries
                let c_left = if t1 >= t_start { c1 } else {
                    c1 + (c2 - c1) * (t_start - t1) / (t2 - t1)
                };
                let c_right = if t2 <= t_end { c2 } else {
                    c1 + (c2 - c1) * (t_end - t1) / (t2 - t1)
                };
                
                auc += 0.5 * (c_left + c_right) * (t_right - t_left);
            }
        }
        
        auc
    }
    
    /// Calculate maximum concentration and time to maximum
    pub fn cmax_tmax(&self) -> (f64, f64) {
        let mut cmax = 0.0;
        let mut tmax = 0.0;
        
        for i in 0..self.concentration.len() {
            if self.concentration[i] > cmax {
                cmax = self.concentration[i];
                tmax = self.time[i];
            }
        }
        
        (cmax, tmax)
    }
    
    /// Calculate terminal elimination rate constant
    pub fn lambda_z(&self, n_points: usize) -> f64 {
        let n = self.concentration.len();
        if n < n_points { return 0.0; }
        
        // Use last n_points for linear regression on log(C) vs t
        let start_idx = n - n_points;
        
        let mut sum_t = 0.0;
        let mut sum_log_c = 0.0;
        let mut sum_t_log_c = 0.0;
        let mut sum_t2 = 0.0;
        let mut count = 0;
        
        for i in start_idx..n {
            if self.concentration[i] > 0.0 {
                let t = self.time[i];
                let log_c = self.concentration[i].ln();
                
                sum_t += t;
                sum_log_c += log_c;
                sum_t_log_c += t * log_c;
                sum_t2 += t * t;
                count += 1;
            }
        }
        
        if count < 2 { return 0.0; }
        
        let slope = (count as f64 * sum_t_log_c - sum_t * sum_log_c) /
                   (count as f64 * sum_t2 - sum_t * sum_t);
        
        -slope // Lambda_z is the negative slope
    }
    
    /// Calculate half-life
    pub fn half_life(&self) -> f64 {
        let lambda_z = self.lambda_z(3);
        if lambda_z > 0.0 {
            0.693 / lambda_z
        } else {
            f64::NAN
        }
    }
    
    /// Calculate clearance (CL = Dose / AUC_inf)
    pub fn clearance(&self) -> f64 {
        let auc_inf = self.auc_extrapolated();
        if auc_inf > 0.0 {
            self.dose / auc_inf
        } else {
            f64::NAN
        }
    }
    
    /// Calculate AUC extrapolated to infinity
    pub fn auc_extrapolated(&self) -> f64 {
        let t_last = self.time[self.time.len() - 1];
        let c_last = self.concentration[self.concentration.len() - 1];
        let lambda_z = self.lambda_z(3);
        
        let auc_obs = self.auc(0.0, t_last);
        
        if lambda_z > 0.0 && c_last > 0.0 {
            auc_obs + c_last / lambda_z
        } else {
            auc_obs
        }
    }
}
