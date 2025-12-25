//! Numerical optimization algorithms

use linalg::{Vector, Matrix}
use autodiff::reverse::{Var, gradient, hessian}

/// Optimization result
pub struct OptResult {
    /// Final parameter values
    pub x: Vector<f64>,
    
    /// Final function value
    pub fun: f64,
    
    /// Final gradient
    pub grad: Vector<f64>,
    
    /// Number of function evaluations
    pub nfev: usize,
    
    /// Number of gradient evaluations
    pub njev: usize,
    
    /// Whether optimization converged
    pub success: bool,
    
    /// Termination message
    pub message: string,
}

/// Optimization configuration
pub struct OptConfig {
    /// Gradient tolerance
    pub gtol: f64,
    
    /// Function tolerance
    pub ftol: f64,
    
    /// Parameter tolerance
    pub xtol: f64,
    
    /// Maximum iterations
    pub maxiter: usize,
    
    /// Maximum function evaluations
    pub maxfev: usize,
}

impl Default for OptConfig {
    fn default() -> Self {
        OptConfig {
            gtol: 1e-5,
            ftol: 1e-9,
            xtol: 1e-9,
            maxiter: 1000,
            maxfev: 10000,
        }
    }
}

/// Gradient descent optimizer
pub struct GradientDescent {
    config: OptConfig,
    learning_rate: f64,
    momentum: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> Self {
        GradientDescent {
            config: OptConfig::default(),
            learning_rate,
            momentum: 0.0,
        }
    }
    
    pub fn with_momentum(learning_rate: f64, momentum: f64) -> Self {
        GradientDescent {
            config: OptConfig::default(),
            learning_rate,
            momentum,
        }
    }
    
    /// Minimize function using gradient descent
    pub fn minimize<F>(&self, f: F, x0: &Vector<f64>) -> OptResult
    where F: Fn(&Vector<Var>) -> Var + Clone
    {
        let mut x = x0.clone();
        let mut velocity = Vector::zeros(x.len());
        let mut nfev = 0;
        let mut njev = 0;
        
        for iter in 0..self.config.maxiter {
            // Compute gradient
            let grad = gradient(f.clone(), &x);
            njev += 1;
            
            // Check convergence
            let grad_norm = grad.norm();
            if grad_norm < self.config.gtol {
                let fun_val = {
                    let x_vars: Vector<Var> = x.iter().map(|&xi| Var::new(xi)).collect();
                    f(&x_vars).value()
                };
                nfev += 1;
                
                return OptResult {
                    x,
                    fun: fun_val,
                    grad,
                    nfev,
                    njev,
                    success: true,
                    message: "gradient tolerance reached".to_string(),
                };
            }
            
            // Update with momentum
            for i in 0..x.len() {
                velocity[i] = self.momentum * velocity[i] - self.learning_rate * grad[i];
                x[i] += velocity[i];
            }
        }
        
        // Max iterations reached
        let fun_val = {
            let x_vars: Vector<Var> = x.iter().map(|&xi| Var::new(xi)).collect();
            f(&x_vars).value()
        };
        let grad = gradient(f, &x);
        nfev += 1;
        njev += 1;
        
        OptResult {
            x,
            fun: fun_val,
            grad,
            nfev,
            njev,
            success: false,
            message: "maximum iterations exceeded".to_string(),
        }
    }
}

/// BFGS quasi-Newton optimizer
pub struct BFGS {
    config: OptConfig,
}

impl BFGS {
    pub fn new() -> Self {
        BFGS {
            config: OptConfig::default(),
        }
    }
    
    /// Minimize function using BFGS
    pub fn minimize<F>(&self, f: F, x0: &Vector<f64>) -> OptResult
    where F: Fn(&Vector<Var>) -> Var + Clone
    {
        let n = x0.len();
        let mut x = x0.clone();
        let mut h = Matrix::eye(n); // Inverse Hessian approximation
        let mut nfev = 0;
        let mut njev = 0;
        
        // Initial gradient
        let mut grad = gradient(f.clone(), &x);
        njev += 1;
        
        for iter in 0..self.config.maxiter {
            // Check convergence
            let grad_norm = grad.norm();
            if grad_norm < self.config.gtol {
                let fun_val = {
                    let x_vars: Vector<Var> = x.iter().map(|&xi| Var::new(xi)).collect();
                    f(&x_vars).value()
                };
                nfev += 1;
                
                return OptResult {
                    x,
                    fun: fun_val,
                    grad,
                    nfev,
                    njev,
                    success: true,
                    message: "gradient tolerance reached".to_string(),
                };
            }
            
            // Search direction: p = -H * grad
            let p = &h * &(-&grad);
            
            // Line search (simple backtracking)
            let mut alpha = 1.0;
            let c1 = 1e-4; // Armijo condition parameter
            
            let f_x = {
                let x_vars: Vector<Var> = x.iter().map(|&xi| Var::new(xi)).collect();
                f(&x_vars).value()
            };
            nfev += 1;
            
            let grad_dot_p = grad.dot(&p);
            
            for _ in 0..20 { // Max line search iterations
                let x_new = &x + &(&p * alpha);
                let f_new = {
                    let x_vars: Vector<Var> = x_new.iter().map(|&xi| Var::new(xi)).collect();
                    f(&x_vars).value()
                };
                nfev += 1;
                
                if f_new <= f_x + c1 * alpha * grad_dot_p {
                    break;
                }
                alpha *= 0.5;
            }
            
            // Update position
            let s = &p * alpha;
            let x_new = &x + &s;
            
            // New gradient
            let grad_new = gradient(f.clone(), &x_new);
            njev += 1;
            
            // BFGS update
            let y = &grad_new - &grad;
            let sy = s.dot(&y);
            
            if sy > 1e-10 { // Ensure positive definiteness
                let hy = &h * &y;
                let yhy = y.dot(&hy);
                
                // H = H + (sy + yHy)(ss^T)/(sy)^2 - (Hys^T + sy^TH)/(sy)
                for i in 0..n {
                    for j in 0..n {
                        h[(i, j)] += (sy + yhy) * s[i] * s[j] / (sy * sy)
                            - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                    }
                }
            }
            
            x = x_new;
            grad = grad_new;
        }
        
        // Max iterations reached
        let fun_val = {
            let x_vars: Vector<Var> = x.iter().map(|&xi| Var::new(xi)).collect();
            f(&x_vars).value()
        };
        nfev += 1;
        
        OptResult {
            x,
            fun: fun_val,
            grad,
            nfev,
            njev,
            success: false,
            message: "maximum iterations exceeded".to_string(),
        }
    }
}
