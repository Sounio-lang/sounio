//! Higher-order automatic differentiation

use super::forward::Dual;
use super::reverse::{Var, gradient, reset_tape};
use linalg::{Vector, Matrix}

/// Second-order dual number for computing Hessians
#[derive(Clone, Copy, Debug)]
pub struct Dual2 {
    /// Function value
    pub val: f64,
    
    /// First derivative
    pub d1: f64,
    
    /// Second derivative
    pub d2: f64,
}

impl Dual2 {
    pub fn new(val: f64, d1: f64, d2: f64) -> Self {
        Dual2 { val, d1, d2 }
    }
    
    pub fn constant(val: f64) -> Self {
        Dual2 { val, d1: 0.0, d2: 0.0 }
    }
    
    pub fn variable(val: f64) -> Self {
        Dual2 { val, d1: 1.0, d2: 0.0 }
    }
    
    pub fn add(self, other: Dual2) -> Dual2 {
        Dual2 {
            val: self.val + other.val,
            d1: self.d1 + other.d1,
            d2: self.d2 + other.d2,
        }
    }
    
    pub fn mul(self, other: Dual2) -> Dual2 {
        // Product rule for second derivatives: (fg)'' = f''g + 2f'g' + fg''
        Dual2 {
            val: self.val * other.val,
            d1: self.d1 * other.val + self.val * other.d1,
            d2: self.d2 * other.val + 2.0 * self.d1 * other.d1 + self.val * other.d2,
        }
    }
    
    pub fn exp(self) -> Dual2 {
        let e = self.val.exp();
        Dual2 {
            val: e,
            d1: self.d1 * e,
            d2: (self.d2 + self.d1 * self.d1) * e,
        }
    }
    
    pub fn sin(self) -> Dual2 {
        let s = self.val.sin();
        let c = self.val.cos();
        Dual2 {
            val: s,
            d1: self.d1 * c,
            d2: self.d2 * c - self.d1 * self.d1 * s,
        }
    }
    
    pub fn pow(self, n: f64) -> Dual2 {
        let v = self.val.powf(n);
        let v1 = n * self.val.powf(n - 1.0);
        let v2 = n * (n - 1.0) * self.val.powf(n - 2.0);
        
        Dual2 {
            val: v,
            d1: self.d1 * v1,
            d2: self.d2 * v1 + self.d1 * self.d1 * v2,
        }
    }
}

/// Compute Hessian using forward-over-forward mode
pub fn hessian_forward<F>(f: F, x: &Vector<f64>) -> Matrix<f64>
where F: Fn(&Vector<Dual2>) -> Dual2
{
    let n = x.len();
    let mut hess = Matrix::zeros(n, n);
    
    for i in 0..n {
        for j in i..n {
            // Set up dual numbers for second-order differentiation
            let mut x_dual = Vector::new(n);
            for k in 0..n {
                x_dual[k] = if k == i && k == j {
                    // Diagonal element: d²/dx_i²
                    Dual2::new(x[k], 1.0, 1.0)
                } else if k == i {
                    // First variable
                    Dual2::new(x[k], 1.0, 0.0)
                } else if k == j {
                    // Second variable
                    Dual2::new(x[k], 1.0, 0.0)
                } else {
                    // Constant
                    Dual2::constant(x[k])
                };
            }
            
            let result = f(&x_dual);
            
            if i == j {
                hess[(i, j)] = result.d2;
            } else {
                // Mixed partial: use the fact that d²f/dx_i dx_j = d²f/dx_j dx_i
                // We need to compute this separately
                let mut x_dual_mixed = Vector::new(n);
                for k in 0..n {
                    x_dual_mixed[k] = if k == i {
                        Dual2::new(x[k], 1.0, 0.0)
                    } else if k == j {
                        Dual2::new(x[k], 0.0, 1.0)  // Second derivative direction
                    } else {
                        Dual2::constant(x[k])
                    };
                }
                
                let mixed_result = f(&x_dual_mixed);
                hess[(i, j)] = mixed_result.d2;
                hess[(j, i)] = mixed_result.d2;  // Symmetry
            }
        }
    }
    
    hess
}

/// Compute Hessian using reverse-over-forward mode (more efficient)
pub fn hessian_mixed<F>(f: F, x: &Vector<f64>) -> Matrix<f64>
where F: Fn(&Vector<Var>) -> Var + Clone
{
    let n = x.len();
    let mut hess = Matrix::zeros(n, n);
    
    // Compute gradient first
    let grad = gradient(f.clone(), x);
    
    // For each gradient component, compute its gradient (Hessian row)
    for i in 0..n {
        let grad_i = move |x_vars: &Vector<Var>| -> Var {
            reset_tape();
            let result = f(x_vars);
            result.backward();
            x_vars[i].grad()
        };
        
        let hess_row = gradient(grad_i, x);
        for j in 0..n {
            hess[(i, j)] = hess_row[j];
        }
    }
    
    hess
}

/// Taylor series expansion using higher-order derivatives
pub struct TaylorSeries {
    /// Coefficients (0th, 1st, 2nd, ... order)
    pub coefficients: Vector<f64>,
    
    /// Center point
    pub center: f64,
    
    /// Order of expansion
    pub order: usize,
}

impl TaylorSeries {
    /// Evaluate Taylor series at point x
    pub fn eval(&self, x: f64) -> f64 {
        let dx = x - self.center;
        let mut result = 0.0;
        let mut dx_power = 1.0;
        let mut factorial = 1.0;
        
        for i in 0..=self.order {
            result += self.coefficients[i] * dx_power / factorial;
            dx_power *= dx;
            factorial *= (i + 1) as f64;
        }
        
        result
    }
    
    /// Compute derivative of Taylor series
    pub fn derivative(&self) -> TaylorSeries {
        if self.order == 0 {
            return TaylorSeries {
                coefficients: Vector::from_slice(&[0.0]),
                center: self.center,
                order: 0,
            };
        }
        
        let mut new_coeffs = Vector::new(self.order);
        for i in 0..self.order {
            new_coeffs[i] = self.coefficients[i + 1] * (i + 1) as f64;
        }
        
        TaylorSeries {
            coefficients: new_coeffs,
            center: self.center,
            order: self.order - 1,
        }
    }
}

/// Compute Taylor series expansion of a function
pub fn taylor_series<F>(f: F, center: f64, order: usize) -> TaylorSeries
where F: Fn(Dual2) -> Dual2
{
    let mut coefficients = Vector::new(order + 1);
    
    // Use nested dual numbers for higher-order derivatives
    let x = Dual2::variable(center);
    let result = f(x);
    
    coefficients[0] = result.val;
    if order >= 1 {
        coefficients[1] = result.d1;
    }
    if order >= 2 {
        coefficients[2] = result.d2;
    }
    
    // For higher orders, we'd need higher-order dual numbers
    // This is a simplified implementation
    
    TaylorSeries {
        coefficients,
        center,
        order,
    }
}

/// Compute directional second derivative
pub fn directional_hessian<F>(
    f: F, 
    x: &Vector<f64>, 
    v1: &Vector<f64>, 
    v2: &Vector<f64>
) -> f64
where F: Fn(&Vector<Dual2>) -> Dual2
{
    let n = x.len();
    let mut x_dual = Vector::new(n);
    
    for i in 0..n {
        x_dual[i] = Dual2::new(x[i], v1[i], v2[i]);
    }
    
    f(&x_dual).d2
}
