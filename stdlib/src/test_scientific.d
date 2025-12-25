//! Comprehensive test suite for scientific computing libraries

use test::{assert_eq, assert_approx_eq, test_case}
use linalg::{Matrix, Vector}
use linalg::blas::{daxpy, ddot, dgemm}
use linalg::lapack::{lu, solve, cholesky, svd, eig}
use numerics::ode::{odeint, ODESystem}
use numerics::optimize::{GradientDescent, BFGS}
use numerics::integrate::{quad, simpson}
use numerics::fft::{fft, ifft, Complex}
use autodiff::forward::{Dual, gradient as grad_forward}
use autodiff::reverse::{Var, gradient as grad_reverse}
use prob::distributions::{Normal, MultivariateNormal, Gamma, Beta}
use prob::mcmc::{MetropolisHastings, HMC}
use pkpd::compartment::{PKParameters, simulate_pk, DoseEvent}
use pkpd::nca::{nca_analysis}

/// Test linear algebra operations
#[test_case]
fn test_matrix_operations() {
    // Test matrix creation and indexing
    let mut a = Matrix::zeros(3, 3);
    a[(0, 0)] = 1.0;
    a[(1, 1)] = 2.0;
    a[(2, 2)] = 3.0;
    
    assert_eq!(a[(0, 0)], 1.0);
    assert_eq!(a[(1, 1)], 2.0);
    assert_eq!(a[(2, 2)], 3.0);
    
    // Test matrix multiplication
    let b = Matrix::eye(3);
    let c = &a * &b;
    
    assert_eq!(c[(0, 0)], 1.0);
    assert_eq!(c[(1, 1)], 2.0);
    assert_eq!(c[(2, 2)], 3.0);
}

#[test_case]
fn test_blas_operations() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[4.0, 5.0, 6.0]);
    
    // Test dot product
    let dot_result = ddot(&x, &y);
    assert_approx_eq!(dot_result, 32.0, 1e-10);
    
    // Test AXPY operation
    let mut z = y.clone();
    daxpy(2.0, &x, &!z);
    
    assert_approx_eq!(z[0], 6.0, 1e-10);  // 4 + 2*1
    assert_approx_eq!(z[1], 9.0, 1e-10);  // 5 + 2*2
    assert_approx_eq!(z[2], 12.0, 1e-10); // 6 + 2*3
}

#[test_case]
fn test_lapack_decompositions() {
    // Test LU decomposition
    let a = Matrix::from_nested(&[
        [2.0, 1.0],
        [1.0, 2.0],
    ]);
    
    let lu_result = lu(&a).unwrap();
    assert!(lu_result.factors.nrows() == 2);
    assert!(lu_result.pivots.len() == 2);
    
    // Test linear system solving
    let b = Vector::from_slice(&[3.0, 3.0]);
    let x = solve(&a, &b).unwrap();
    
    // Solution should be [1, 1]
    assert_approx_eq!(x[0], 1.0, 1e-10);
    assert_approx_eq!(x[1], 1.0, 1e-10);
    
    // Test Cholesky decomposition (positive definite matrix)
    let chol_result = cholesky(&a).unwrap();
    assert!(chol_result.factor.nrows() == 2);
}

#[test_case]
fn test_ode_solver() {
    // Simple exponential decay: dy/dt = -k*y
    struct ExponentialDecay {
        k: f64,
    }
    
    impl ODESystem for ExponentialDecay {
        fn eval(&self, t: f64, y: &Vector<f64>, dydt: &!Vector<f64>) {
            dydt[0] = -self.k * y[0];
        }
        
        fn dim(&self) -> usize { 1 }
    }
    
    let system = ExponentialDecay { k: 1.0 };
    let y0 = Vector::from_slice(&[1.0]);
    
    let solution = odeint(system, &y0, (0.0, 1.0), "RK45");
    
    assert!(solution.success);
    
    // At t=1, y should be approximately e^(-1) ≈ 0.368
    let final_y = solution.y[(solution.y.nrows() - 1, 0)];
    assert_approx_eq!(final_y, 0.368, 0.01);
}

#[test_case]
fn test_optimization() {
    // Minimize f(x) = (x-2)^2, minimum at x=2
    let objective = |x: &Vector<Var>| -> Var {
        let diff = x[0] - Var::new(2.0);
        diff * diff
    };
    
    let optimizer = GradientDescent::new(0.1);
    let x0 = Vector::from_slice(&[0.0]);
    
    let result = optimizer.minimize(objective, &x0);
    
    assert!(result.success);
    assert_approx_eq!(result.x[0], 2.0, 0.1);
}

#[test_case]
fn test_numerical_integration() {
    // Integrate x^2 from 0 to 1, should be 1/3
    let f = |x: f64| x * x;
    
    let result = quad(f, 0.0, 1.0, Default::default());
    
    assert!(result.success);
    assert_approx_eq!(result.value, 1.0/3.0, 1e-6);
    
    // Test Simpson's rule
    let simpson_result = simpson(f, 0.0, 1.0, 1e-6);
    assert_approx_eq!(simpson_result.value, 1.0/3.0, 1e-6);
}

#[test_case]
fn test_fft() {
    // Test FFT of simple signal
    let input = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    
    let fft_result = fft(&input);
    assert!(fft_result.success);
    
    // Test inverse FFT
    let ifft_result = ifft(&fft_result.data);
    assert!(ifft_result.success);
    
    // Should recover original signal
    for i in 0..input.len() {
        assert_approx_eq!(ifft_result.data[i].re, input[i].re, 1e-10);
        assert_approx_eq!(ifft_result.data[i].im, input[i].im, 1e-10);
    }
}

#[test_case]
fn test_automatic_differentiation() {
    // Test forward mode AD
    let f_forward = |x: &Vector<Dual>| -> Dual {
        x[0] * x[0] + x[1] * x[1] // f(x,y) = x^2 + y^2
    };
    
    let x = Vector::from_slice(&[1.0, 2.0]);
    let grad = grad_forward(f_forward, &x);
    
    // Gradient should be [2*x, 2*y] = [2, 4]
    assert_approx_eq!(grad[0], 2.0, 1e-10);
    assert_approx_eq!(grad[1], 4.0, 1e-10);
    
    // Test reverse mode AD
    let f_reverse = |x: &Vector<Var>| -> Var {
        x[0] * x[0] + x[1] * x[1]
    };
    
    let grad_rev = grad_reverse(f_reverse, &x);
    
    assert_approx_eq!(grad_rev[0], 2.0, 1e-10);
    assert_approx_eq!(grad_rev[1], 4.0, 1e-10);
}

#[test_case]
fn test_probability_distributions() {
    let mut rng = rand::thread_rng();
    
    // Test normal distribution
    let normal = Normal::new(0.0, 1.0);
    
    // Test PDF at mean
    let pdf_at_mean = normal.pdf(0.0);
    let expected_pdf = 1.0 / (2.0 * std::math::PI).sqrt();
    assert_approx_eq!(pdf_at_mean, expected_pdf, 1e-10);
    
    // Test CDF at mean (should be 0.5)
    let cdf_at_mean = normal.cdf(0.0);
    assert_approx_eq!(cdf_at_mean, 0.5, 1e-10);
    
    // Test sampling (just check it doesn't crash)
    let sample = normal.sample(&!rng);
    assert!(sample.is_finite());
    
    // Test multivariate normal
    let mean = Vector::from_slice(&[0.0, 0.0]);
    let cov = Matrix::eye(2);
    let mvn = MultivariateNormal::new(mean, cov).unwrap();
    
    let mv_sample = mvn.sample(&!rng);
    assert!(mv_sample.len() == 2);
}

#[test_case]
fn test_mcmc_sampling() {
    // Test Metropolis-Hastings on simple 1D normal
    let log_prob = |x: &Vector<f64>| -> f64 {
        let normal = Normal::new(0.0, 1.0);
        normal.log_pdf(x[0])
    };
    
    let mut sampler = MetropolisHastings::new(1);
    let x0 = Vector::from_slice(&[0.0]);
    let mut rng = rand::thread_rng();
    
    let samples = sampler.sample(log_prob, &x0, 100, &!rng);
    
    assert!(samples.n_samples == 100);
    assert!(samples.acceptance_rate > 0.0);
    assert!(samples.acceptance_rate <= 1.0);
}

#[test_case]
fn test_pharmacokinetics() {
    // Test 1-compartment PK model
    let params = PKParameters::one_compartment(10.0, 50.0); // CL=10 L/h, V=50 L
    
    let dose = DoseEvent::iv_bolus(0.0, 100.0); // 100 mg at t=0
    let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    
    let result = simulate_pk(&params, &[dose], &times);
    
    // Check that concentrations decrease over time
    let conc = &result.concentrations;
    for i in 1..times.len() {
        assert!(conc[(i, 0)] < conc[(i-1, 0)]);
    }
    
    // Initial concentration should be dose/volume = 100/50 = 2 mg/L
    assert_approx_eq!(conc[(0, 0)], 2.0, 1e-10);
}

#[test_case]
fn test_nca_analysis() {
    // Create synthetic PK data
    let times = Vector::from_slice(&[0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]);
    let concentrations = Vector::from_slice(&[2.0, 1.5, 1.1, 0.6, 0.2, 0.07, 0.01]);
    
    let nca_result = nca_analysis(&times, &concentrations, 100.0, 0.0, 3);
    
    // Check that basic parameters are calculated
    assert!(nca_result.cmax > 0.0);
    assert!(nca_result.tmax >= 0.0);
    assert!(nca_result.auc_last > 0.0);
    assert!(nca_result.t_half > 0.0);
    assert!(nca_result.cl > 0.0);
}

/// Run all scientific computing tests
pub fn run_all_tests() {
    println!("Running scientific computing test suite...");
    
    test_matrix_operations();
    println!("✓ Matrix operations");
    
    test_blas_operations();
    println!("✓ BLAS operations");
    
    test_lapack_decompositions();
    println!("✓ LAPACK decompositions");
    
    test_ode_solver();
    println!("✓ ODE solver");
    
    test_optimization();
    println!("✓ Optimization");
    
    test_numerical_integration();
    println!("✓ Numerical integration");
    
    test_fft();
    println!("✓ FFT");
    
    test_automatic_differentiation();
    println!("✓ Automatic differentiation");
    
    test_probability_distributions();
    println!("✓ Probability distributions");
    
    test_mcmc_sampling();
    println!("✓ MCMC sampling");
    
    test_pharmacokinetics();
    println!("✓ Pharmacokinetics");
    
    test_nca_analysis();
    println!("✓ Non-compartmental analysis");
    
    println!("All scientific computing tests passed! ✅");
}
