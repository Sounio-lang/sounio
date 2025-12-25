// Bayesian Inference Module
// Probabilistic programming with MCMC and variational inference
//
// Submodules:
// - prior: Prior distributions (Normal, LogNormal, Beta, Gamma, etc.)
// - mcmc: Metropolis-Hastings sampling with adaptive proposals
// - vi: Mean-field variational inference with ELBO optimization
// - diagnostics: Gelman-Rubin R-hat, ESS, autocorrelation

pub mod prior;
pub mod mcmc;
pub mod vi;
pub mod diagnostics;
