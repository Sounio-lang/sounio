//! Machine Learning Integration Module
//!
//! This module provides integration with specialized AI models for scientific computing:
//! - BioMedLM/Galactica: Trained specifically on scientific literature
//! - Polymath: Fine-tuned for mathematical and scientific reasoning  
//! - SantaCoder/InCoder: Strong for scientific code generation
//!
//! These models enhance Sounio's scientific computing capabilities by providing
//! AI-powered code generation, scientific reasoning, and domain expertise.

pub mod scientific_models;

pub use scientific_models::*;
