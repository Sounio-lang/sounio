//! Biological and Quaternionic Computing Module
//!
//! This module implements the mathematical framework from "The Quaternionic Syntax
//! of Existence" (Agourakis & Agourakis), providing:
//!
//! - **DNA Operators**: Shift, reverse, complement, edit with dihedral group structure
//! - **Quaternion Operations**: Unit quaternions for SU(2) representations
//! - **GF(4) Arithmetic**: Galois field for quaternary DNA encoding
//! - **Transmission Dynamics**: Norm-preserving information channels
//!
//! # Theoretical Background
//!
//! DNA transformations form a dihedral group D_n with composition rules:
//! ```text
//! σ ∘ ρ = ρ^(-1) ∘ σ    (shift-reverse anticommute)
//! κ ∘ κ = id             (complement is involution)
//! ```
//!
//! Transmission channels are parameterized as unit quaternions q ∈ SU(2):
//! ```text
//! q = g + ti + pj + ek,  |q| = 1
//! ```
//!
//! # Example
//!
//! ```ignore
//! use sounio::bio::{DNAString, Base, UnitQuat};
//!
//! let dna = DNAString::new(&[Base::A, Base::C, Base::G, Base::T]);
//! let shifted = dna.shift();
//! let rev_comp = dna.reverse().complement();
//!
//! let q = UnitQuat::new(0.5, 0.5, 0.5, 0.5);
//! assert!((q.norm() - 1.0).abs() < 1e-10);
//! ```

pub mod dna;
pub mod gf4;
pub mod quaternion;
pub mod transmission;

pub use dna::{Base, DNAOperator, DNAString};
pub use gf4::{GF4, GF4Element};
pub use quaternion::{QuatOps, UnitQuat};
pub use transmission::{Channel, Transmission};
