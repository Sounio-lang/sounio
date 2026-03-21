/// PyO3 binding for Sounio's Knowledge<T> epistemic type.
///
/// Models a measured quantity with GUM-style uncertainty:
///   value   — central estimate
///   epsilon — standard uncertainty (k=1, one sigma)
///   prov    — provenance string (sensor / model / source)
///
/// Canonical display format (matches souc output):
///   Knowledge { value: 42.000 epsilon: 0.100 prov: "name" }
use pyo3::prelude::*;

#[pyclass(name = "Knowledge")]
#[derive(Clone, Debug)]
pub struct Knowledge {
    #[pyo3(get, set)]
    pub value: f64,
    #[pyo3(get, set)]
    pub epsilon: f64,
    #[pyo3(get, set)]
    pub prov: String,
}

#[pymethods]
impl Knowledge {
    #[new]
    #[pyo3(signature = (value, epsilon=0.0, prov=""))]
    pub fn new(value: f64, epsilon: f64, prov: &str) -> Self {
        Knowledge {
            value,
            epsilon,
            prov: prov.to_string(),
        }
    }

    /// Canonical format matching souc output.
    pub fn __repr__(&self) -> String {
        format!(
            "Knowledge {{ value: {:.3} epsilon: {:.3} prov: \"{}\" }}",
            self.value, self.epsilon, self.prov
        )
    }

    pub fn __str__(&self) -> String {
        format!(
            "Knowledge({:.3} ± {:.3}, prov='{}')",
            self.value, self.epsilon, self.prov
        )
    }

    /// GUM addition: value sums, uncertainties add in quadrature.
    pub fn __add__(&self, other: &Knowledge) -> Knowledge {
        Knowledge {
            value: self.value + other.value,
            epsilon: (self.epsilon.powi(2) + other.epsilon.powi(2)).sqrt(),
            prov: format!("({})+({})", self.prov, other.prov),
        }
    }

    /// GUM subtraction: value differences, uncertainties add in quadrature.
    pub fn __sub__(&self, other: &Knowledge) -> Knowledge {
        Knowledge {
            value: self.value - other.value,
            epsilon: (self.epsilon.powi(2) + other.epsilon.powi(2)).sqrt(),
            prov: format!("({})-({})", self.prov, other.prov),
        }
    }

    /// GUM multiplication: relative uncertainties add in quadrature.
    pub fn __mul__(&self, other: &Knowledge) -> Knowledge {
        let result_value = self.value * other.value;
        let rel_self = if self.value != 0.0 { self.epsilon / self.value.abs() } else { 0.0 };
        let rel_other = if other.value != 0.0 { other.epsilon / other.value.abs() } else { 0.0 };
        let rel_combined = (rel_self.powi(2) + rel_other.powi(2)).sqrt();
        Knowledge {
            value: result_value,
            epsilon: result_value.abs() * rel_combined,
            prov: format!("({})*({})", self.prov, other.prov),
        }
    }

    /// GUM division: relative uncertainties add in quadrature.
    pub fn __truediv__(&self, other: &Knowledge) -> PyResult<Knowledge> {
        if other.value == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Knowledge division by zero",
            ));
        }
        let result_value = self.value / other.value;
        let rel_self = if self.value != 0.0 { self.epsilon / self.value.abs() } else { 0.0 };
        let rel_other = if other.value != 0.0 { other.epsilon / other.value.abs() } else { 0.0 };
        let rel_combined = (rel_self.powi(2) + rel_other.powi(2)).sqrt();
        Ok(Knowledge {
            value: result_value,
            epsilon: result_value.abs() * rel_combined,
            prov: format!("({})/({})", self.prov, other.prov),
        })
    }

    /// Scale by a plain float (no uncertainty contribution from scalar).
    pub fn scale(&self, factor: f64) -> Knowledge {
        Knowledge {
            value: self.value * factor,
            epsilon: self.epsilon * factor.abs(),
            prov: self.prov.clone(),
        }
    }

    /// Relative uncertainty (epsilon / |value|), or 0 if value is zero.
    #[getter]
    pub fn relative_uncertainty(&self) -> f64 {
        if self.value == 0.0 {
            0.0
        } else {
            self.epsilon / self.value.abs()
        }
    }

    /// Confidence (1 - relative_uncertainty), clamped to [0, 1].
    #[getter]
    pub fn confidence(&self) -> f64 {
        (1.0 - self.relative_uncertainty()).max(0.0).min(1.0)
    }

    /// True if epsilon/|value| < threshold (default: 5%).
    #[pyo3(signature = (threshold=0.05))]
    pub fn is_reliable(&self, threshold: f64) -> bool {
        self.relative_uncertainty() < threshold
    }

    pub fn __eq__(&self, other: &Knowledge) -> bool {
        (self.value - other.value).abs() < f64::EPSILON
            && (self.epsilon - other.epsilon).abs() < f64::EPSILON
            && self.prov == other.prov
    }

    pub fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();
        self.value.to_bits().hash(&mut h);
        self.epsilon.to_bits().hash(&mut h);
        self.prov.hash(&mut h);
        h.finish()
    }
}

/// Add two Knowledge values (GUM addition).
#[pyfunction]
pub fn knowledge_add(a: &Knowledge, b: &Knowledge) -> Knowledge {
    a.__add__(b)
}

/// Multiply two Knowledge values (GUM multiplication).
#[pyfunction]
pub fn knowledge_mul(a: &Knowledge, b: &Knowledge) -> Knowledge {
    a.__mul__(b)
}
