/// PyO3 binding for Sounio's Knowledge<T> epistemic type.
///
/// Models a measured quantity with:
///   value       — central estimate
///   uncertainty — standard uncertainty (k=1, σ) — alias: epsilon
///   confidence  — epistemic trust in the claim [0,1]
///   unit        — physical unit string
///   prov        — provenance string (sensor / model / source)
///
/// GUM (JCGM 100:2008) first-order uncertainty propagation for all arithmetic.
use pyo3::prelude::*;
use std::f64;

#[pyclass(name = "Knowledge")]
#[derive(Clone, Debug)]
pub struct Knowledge {
    #[pyo3(get, set)]
    pub value: f64,
    #[pyo3(get, set)]
    pub uncertainty: f64,   // standard uncertainty (σ), GUM §4.2
    #[pyo3(get, set)]
    pub confidence: f64,    // epistemic confidence [0,1] — orthogonal to uncertainty
    #[pyo3(get, set)]
    pub unit: String,
    #[pyo3(get, set)]
    pub prov: String,
}

#[pymethods]
impl Knowledge {
    #[new]
    #[pyo3(signature = (value, uncertainty=0.0, confidence=1.0, unit="", prov=""))]
    pub fn new(value: f64, uncertainty: f64, confidence: f64, unit: &str, prov: &str) -> PyResult<Self> {
        if uncertainty < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "uncertainty must be >= 0",
            ));
        }
        if !(0.0..=1.0).contains(&confidence) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "confidence must be in [0, 1]",
            ));
        }
        Ok(Knowledge {
            value,
            uncertainty,
            confidence,
            unit: unit.to_string(),
            prov: prov.to_string(),
        })
    }

    /// Backward-compat alias: epsilon = uncertainty
    #[getter]
    pub fn epsilon(&self) -> f64 { self.uncertainty }
    #[setter]
    pub fn set_epsilon(&mut self, v: f64) { self.uncertainty = v; }

    pub fn __repr__(&self) -> String {
        let unit_str = if self.unit.is_empty() { String::new() } else { format!(" {}", self.unit) };
        format!(
            "Knowledge({} ± {}{}, confidence={})",
            self.value, self.uncertainty, unit_str, self.confidence
        )
    }

    pub fn __str__(&self) -> String { self.__repr__() }

    // ------------------------------------------------------------------
    // Arithmetic — GUM first-order (JCGM 100:2008 §13)
    // ------------------------------------------------------------------

    pub fn __add__(&self, other: &Knowledge) -> Knowledge {
        Knowledge {
            value: self.value + other.value,
            uncertainty: (self.uncertainty.powi(2) + other.uncertainty.powi(2)).sqrt(),
            confidence: self.confidence.min(other.confidence),
            unit: self.unit.clone(),
            prov: format!("({})+({})", self.prov, other.prov),
        }
    }

    pub fn __radd__(&self, other: f64) -> Knowledge {
        Knowledge {
            value: self.value + other,
            uncertainty: self.uncertainty,
            confidence: self.confidence,
            unit: self.unit.clone(),
            prov: self.prov.clone(),
        }
    }

    pub fn __sub__(&self, other: &Knowledge) -> Knowledge {
        Knowledge {
            value: self.value - other.value,
            uncertainty: (self.uncertainty.powi(2) + other.uncertainty.powi(2)).sqrt(),
            confidence: self.confidence.min(other.confidence),
            unit: self.unit.clone(),
            prov: format!("({})-({})", self.prov, other.prov),
        }
    }

    pub fn __mul__(&self, other: &Knowledge) -> Knowledge {
        let result_value = self.value * other.value;
        let rel_self  = if self.value  != 0.0 { self.uncertainty  / self.value.abs()  } else { 0.0 };
        let rel_other = if other.value != 0.0 { other.uncertainty / other.value.abs() } else { 0.0 };
        let rel = (rel_self.powi(2) + rel_other.powi(2)).sqrt();
        Knowledge {
            value: result_value,
            uncertainty: result_value.abs() * rel,
            confidence: self.confidence.min(other.confidence),
            unit: String::new(),
            prov: format!("({})*({})", self.prov, other.prov),
        }
    }

    pub fn __rmul__(&self, other: f64) -> Knowledge {
        Knowledge {
            value: self.value * other,
            uncertainty: self.uncertainty * other.abs(),
            confidence: self.confidence,
            unit: self.unit.clone(),
            prov: self.prov.clone(),
        }
    }

    pub fn __truediv__(&self, other: &Knowledge) -> PyResult<Knowledge> {
        if other.value == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Knowledge division by zero",
            ));
        }
        let result_value = self.value / other.value;
        let rel_self  = if self.value  != 0.0 { self.uncertainty  / self.value.abs()  } else { 0.0 };
        let rel_other = if other.value != 0.0 { other.uncertainty / other.value.abs() } else { 0.0 };
        let rel = (rel_self.powi(2) + rel_other.powi(2)).sqrt();
        Ok(Knowledge {
            value: result_value,
            uncertainty: result_value.abs() * rel,
            confidence: self.confidence.min(other.confidence),
            unit: String::new(),
            prov: format!("({})/({})", self.prov, other.prov),
        })
    }

    pub fn __neg__(&self) -> Knowledge {
        Knowledge {
            value: -self.value,
            uncertainty: self.uncertainty,
            confidence: self.confidence,
            unit: self.unit.clone(),
            prov: format!("-({})", self.prov),
        }
    }

    pub fn __abs__(&self) -> Knowledge {
        Knowledge {
            value: self.value.abs(),
            uncertainty: self.uncertainty,
            confidence: self.confidence,
            unit: self.unit.clone(),
            prov: format!("|{}|", self.prov),
        }
    }

    // ------------------------------------------------------------------
    // Comparison (nominal values)
    // ------------------------------------------------------------------
    pub fn __lt__(&self, other: &Knowledge) -> bool { self.value < other.value }
    pub fn __le__(&self, other: &Knowledge) -> bool { self.value <= other.value }
    pub fn __gt__(&self, other: &Knowledge) -> bool { self.value > other.value }
    pub fn __ge__(&self, other: &Knowledge) -> bool { self.value >= other.value }

    pub fn __eq__(&self, other: &Knowledge) -> bool {
        (self.value - other.value).abs() < f64::EPSILON
            && (self.uncertainty - other.uncertainty).abs() < f64::EPSILON
    }

    pub fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();
        self.value.to_bits().hash(&mut h);
        self.uncertainty.to_bits().hash(&mut h);
        self.prov.hash(&mut h);
        h.finish()
    }

    // ------------------------------------------------------------------
    // Epistemic helpers
    // ------------------------------------------------------------------

    /// Coefficient of variation in percent.
    #[getter]
    pub fn cv_percent(&self) -> f64 {
        if self.value == 0.0 { f64::INFINITY } else { (self.uncertainty / self.value.abs()) * 100.0 }
    }

    /// Relative uncertainty (u / |value|).
    #[getter]
    pub fn relative_uncertainty(&self) -> f64 {
        if self.value == 0.0 { 0.0 } else { self.uncertainty / self.value.abs() }
    }

    /// Expanded uncertainty at coverage factor k (default k=2, ~95%).
    #[pyo3(signature = (k=2.0))]
    pub fn expanded_uncertainty(&self, k: f64) -> f64 { k * self.uncertainty }

    /// P(value > x) using normal approximation.
    pub fn prob_gt(&self, x: f64) -> f64 {
        if self.uncertainty == 0.0 {
            return if self.value > x { 1.0 } else { 0.0 };
        }
        norm_cdf((self.value - x) / self.uncertainty)
    }

    /// True if expanded uncertainty (k=2) <= threshold.
    pub fn is_within_threshold(&self, threshold: f64) -> bool {
        2.0 * self.uncertainty <= threshold
    }

    /// True if uncertainty/|value| < threshold (default 5%).
    #[pyo3(signature = (threshold=0.05))]
    pub fn is_reliable(&self, threshold: f64) -> bool {
        self.relative_uncertainty() < threshold
    }

    /// Convert to dict for JSON serialization.
    pub fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("value", self.value)?;
        dict.set_item("uncertainty", self.uncertainty)?;
        dict.set_item("confidence", self.confidence)?;
        dict.set_item("unit", &self.unit)?;
        dict.set_item("cv_percent", self.cv_percent())?;
        dict.set_item("prov", &self.prov)?;
        Ok(dict.into())
    }

    /// Scale by a plain float.
    pub fn scale(&self, factor: f64) -> Knowledge {
        Knowledge {
            value: self.value * factor,
            uncertainty: self.uncertainty * factor.abs(),
            confidence: self.confidence,
            unit: self.unit.clone(),
            prov: self.prov.clone(),
        }
    }
}

/// Approximation of the standard normal CDF (Abramowitz & Stegun 26.2.17).
pub fn norm_cdf(z: f64) -> f64 {
    if z >= 8.0 { return 1.0; }
    if z <= -8.0 { return 0.0; }
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422820 * (-0.5 * z * z).exp();
    let poly = t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))));
    let p = 1.0 - d * poly;
    if z >= 0.0 { p } else { 1.0 - p }
}

/// Add two Knowledge values (GUM addition).
#[pyfunction]
pub fn knowledge_add(a: &Knowledge, b: &Knowledge) -> Knowledge { a.__add__(b) }

/// Multiply two Knowledge values (GUM multiplication).
#[pyfunction]
pub fn knowledge_mul(a: &Knowledge, b: &Knowledge) -> Knowledge { a.__mul__(b) }

/// Construct a Knowledge measurement (convenience).
#[pyfunction]
#[pyo3(signature = (value, uncertainty=0.0, confidence=1.0, unit="", source=""))]
pub fn measure(
    value: f64, uncertainty: f64, confidence: f64, unit: &str, source: &str,
) -> PyResult<Knowledge> {
    Knowledge::new(value, uncertainty, confidence, unit, source)
}

/// Raise ConfidenceGateError if k.confidence < min_confidence.
#[pyfunction]
pub fn confidence_gate(k: &Knowledge, min_confidence: f64) -> PyResult<()> {
    if k.confidence < min_confidence {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "confidence_gate failed: {} < {} (prov='{}')",
            k.confidence, min_confidence, k.prov,
        )))
    } else {
        Ok(())
    }
}
