/// GUM uncertainty propagation engine and epistemic utilities.
///
/// Implements:
///   - GUMPropagation: symbolic / numerical partial-derivative propagation
///   - EpistemicResult: named collection of Knowledge<T> outputs
///   - SensitivityCoefficient: ∂y/∂xᵢ and contribution ranking
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::knowledge::Knowledge;

const GUM_H: f64 = 1e-6; // step for numerical differentiation

// ============================================================================
// SENSITIVITY COEFFICIENT
// ============================================================================

#[pyclass]
#[derive(Clone, Debug)]
pub struct SensitivityCoefficient {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub coefficient: f64,       // ∂y/∂xᵢ
    #[pyo3(get)]
    pub contribution: f64,      // (cᵢ · uᵢ)² share in [0,1]
    #[pyo3(get)]
    pub abs_contribution: f64,  // cᵢ · uᵢ absolute
}

#[pymethods]
impl SensitivityCoefficient {
    pub fn __repr__(&self) -> String {
        format!(
            "SensitivityCoefficient(name='{}', c={}, share={}%)",
            self.name, self.coefficient, self.contribution * 100.0
        )
    }
}

// ============================================================================
// EPISTEMIC RESULT
// ============================================================================

/// A named collection of Knowledge<T> outputs — the result of a Sounio model run.
#[pyclass]
#[derive(Clone, Debug)]
pub struct EpistemicResult {
    #[pyo3(get)]
    pub label: String,
    outputs: Vec<(String, Knowledge)>,
}

#[pymethods]
impl EpistemicResult {
    #[new]
    #[pyo3(signature = (label="result"))]
    pub fn new(label: &str) -> Self {
        EpistemicResult { label: label.to_string(), outputs: Vec::new() }
    }

    pub fn add(&mut self, name: &str, k: Knowledge) {
        self.outputs.push((name.to_string(), k));
    }

    pub fn get(&self, name: &str) -> Option<Knowledge> {
        self.outputs.iter().find(|(n, _)| n == name).map(|(_, k)| k.clone())
    }

    pub fn names(&self) -> Vec<String> {
        self.outputs.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Overall result confidence: min of all outputs.
    #[getter]
    pub fn confidence(&self) -> f64 {
        self.outputs.iter().map(|(_, k)| k.confidence).fold(1.0_f64, f64::min)
    }

    pub fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (name, k) in &self.outputs {
            dict.set_item(name, k.to_dict(py)?)?;
        }
        Ok(dict.into())
    }

    pub fn __repr__(&self) -> String {
        let names: Vec<&str> = self.outputs.iter().map(|(n, _)| n.as_str()).collect();
        format!("EpistemicResult('{}', outputs=[{}], confidence={})",
            self.label, names.join(", "), self.confidence())
    }
}

// ============================================================================
// GUM PROPAGATION ENGINE
// ============================================================================

/// First-order GUM uncertainty propagation via numerical differentiation.
///
/// Usage:
///   gum = GUMPropagation()
///   gum.add_input("dose",   measure(500.0, uncertainty=25.0, unit="mg"))
///   gum.add_input("volume", measure(10.0,  uncertainty=0.2,  unit="mL"))
///
///   def concentration(dose_val, vol_val):
///       return dose_val / vol_val
///
///   result = gum.propagate(concentration)
///   print(result)        # Knowledge(50.0 ± 1.25 mg/mL, confidence=1.00)
///   gum.uncertainty_budget()
#[pyclass]
pub struct GUMPropagation {
    inputs: Vec<(String, Knowledge)>,
    #[pyo3(get, set)]
    pub h: f64,
}

#[pymethods]
impl GUMPropagation {
    #[new]
    #[pyo3(signature = (h=None))]
    pub fn new(h: Option<f64>) -> Self {
        GUMPropagation {
            inputs: Vec::new(),
            h: h.unwrap_or(GUM_H),
        }
    }

    pub fn add_input(&mut self, name: &str, k: Knowledge) {
        self.inputs.push((name.to_string(), k));
    }

    pub fn input_names(&self) -> Vec<String> {
        self.inputs.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Propagate uncertainty through `func(*nominal_values) -> f64`.
    /// Returns a Knowledge with GUM combined standard uncertainty.
    pub fn propagate(&self, py: Python<'_>, func: PyObject) -> PyResult<Knowledge> {
        let n = self.inputs.len();
        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("No inputs registered"));
        }

        let nominals: Vec<f64> = self.inputs.iter().map(|(_, k)| k.value).collect();

        // Evaluate at nominal point
        let y0 = call_func_f64(py, &func, &nominals)?;

        // Compute partial derivatives and combined uncertainty
        let mut combined_var = 0.0_f64;
        let mut min_confidence = 1.0_f64;

        let coefficients = self.compute_sensitivity_internal(py, &func, &nominals)?;
        for (i, (_, k)) in self.inputs.iter().enumerate() {
            let ci = coefficients[i];
            combined_var += (ci * k.uncertainty).powi(2);
            min_confidence = min_confidence.min(k.confidence);
        }

        let units: Vec<&str> = self.inputs.iter().map(|(_, k)| k.unit.as_str()).collect();
        let prov_parts: Vec<String> = self.inputs.iter().map(|(n, k)| {
            format!("{}({})", n, k.prov)
        }).collect();

        Knowledge::new(
            y0,
            combined_var.sqrt(),
            min_confidence,
            "",   // output unit not inferred
            &prov_parts.join("; "),
        )
    }

    /// Returns list of SensitivityCoefficient ordered by contribution (desc).
    pub fn sensitivity_coefficients(
        &self, py: Python<'_>, func: PyObject,
    ) -> PyResult<Vec<SensitivityCoefficient>> {
        let nominals: Vec<f64> = self.inputs.iter().map(|(_, k)| k.value).collect();
        let coefficients = self.compute_sensitivity_internal(py, &func, &nominals)?;

        // Compute absolute contributions cᵢ · uᵢ
        let abs_contribs: Vec<f64> = coefficients.iter().zip(self.inputs.iter())
            .map(|(ci, (_, k))| (ci * k.uncertainty).abs())
            .collect();
        let total_var: f64 = abs_contribs.iter().map(|a| a.powi(2)).sum();

        let mut result: Vec<SensitivityCoefficient> = self.inputs.iter().enumerate()
            .map(|(i, (name, _))| {
                let share = if total_var > 0.0 { abs_contribs[i].powi(2) / total_var } else { 0.0 };
                SensitivityCoefficient {
                    name: name.clone(),
                    coefficient: coefficients[i],
                    contribution: share,
                    abs_contribution: abs_contribs[i],
                }
            })
            .collect();

        result.sort_by(|a, b| b.contribution.partial_cmp(&a.contribution).unwrap());
        Ok(result)
    }

    /// Print uncertainty budget table to stdout.
    pub fn uncertainty_budget(&self, py: Python<'_>, func: PyObject) -> PyResult<()> {
        let coeffs = self.sensitivity_coefficients(py, func)?;
        println!("{:<20} {:>12} {:>12} {:>10}", "Input", "Sensitivity", "|c·u|", "Share%");
        println!("{}", "-".repeat(58));
        for c in &coeffs {
            println!("{} {} {} {}",
                c.name, c.coefficient, c.abs_contribution, c.contribution * 100.0);
        }
        Ok(())
    }
}

impl GUMPropagation {
    fn compute_sensitivity_internal(
        &self,
        py: Python<'_>,
        func: &PyObject,
        nominals: &[f64],
    ) -> PyResult<Vec<f64>> {
        let n = nominals.len();
        let y0 = call_func_f64(py, func, nominals)?;
        let mut coefficients = vec![0.0_f64; n];
        for i in 0..n {
            let mut perturbed = nominals.to_vec();
            perturbed[i] += self.h;
            let yh = call_func_f64(py, func, &perturbed)?;
            coefficients[i] = (yh - y0) / self.h;
        }
        Ok(coefficients)
    }
}

fn call_func_f64(py: Python<'_>, func: &PyObject, args: &[f64]) -> PyResult<f64> {
    let py_args = PyList::new_bound(py, args);
    let result = func.call1(py, (py_args,))?;
    result.extract::<f64>(py)
}
