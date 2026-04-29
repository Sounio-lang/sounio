mod epistemic;
mod knowledge;
mod protocol;
mod process;

use pyo3::prelude::*;

/// Python module `sounio._sounio_native`.
///
/// High-performance Rust core for Sounio's Python bindings.
/// Exposes Knowledge<T>, GUM propagation, and epistemic result types.
#[pymodule]
fn _sounio_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Knowledge type
    m.add_class::<knowledge::Knowledge>()?;

    // Knowledge arithmetic functions (convenience)
    m.add_function(wrap_pyfunction!(knowledge::knowledge_add, m)?)?;
    m.add_function(wrap_pyfunction!(knowledge::knowledge_mul, m)?)?;

    // Epistemic constructors
    m.add_function(wrap_pyfunction!(knowledge::measure, m)?)?;
    m.add_function(wrap_pyfunction!(knowledge::confidence_gate, m)?)?;

    // GUM propagation engine
    m.add_class::<epistemic::GUMPropagation>()?;
    m.add_class::<epistemic::EpistemicResult>()?;
    m.add_class::<epistemic::SensitivityCoefficient>()?;

    m.add("__version__", "0.2.0")?;
    Ok(())
}
