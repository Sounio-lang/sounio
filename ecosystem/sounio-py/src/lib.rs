mod knowledge;
mod protocol;
mod process;

use pyo3::prelude::*;

/// Python module `sounio._sounio_native`.
///
/// Exposes the Rust-side Knowledge type and the executor functions.
#[pymodule]
fn _sounio_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<knowledge::Knowledge>()?;
    m.add_function(wrap_pyfunction!(knowledge::knowledge_add, m)?)?;
    m.add_function(wrap_pyfunction!(knowledge::knowledge_mul, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
