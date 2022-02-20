use pyo3::prelude::*;

#[pyfunction]
fn levenshtein_search(needle: &str, haystack: &str) -> Vec<(usize, usize, u32)> {
    triple_accel::levenshtein_search(needle.as_bytes(), haystack.as_bytes())
        .map(|m| (m.start, m.end, m.k))
        .collect()
}

#[pyfunction]
fn levenshtein(a: &str, b: &str) -> u32 {
    triple_accel::levenshtein_exp(a.as_bytes(), b.as_bytes())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(levenshtein_search, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    Ok(())
}
