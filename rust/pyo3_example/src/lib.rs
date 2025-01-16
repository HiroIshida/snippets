use pyo3::prelude::*;

#[pyfunction]
fn hello_from_rust(name: &str) -> PyResult<String> {
    Ok(format!("Hello, {}! This is Rust talking.", name))
}

#[pymodule]
fn my_pyo3_project(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_rust, m)?)?;
    Ok(())
}
