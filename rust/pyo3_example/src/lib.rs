use pyo3::prelude::*;

#[pyfunction]
fn hello_from_rust(name: &str) -> PyResult<String> {
    Ok(format!("Hello, {}! This is Rust talking.", name))
}

#[pyclass]
struct MyRustClass {
    name: String,
}

#[pymethods]
impl MyRustClass {
    #[new]
    fn new(name: String) -> Self {
        MyRustClass { name }
    }

    fn greet(&self) -> PyResult<String> {
        Ok(format!(
            "Hello, {}! MyRustClass is greeting you from Rust.",
            self.name
        ))
    }
}

#[pymodule]
fn my_pyo3_project(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_rust, m)?)?;
    m.add_class::<MyRustClass>()?;
    Ok(())
}
