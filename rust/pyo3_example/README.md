maturin build
pip install target/wheels/my_pyo3_project-0.1.0-...-any.whl
python -c "import my_pyo3_project; print(my_pyo3_project.hello_from_rust('PyO3'))"
