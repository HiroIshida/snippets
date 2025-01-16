```
maturin build
pip install target/wheels/my_pyo3_project-0.1.0-cp38-cp38-manylinux_2_28_x86_64.whl  --force-reinstall
python -c "import my_pyo3_project; print(my_pyo3_project.hello_from_rust('PyO3'))"
python -c "import my_pyo3_project; a = my_pyo3_project.MyRustClass('unko'); print(a.greet())"
```
