## PITFALLS!
- if you use pyenv, please specify python executable path like `cmake -DPYTHON_EXECUTABLE=path/to/python ..`
- install [pybind11](https://github.com/pybind/pybind11) under the root directory
- module name specified by `PYBIND11_MODULE(module_name, m)` must match the name of `module_name.so` of cmake side.

## PITFALLS in installing python module
python's output is with "\n" by default. Thus we must specify `end=''` so that it's properly installed.
"""cmake
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(site.getsitepackages()[0], end='')"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_INSTALL_DIR
    )
"""

