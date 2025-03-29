from setuptools import setup, Extension

module = Extension(
    "myext",                # name of the module to import in Python
    sources=["myext.c"],    # C source file(s)
)

setup(
    name="myext",
    version="1.0",
    description="Example C extension that calls a callback endlessly.",
    ext_modules=[module],
)
