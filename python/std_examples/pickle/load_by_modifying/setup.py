from setuptools import find_packages, setup

setup(
    name="dummy",
    version="0.0.0",
    description="experimental",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
)

