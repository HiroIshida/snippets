from setuptools import find_packages, setup

setup_requires = []

install_requires = [
]

setup(
    name="blendpred",
    version="0.0.1",
    description="",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests", "docs")),
    package_data={"mohou": ["py.typed"]},
)
