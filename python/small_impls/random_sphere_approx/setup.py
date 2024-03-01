from setuptools import setup

setup_requires = []

install_requires = [
    "numpy",
    "trimesh",
    "tqdm",
    "scipy",
    "pysdfgen",
]

setup(
    name="mesh2spheres",
    version="0.0.0",
    description="compute approximating random spheres from a mesh",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    package_data={"mesh2spheres": ["py.typed"]},
)
