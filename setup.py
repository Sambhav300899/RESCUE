from setuptools import find_namespace_packages, setup

setup(
    name="rescue",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    python_requires=">=3.10",
)
