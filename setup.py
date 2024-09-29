from setuptools import setup, find_packages

setup(
    name="intreg",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=2.1.1",
        "scipy>=1.14.1",
    ],
)
