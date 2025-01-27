from setuptools import setup, find_packages

setup(
    name="intreg",
    version="0.1.8",
    author="Dylan Adlard, Philip W Fowler",
    author_email="philip.fowler@ndm.ox.ac.uk",
    description="A package for fitting an interval and mixed effect interval regression model to censored and/or uncensored point and/or interval data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fowler-lab/intreg",
    keywords="interval, regression",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
    "numpy>=1.21.1,<=2.2.2",
    "scipy>=1.8.0,<=1.15.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
)
