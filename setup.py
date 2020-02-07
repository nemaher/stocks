from setuptools import find_packages, setup

setup(
    name="stocks",
    author="Nicholas Maher",
    version="0.0.1",
    packages=find_packages(),
    python_requires="~=3.7",
    install_requires=[
        "alpaca-trade-api",
        "click",
        "pandas",
        "pyyaml",
        "sklearn",
        "tensorflow",
    ],
    entry_points={"console_scripts": ["stocks = stocks.cli:cli"]},
)
