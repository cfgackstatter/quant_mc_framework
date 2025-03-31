from setuptools import setup, find_packages

setup(
    name="quant_mc_framework",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "cvxpy",
        "signac",
    ],
    author="cfgackstatter",
    author_email="your-email@example.com",
    description="A framework for Monte Carlo simulations of quantitative trading strategies",
)