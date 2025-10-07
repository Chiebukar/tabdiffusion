# setup.py (minimal - you can replace with pyproject.toml if you prefer)
from setuptools import setup, find_packages

setup(
    name="tabdiffusion",
    version="0.1.0",
    description="Tabular conditional diffusion generator (TabDiffusion) - transformer based",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Samuel Ozechi",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "matplotlib",
        "seaborn"
    ],
    python_requires=">=3.8",
)
