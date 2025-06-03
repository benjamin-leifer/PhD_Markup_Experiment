#!/usr/bin/env python
"""
Setup script for the battery_analysis package.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="battery_analysis",
    version="0.1.0",
    description="A package for ingesting, analyzing, and managing electrochemical test data (battery tests)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Battery Research Team",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/battery_analysis",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "mongoengine>=0.24.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "reportlab>=3.5.0"
    ],
    extras_require={
        "biologic_mpr": ["galvani>=0.4.0"],  # Optional extra for reading .mpr files
        "eis": ["impedance>=1.0.0"],  # Optional extra for EIS analysis
        "gui": ["matplotlib>=3.4.0"],  # GUI dependencies (tkinter is included with Python)
        "pybamm": ["pybamm>=23.5"],  # PyBAMM for battery modeling
        "all": ["galvani>=0.4.0", "impedance>=1.0.0", "matplotlib>=3.4.0", "pybamm>=23.5"],  # All optional dependencies
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="battery, electrochemistry, analysis, mongodb, data management",
)
