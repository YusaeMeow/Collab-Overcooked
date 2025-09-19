#!/usr/bin/env python
"""Setup script for Collab-Overcooked."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="collab-overcooked",
    version="1.0.0",
    author="Collab-Overcooked Team",
    author_email="contact@collab-overcooked.org",
    description="A Multi-Agent Collaborative Benchmark based on Overcooked-AI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/Collab-Overcooked",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.8",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "collab-overcooked=collab_overcooked.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "collab_overcooked": ["configs/*.txt", "configs/*.yaml"],
    },
    zip_safe=False,
)