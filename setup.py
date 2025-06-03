#!/usr/bin/env python3
"""
Setup script for Particle Tracking Application
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\\n')
    requirements = [req for req in requirements if req and not req.startswith('#')]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "scikit-image>=0.18.0",
        "PyQt6>=6.0.0",
        "pyqtgraph>=0.12.0",
        "matplotlib>=3.4.0",
    ]

setup(
    name="particle-tracker",
    version="1.0.0",
    description="A comprehensive application for analyzing particle trajectories from microscopy data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/particle-tracker",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-qt>=4.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "optional": [
            "trackpy>=0.5.0",
            "opencv-python>=4.5.0",
            "numba>=0.54.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "particle-tracker=particle_tracker.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    keywords="particle tracking, microscopy, image analysis, trajectories, biophysics",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/particle-tracker/issues",
        "Source": "https://github.com/yourusername/particle-tracker",
        "Documentation": "https://particle-tracker.readthedocs.io/",
    },
)
