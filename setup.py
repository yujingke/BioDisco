#!/usr/bin/env python3
"""Setup script for BioDisco package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from requirements file."""
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

requirements = read_requirements('requirements.txt')

setup(
    name="biodisco",
    version="0.1.0",
    author="BioDisco Team",
    # author_email="contact@biodisco.ai",
    description="AI-powered Biomedical Discovery Agent System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yujingke/BioDisco",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    # extras_require={
    #     "dev": [
    #         "pytest>=6.0",
    #         "pytest-cov>=2.0",
    #         "black>=21.0",
    #         "flake8>=3.8",
    #         "mypy>=0.8",
    #         "pre-commit>=2.0",
    #     ],
    #     "docs": [
    #         "sphinx>=4.0",
    #         "sphinx-rtd-theme>=1.0",
    #         "myst-parser>=0.15",
    #     ],
    # },
    # entry_points={
    #     "console_scripts": [
    #         "biodisco-cli=BioDisco.cli:main",
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
    keywords="biomedical, AI, agents, hypothesis generation, literature mining, knowledge graph",
    project_urls={
        "Bug Reports": "https://github.com/yujingke/BioDisco/issues",
        "Source": "https://github.com/yujingke/BioDisco",
        # "Documentation": "https://biodisco.readthedocs.io/",
    },
)
