"""
BioDisco: AI-powered Biomedical Discovery Agent System

This package provides tools for biomedical hypothesis generation, literature mining,
and knowledge graph integration for scientific discovery.
"""

__version__ = "0.1.0"
__author__ = "BioDisco Team"

# Import main modules
from . import agents_auto_background
from . import agents_evidence
from . import utils

# Import key functions for easy access
from .agents_auto_background import generate

__all__ = [
    "generate",  # Main public interface
]
