#!/usr/bin/env python3
"""
Basic Usage Example for BioDisco Package

This example demonstrates the core functionality of BioDisco including:
- Creating and managing hypotheses
- Literature management with PubMed integration
- Knowledge graph operations
- Evidence linking

Author: BioDisco Team
"""

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from BioDisco.agents_auto_background import (
    generate
)

# Configuration parameters
params = dict(
    start_year=2019,
    min_results=3,
    max_results=10,
    n_iterations=1,
    max_articles_per_round=10,
)

def main():
    """Main demonstration function."""

    hypothesis = generate("T Cell Exhaustion Mechanisms and Therapeutic Targets in NSCLC", params=params)

    print(f"Final hypothesis: {hypothesis}")

if __name__ == "__main__":
    # Run basic example
    main()
