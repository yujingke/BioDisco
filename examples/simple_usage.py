#!/usr/bin/env python3
"""
Simple BioDisco Usage Example

This example demonstrates the basic usage of BioDisco with the simple generate() interface.
"""

import BioDisco

def main():
    """Demonstrate basic BioDisco usage"""
    
    print("🧬 BioDisco Simple Usage Example 🤖")
    print("=" * 50)
    
    # Example 1: Simple disease query
    print("\n1. Simple disease-based discovery:")
    print("   params = {")
    print("       start_year=2019,")
    print("       min_results=3,")
    print("       max_results=10,")
    print("       n_iterations=1,")
    print("       max_articles_per_round=10")
    print("   }")
    print("   BioDisco.generate('T Cell Exhaustion Mechanisms and Therapeutic Targets in NSCLC', params=params)")
    params = dict(
        start_year=2019,
        min_results=3,
        max_results=10,
        n_iterations=1,
        max_articles_per_round=10,
    )
    results = BioDisco.generate("T Cell Exhaustion Mechanisms and Therapeutic Targets in NSCLC", params=params)
    print('Results:', results)

    print("\n✅ BioDisco is ready to use!")
    print("\nTo actually run the discovery process:")
    print("1. Set up your .env file with API keys")
    print("2. Uncomment the function calls above")
    print("3. Run this script again")
    
    print("\n📖 Check README.md for detailed setup instructions")

if __name__ == "__main__":
    main()
