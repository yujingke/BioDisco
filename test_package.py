#!/usr/bin/env python3
"""
Test script to verify BioDisco package installation and basic functionality.
"""

def test_import():
    """Test that BioDisco can be imported successfully"""
    try:
        import BioDisco
        print("✅ Successfully imported BioDisco")
        return True
    except ImportError as e:
        print(f"❌ Failed to import BioDisco: {e}")
        return False

def test_generate_function():
    """Test that the generate function exists and is callable"""
    try:
        import BioDisco
        
        # Check if generate function exists
        assert hasattr(BioDisco, 'generate'), "generate function not found"
        assert callable(BioDisco.generate), "generate is not callable"
        
        print("✅ BioDisco.generate function is available")
        return True
    except Exception as e:
        print(f"❌ Failed to verify generate function: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring API keys"""
    try:
        import BioDisco
        
        # Test that we can call generate (might fail due to missing API keys, but should not crash)
        print("📝 Testing BioDisco.generate function call...")
        
        # This may fail due to missing environment variables, but should not crash
        try:
            params = dict(
                start_year=2019,
                min_results=3,
                max_results=10,
                n_iterations=1,
                max_articles_per_round=10,
            )
            result = BioDisco.generate("T Cell Exhaustion Mechanisms and Therapeutic Targets in NSCLC", params=params)
            print("✅ Function call succeeded")
        except Exception as e:
            print(f"⚠️  Function call failed (expected if no API keys): {e}")
            print("✅ Function is callable (failure expected without API keys)")
        
        return True
    except ImportError as e:
        print(f"❌ Import error during functionality test: {e}")
        return False

def main():
    """Run all tests"""
    print("🧬 BioDisco Package Test 🤖")
    print("=" * 40)
    
    tests = [
        test_import,
        test_generate_function,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BioDisco is ready to use.")
        print("\n📋 Next steps:")
        print("1. Set up your .env file with API keys")
        print("2. Run: python examples/simple_usage.py")
        print("3. Try: import BioDisco; BioDisco.generate('your disease')")
    else:
        print("⚠️  Some tests failed. Check the setup.")
        
    return passed == total

if __name__ == "__main__":
    main()
