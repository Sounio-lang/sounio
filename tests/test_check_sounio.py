#!/usr/bin/env python3
"""Test script to verify check_sounio.sh functionality"""

import subprocess
import os
import sys

def run_check_sounio(file_path):
    """Run check_sounio.sh on a file and return output"""
    try:
        # Make sure the script is executable
        os.chmod('check_sounio.sh', 0o755)
        
        # Run the script
        result = subprocess.run(
            ['bash', 'check_sounio.sh', file_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, '', str(e)

def test_rusty_file():
    """Test the script on a file with Rust-isms"""
    print("Testing check_sounio.sh on test_rusty_sounio.sio...")
    returncode, stdout, stderr = run_check_sounio('test_rusty_sounio.sio')
    
    print(f"Return code: {returncode}")
    print(f"Stdout:\n{stdout}")
    if stderr:
        print(f"Stderr:\n{stderr}")
    
    # Check for expected errors
    expected_patterns = [
        'Found &mut',
        'Found Vec',
        'Rust-like method call',
        'unnecessary semicolon',
        "may need 'with Div' effect"
    ]
    
    found_patterns = []
    for pattern in expected_patterns:
        if pattern in stdout:
            found_patterns.append(pattern)
            print(f"✓ Found expected pattern: {pattern}")
        else:
            print(f"✗ Missing expected pattern: {pattern}")
    
    # Should have errors
    if returncode != 0:
        print("✓ Script correctly returned non-zero exit code for errors")
    else:
        print("✗ Script should return non-zero for files with errors")
    
    return len(found_patterns) == len(expected_patterns) and returncode != 0

def test_good_file():
    """Test the script on a good Sounio file"""
    print("\nTesting check_sounio.sh on examples/hello.sio...")
    returncode, stdout, stderr = run_check_sounio('examples/hello.sio')
    
    print(f"Return code: {returncode}")
    print(f"Stdout (first 500 chars):\n{stdout[:500]}")
    
    # Should have no errors
    if returncode == 0:
        print("✓ Script correctly returned zero exit code for good file")
    else:
        print("✗ Script should return zero for good files")
        print(f"Full output:\n{stdout}")
    
    return returncode == 0

def test_extern_c_file():
    """Test that extern C blocks don't generate false-positive semicolon errors"""
    print("\nTesting check_sounio.sh on test_extern_c_sounio.sio...")
    returncode, stdout, stderr = run_check_sounio('test_extern_c_sounio.sio')

    print(f"Return code: {returncode}")
    print(f"Stdout (first 500 chars):\n{stdout[:500]}")

    # Should pass with no critical errors (extern "C" semicolons must not be flagged)
    # Use 'unnecessary semicolon(' to match the error form "Found N unnecessary semicolon(s)"
    # but NOT the success form "No unnecessary semicolons"
    if 'unnecessary semicolon(' in stdout or returncode != 0:
        print("✗ extern C semicolons incorrectly flagged as errors")
        print(f"Full output:\n{stdout}")
        return False

    print("✓ extern C file passed with no errors")
    return True


def main():
    """Main test function"""
    print("=== Testing check_sounio.sh ===\n")

    # Test 1: File with Rust-isms
    test1_passed = test_rusty_file()

    # Test 2: Good Sounio file
    test2_passed = test_good_file()

    # Test 3: extern "C" block — must not flag semicolons
    test3_passed = test_extern_c_file()

    # Summary
    print("\n=== Test Summary ===")
    print(f"Test 1 (Rusty file): {'PASS' if test1_passed else 'FAIL'}")
    print(f"Test 2 (Good file): {'PASS' if test2_passed else 'FAIL'}")
    print(f"Test 3 (extern C file): {'PASS' if test3_passed else 'FAIL'}")

    if test1_passed and test2_passed and test3_passed:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())