#!/usr/bin/env python3
"""End-to-end integration test for Triple Sounio Ecosystem

This script validates that all three projects (sounio-py, sounio-jupyter, drug-discovery)
work together correctly. Designed to run on Day 12 after all tracks complete.

Test 1: sounio-py Knowledge class instantiation and basic operations
Test 2: drug-discovery pipeline runs end-to-end via sounio.run_file()
Test 3: Jupyter kernel is installed and discoverable
"""

import sys
import subprocess
import os
import re
from pathlib import Path


def test_sounio_py_knowledge():
    """Test 1: sounio-py Knowledge class works with epsilon and provenance"""
    try:
        import sounio

        # Create Knowledge object with value, epsilon, and provenance
        x = sounio.Knowledge(42.0, 0.1, "test_source")

        # Verify attributes
        assert abs(x.value - 42.0) < 1e-6, f"Expected value=42.0, got {x.value}"
        assert abs(x.epsilon - 0.1) < 1e-6, f"Expected epsilon=0.1, got {x.epsilon}"
        assert x.provenance == "test_source", f"Expected provenance='test_source', got {x.provenance}"

        # Verify string representation
        str_repr = str(x)
        assert "42" in str_repr and "0.1" in str_repr, f"String repr missing values: {str_repr}"

        print("✅ Test 1 PASS: sounio-py Knowledge class")
        return True
    except ImportError as e:
        print(f"⚠️  Test 1 SKIP: sounio not installed (expected on Day 11): {e}")
        return None
    except Exception as e:
        print(f"❌ Test 1 FAIL: {e}")
        return False


def test_drug_discovery_pipeline():
    """Test 2: drug-discovery pipeline runs end-to-end"""
    try:
        # Verify souc binary exists
        souc_path = os.environ.get('SOUC')
        if not souc_path:
            souc_path = "/home/demetrios/RustroverProjects/sounio/artifacts/omega/souc-bin/souc-linux-x86_64-jit"

        if not os.path.exists(souc_path):
            print(f"⚠️  Test 2 SKIP: souc binary not found at {souc_path}")
            return None

        pipeline_path = os.path.join(
            os.path.dirname(__file__),
            "drug-discovery/examples/full_pipeline.sio"
        )

        if not os.path.exists(pipeline_path):
            print(f"⚠️  Test 2 SKIP: pipeline file not found at {pipeline_path}")
            return None

        # Set up environment
        env = os.environ.copy()
        env['SOUC'] = souc_path
        env['SOUNIO_STDLIB_PATH'] = os.path.join(
            os.path.dirname(__file__),
            "../stdlib"
        )

        # Run pipeline
        result = subprocess.run(
            [souc_path, 'run', pipeline_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

        # Check for success markers
        output = result.stdout + result.stderr
        assert result.returncode == 0, f"Pipeline failed with exit code {result.returncode}"
        assert "Pipeline complete" in output or "Stage" in output, \
            f"No pipeline markers in output: {output[:200]}"

        print("✅ Test 2 PASS: drug-discovery pipeline runs end-to-end")
        return True
    except subprocess.TimeoutExpired:
        print(f"❌ Test 2 FAIL: Pipeline timeout (30s)")
        return False
    except Exception as e:
        print(f"❌ Test 2 FAIL: {e}")
        return False


def test_jupyter_kernel_installed():
    """Test 3: jupyter kernel is installed and discoverable"""
    try:
        # Check if jupyter is installed
        try:
            result = subprocess.run(
                ["jupyter", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                print("⚠️  Test 3 SKIP: jupyter not installed")
                return None
        except FileNotFoundError:
            print("⚠️  Test 3 SKIP: jupyter command not found")
            return None

        # List kernels
        result = subprocess.run(
            ["jupyter", "kernelspec", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, f"kernelspec list failed: {result.stderr}"
        assert "sounio" in result.stdout.lower(), \
            f"'sounio' kernel not found in: {result.stdout}"

        print("✅ Test 3 PASS: jupyter sounio kernel installed")
        return True
    except subprocess.TimeoutExpired:
        print("❌ Test 3 FAIL: jupyter kernelspec list timeout")
        return False
    except Exception as e:
        print(f"❌ Test 3 FAIL: {e}")
        return False


def test_knowledge_parsing():
    """Bonus test: validate canonical Knowledge output format parsing"""
    try:
        # Pattern from plan: Knowledge { value: 42.000 epsilon: 0.850 prov: "source_name" }
        pattern = r"Knowledge \{ value: ([\d.e+-]+) epsilon: ([\d.e+-]+) prov: \"([^\"]+)\" \}"

        test_string = 'Knowledge { value: 42.000 epsilon: 0.850 prov: "test_source" }'
        match = re.search(pattern, test_string)

        assert match is not None, f"Pattern failed to match canonical format"
        value, epsilon, prov = match.groups()
        assert abs(float(value) - 42.0) < 1e-3
        assert abs(float(epsilon) - 0.85) < 1e-3
        assert prov == "test_source"

        print("✅ Bonus PASS: canonical Knowledge format parsing works")
        return True
    except Exception as e:
        print(f"⚠️  Bonus FAIL: {e}")
        return False


def main():
    """Run all integration tests"""
    print("=" * 70)
    print("Triple Sounio Ecosystem Integration Tests (Day 12)")
    print("=" * 70)
    print()

    tests = [
        ("Knowledge class", test_sounio_py_knowledge),
        ("Pipeline execution", test_drug_discovery_pipeline),
        ("Jupyter kernel", test_jupyter_kernel_installed),
        ("Bonus: Knowledge parsing", test_knowledge_parsing),
    ]

    results = []
    for name, test_func in tests:
        print(f"Running: {name}...")
        result = test_func()
        results.append(result)
        print()

    # Summary
    print("=" * 70)
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)

    print(f"Results: {passed} PASS, {failed} FAIL, {skipped} SKIP")

    if failed == 0 and passed > 0:
        print("\n🎉 ALL CRITICAL TESTS PASS")
        return 0
    elif failed > 0:
        print(f"\n❌ {failed} tests failed")
        return 1
    else:
        print("\n⚠️  All tests skipped (dependencies not ready)")
        return 2


if __name__ == "__main__":
    sys.exit(main())
