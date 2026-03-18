#!/usr/bin/env python3
"""End-to-end integration test for Triple Sounio Ecosystem

This script validates that all three projects (sounio-py, sounio-jupyter, drug-discovery)
work together correctly. Designed to run on Day 12 after all tracks complete.

Test 1: sounio-py Knowledge class instantiation and basic operations
Test 2: drug-discovery pipeline runs end-to-end via sounio.run_file()
Test 3: Jupyter kernel is installed and discoverable
Test 4: Knowledge round-trip (repr → parse → verify)
Test 5: Pipeline Knowledge parsing (verify all 9 values are parseable)
Test 6: Knowledge arithmetic (GUM propagation correctness)
"""

import sys
import subprocess
import os
import re
import math
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


def test_knowledge_round_trip():
    """Test 4: Knowledge → repr → parse → verify round-trip"""
    try:
        import sounio

        # Create a Knowledge object
        original = sounio.Knowledge(42.0, 0.1, "test")
        repr_str = repr(original)

        # Parse using the canonical regex
        pattern = r'Knowledge\s*\{\s*value:\s*([0-9eE+\-.]+)\s+epsilon:\s*([0-9eE+\-.]+)\s+prov:\s*"([^"]*)"\s*\}'
        match = re.search(pattern, repr_str)

        assert match is not None, f"Regex failed to match: {repr_str}"
        value, epsilon, prov = match.groups()

        # Reconstruct
        parsed = sounio.Knowledge(float(value), float(epsilon), prov)

        # Verify
        assert abs(parsed.value - original.value) < 1e-6
        assert abs(parsed.epsilon - original.epsilon) < 1e-6
        assert parsed.provenance == original.provenance

        print("✅ Test 4 PASS: Knowledge round-trip works")
        return True
    except ImportError:
        print("⚠️  Test 4 SKIP: sounio not installed")
        return None
    except Exception as e:
        print(f"❌ Test 4 FAIL: {e}")
        return False


def test_pipeline_knowledge_parsing():
    """Test 5: Run pipeline and verify all 9 Knowledge values are parseable"""
    try:
        import sounio

        souc_path = os.environ.get('SOUC')
        if not souc_path:
            souc_path = "/home/demetrios/RustroverProjects/sounio/artifacts/omega/souc-bin/souc-linux-x86_64-jit"

        if not os.path.exists(souc_path):
            print(f"⚠️  Test 5 SKIP: souc binary not found at {souc_path}")
            return None

        pipeline_path = os.path.join(
            os.path.dirname(__file__),
            "drug-discovery/examples/full_pipeline.sio"
        )

        if not os.path.exists(pipeline_path):
            print(f"⚠️  Test 5 SKIP: pipeline file not found at {pipeline_path}")
            return None

        # Set up environment
        env = os.environ.copy()
        env['SOUC'] = souc_path
        stdlib_path = os.environ.get('SOUNIO_STDLIB_PATH')
        if not stdlib_path:
            stdlib_path = os.path.join(
                os.path.dirname(__file__),
                "../stdlib"
            )
        env['SOUNIO_STDLIB_PATH'] = stdlib_path

        # Run pipeline
        result = subprocess.run(
            [souc_path, 'run', pipeline_path],
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )

        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

        # Parse Knowledge values
        executor = sounio.SounioExecutor(souc_path=souc_path, stdlib_path=stdlib_path)
        knowledge_values = executor._parse_knowledge(result.stdout)

        assert len(knowledge_values) == 9, \
            f"Expected 9 Knowledge values, got {len(knowledge_values)}"

        # Verify each value is valid
        for i, k in enumerate(knowledge_values):
            assert isinstance(k, sounio.Knowledge), f"Value {i} is not Knowledge"
            assert isinstance(k.value, float), f"Value {i} value is not float"
            assert isinstance(k.epsilon, float), f"Value {i} epsilon is not float"
            assert isinstance(k.provenance, str), f"Value {i} provenance is not str"

        # Verify expected provenances
        expected_provs = [
            "lipinski_screen", "pk_half_life", "pk_tmax", "pk_cmax", "pk_auc",
            "trial_efficacy", "trial_adverse", "therapeutic_index", "pipeline_decision"
        ]
        actual_provs = [k.provenance for k in knowledge_values]
        assert actual_provs == expected_provs, \
            f"Provenance mismatch: {actual_provs}"

        print("✅ Test 5 PASS: Pipeline Knowledge parsing works (9 values)")
        return True
    except ImportError:
        print("⚠️  Test 5 SKIP: sounio not installed")
        return None
    except subprocess.TimeoutExpired:
        print("❌ Test 5 FAIL: Pipeline timeout (60s)")
        return False
    except Exception as e:
        print(f"❌ Test 5 FAIL: {e}")
        return False


def test_knowledge_arithmetic():
    """Test 6: Knowledge arithmetic GUM propagation correctness"""
    try:
        import sounio

        # Test addition: ε = sqrt(ε1² + ε2²)
        k1 = sounio.Knowledge(10.0, 0.5, "source1")
        k2 = sounio.Knowledge(20.0, 0.3, "source2")
        result = k1 + k2

        expected_eps = math.sqrt(0.5**2 + 0.3**2)
        assert result.value == 30.0
        assert abs(result.epsilon - expected_eps) < 1e-10, \
            f"Addition epsilon mismatch: {result.epsilon} vs {expected_eps}"

        # Test multiplication with relative uncertainty
        k1 = sounio.Knowledge(2.0, 0.1, "measure1")
        k2 = sounio.Knowledge(3.0, 0.15, "measure2")
        result = k1 * k2

        rel_eps_squared = (0.1/2.0)**2 + (0.15/3.0)**2
        expected_eps = 6.0 * math.sqrt(rel_eps_squared)
        assert result.value == 6.0
        assert abs(result.epsilon - expected_eps) < 1e-10, \
            f"Multiplication epsilon mismatch: {result.epsilon} vs {expected_eps}"

        # Test scalar multiplication: ε = |factor| * εa
        k = sounio.Knowledge(7.0, 0.2, "source")
        result = k * 3.0

        assert result.value == 21.0
        assert result.epsilon == 0.6
        assert result.provenance == "source"

        # Test division
        k1 = sounio.Knowledge(10.0, 0.5, "numerator")
        k2 = sounio.Knowledge(2.0, 0.1, "denominator")
        result = k1 / k2

        rel_eps_squared = (0.5/10.0)**2 + (0.1/2.0)**2
        expected_eps = 5.0 * math.sqrt(rel_eps_squared)
        assert result.value == 5.0
        assert abs(result.epsilon - expected_eps) < 1e-10

        print("✅ Test 6 PASS: Knowledge arithmetic GUM propagation correct")
        return True
    except ImportError:
        print("⚠️  Test 6 SKIP: sounio not installed")
        return None
    except Exception as e:
        print(f"❌ Test 6 FAIL: {e}")
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
        ("Test 4: Knowledge round-trip", test_knowledge_round_trip),
        ("Test 5: Pipeline Knowledge parsing", test_pipeline_knowledge_parsing),
        ("Test 6: Knowledge arithmetic", test_knowledge_arithmetic),
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
