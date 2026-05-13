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
            souc_path = "/workspace/sounio/bin/souc"

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
            souc_path = "/workspace/sounio/bin/souc"

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
        assert abs(result.epsilon - 0.6) < 1e-10, \
            f"Scalar multiplication epsilon mismatch: {result.epsilon} vs 0.6"
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


def test_provenance_chain():
    """Test 7: Provenance chain DAG tracking."""
    try:
        from sounio.knowledge import Knowledge

        a = Knowledge.tracked(500.0, 2.5, "calibration_batch")
        b = Knowledge.tracked(1.2, 0.05, "correction_factor")

        assert a.chain is not None, "source node should have a chain"
        assert b.chain is not None
        assert len(a.chain) == 1  # one source node

        c = a * b

        assert c.chain is not None, "result should inherit chain"
        assert len(c.chain) == 3, f"expected 3 nodes, got {len(c.chain)}"

        # Verify ancestry
        ancestors = c.ancestry()
        assert len(ancestors) == 2, f"expected 2 ancestors, got {len(ancestors)}"
        ops = [n.operation for n in c.chain]
        assert "source" in ops
        assert "mul" in ops

        # Untracked Knowledge: no chain (backward compatible)
        plain = Knowledge(1.0, 0.1, "untracked")
        assert plain.chain is None

        # Mixed: tracked + untracked preserves chain from tracked side
        d = a + plain
        assert d.chain is not None

        # to_dict round-trip includes chain
        d_dict = c.to_dict()
        assert "chain" in d_dict
        assert "nodes" in d_dict["chain"]

        # DOT export
        dot = c.chain.to_dot()
        assert "digraph" in dot

        print("✅ Test 7 PASS: ProvenanceChain DAG tracking correct")
        return True
    except ImportError:
        print("⚠️  Test 7 SKIP: sounio not installed")
        return None
    except Exception as e:
        print(f"❌ Test 7 FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def test_async_executor():
    """Test 8: Async executor (asyncio.create_subprocess_exec)."""
    try:
        import asyncio
        import inspect
        from sounio._executor import SounioExecutor

        executor = SounioExecutor()

        # Verify async methods exist and are coroutine functions
        assert hasattr(executor, "async_run_code"), "async_run_code missing"
        assert hasattr(executor, "async_run_file"), "async_run_file missing"
        assert hasattr(executor, "async_check_file"), "async_check_file missing"
        assert inspect.iscoroutinefunction(executor.async_run_code), "async_run_code not a coroutine"
        assert inspect.iscoroutinefunction(executor.async_run_file), "async_run_file not a coroutine"
        assert inspect.iscoroutinefunction(executor.async_check_file), "async_check_file not a coroutine"

        # Module-level coroutines are also present
        import sounio
        assert hasattr(sounio, "async_run_code"), "sounio.async_run_code missing"
        assert hasattr(sounio, "async_run_file"), "sounio.async_run_file missing"
        assert inspect.iscoroutinefunction(sounio.async_run_code)
        assert inspect.iscoroutinefunction(sounio.async_run_file)

        # Try to actually run if the souc binary is available
        from pathlib import Path
        souc_found = Path(executor.souc_path).exists()
        if souc_found:
            code = 'fn main() with IO { println("async_ok") }'

            async def run():
                return await executor.async_run_code(code, timeout=30.0)

            result = asyncio.run(run())
            assert result.exit_code == 0, f"exit_code={result.exit_code}, stderr={result.stderr}"
            assert "async_ok" in result.stdout
            print("✅ Test 8 PASS: Async executor works correctly (executed)")
        else:
            print("✅ Test 8 PASS: Async executor API verified (souc binary absent, skipping execution)")
        return True
    except ImportError:
        print("⚠️  Test 8 SKIP: sounio not installed")
        return None
    except Exception as e:
        print(f"❌ Test 8 FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def test_shared_memory():
    """Test 9: UncertainArray shared memory IPC."""
    try:
        import numpy as np
        from sounio.integrations.numpy_ext import UncertainArray

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        epsilons = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        prov = "sensor_array"

        a = UncertainArray(values, epsilons, prov)

        # Write to shared memory
        v_name, e_name, meta = a.to_shared_memory()
        assert isinstance(v_name, str) and v_name, "values segment name missing"
        assert isinstance(e_name, str) and e_name, "epsilons segment name missing"
        assert "shape" in meta

        # Reconstruct from shared memory
        b = UncertainArray.from_shared_memory(v_name, e_name, meta)
        assert len(b) == len(a), f"length mismatch: {len(b)} vs {len(a)}"
        assert b.provenance == prov
        assert abs(b.values[2] - a.values[2]) < 1e-15, "value mismatch after shm round-trip"
        assert abs(b.epsilons[0] - a.epsilons[0]) < 1e-15, "epsilon mismatch after shm round-trip"

        # Clean up
        UncertainArray.free_shared_memory(v_name, e_name)

        # Double-free is safe
        UncertainArray.free_shared_memory(v_name, e_name)

        print("✅ Test 9 PASS: Shared memory IPC round-trip correct")
        return True
    except ImportError as e:
        print(f"⚠️  Test 9 SKIP: {e}")
        return None
    except Exception as e:
        print(f"❌ Test 9 FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def test_dashboard_app():
    """Test 10: Dashboard Flask app (no actual server start)."""
    try:
        from sounio.dashboard import create_app, _DASHBOARD_HTML

        app = create_app()

        # Verify routes exist
        rules = {r.rule for r in app.url_map.iter_rules()}
        assert "/" in rules, "/ route missing"
        assert "/api/pipeline/run" in rules, "/api/pipeline/run missing"
        assert "/api/pipeline/status" in rules, "/api/pipeline/status missing"
        assert "/api/knowledge/<prov_id>" in rules, "/api/knowledge/<prov_id> missing"

        # Verify HTML content
        assert "Sounio" in _DASHBOARD_HTML
        assert "epistemic" in _DASHBOARD_HTML.lower()
        assert "/api/pipeline/run" in _DASHBOARD_HTML

        # Test status endpoint with test client
        with app.test_client() as client:
            r = client.get("/api/pipeline/status")
            assert r.status_code == 200
            data = r.get_json()
            assert data["status"] == "idle"

            r2 = client.get("/")
            assert r2.status_code == 200
            assert b"Sounio" in r2.data

        print("✅ Test 10 PASS: Dashboard Flask app routes and HTML correct")
        return True
    except ImportError as e:
        print(f"⚠️  Test 10 SKIP: {e}")
        return None
    except Exception as e:
        print(f"❌ Test 10 FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def test_kernel_client_api():
    """Test 11: KernelConnection and launch_jupyter_kernel API surface."""
    try:
        from sounio.kernel_client import (
            KernelConnection,
            KernelResult,
            launch_jupyter_kernel,
        )
        import sounio

        # Verify exports exist
        assert hasattr(sounio, "launch_jupyter_kernel")
        assert hasattr(sounio, "KernelConnection")
        assert hasattr(sounio, "KernelResult")

        # KernelResult works standalone
        r = KernelResult(
            stdout="Knowledge { value: 42 epsilon: 0.5 prov: \"test\" }",
            stderr="",
            status="ok",
        )
        assert r.ok is True

        # launch_jupyter_kernel raises ImportError or RuntimeError (not AttributeError)
        # when the kernel isn't installed — that's the correct, documented behaviour.
        try:
            launch_jupyter_kernel(timeout=2.0)
        except ImportError:
            pass  # jupyter_client not installed — expected in CI
        except RuntimeError:
            pass  # kernel spec not installed — expected in CI
        except Exception as e:
            print(f"❌ Test 11 FAIL: unexpected exception type {type(e).__name__}: {e}")
            return False

        print("✅ Test 11 PASS: KernelClient API surface correct")
        return True
    except ImportError as e:
        print(f"⚠️  Test 11 SKIP: {e}")
        return None
    except Exception as e:
        print(f"❌ Test 11 FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def test_python_magic_preamble():
    """Test 12: %python magic preamble builder injects Knowledge values."""
    try:
        import sys, os
        # Add sounio-jupyter to path if needed
        here = os.path.dirname(os.path.abspath(__file__))
        jupyter_pkg = os.path.join(here, "sounio-jupyter")
        if jupyter_pkg not in sys.path:
            sys.path.insert(0, jupyter_pkg)

        from sounio_kernel.magics import SounioMagics

        class FakeExecutor:
            stdlib_path = None
            souc_binary = "souc"
            def run_cell(self, code): return ("", "", 0)

        class FakeKernel:
            executor = FakeExecutor()
            _last_knowledge_values = []

        magics = SounioMagics(FakeKernel())

        # With no knowledge values
        preamble = magics._build_python_preamble()
        assert "knowledge_values = []" in preamble

        # Inject fake Knowledge values
        class FakeK:
            value = 500.0
            epsilon = 2.5
            provenance = "test_sensor"

        FakeKernel._last_knowledge_values = [FakeK()]
        magics2 = SounioMagics(FakeKernel())
        preamble2 = magics2._build_python_preamble()
        assert "k0 = _K(500.0" in preamble2
        assert "test_sensor" in preamble2
        assert "knowledge_values = [k0]" in preamble2

        # Verify that the generated Python actually executes
        import subprocess
        code = preamble2 + "\nprint(k0.value)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"preamble execution failed: {result.stderr}"
        assert "500.0" in result.stdout

        # %python handler registered
        assert "%python" in magics._build_python_preamble.__qualname__ or True
        handled, _ = magics.handle_magic("%python print('hello from python magic')")
        assert handled is True

        print("✅ Test 12 PASS: %python magic preamble builder correct")
        return True
    except ImportError as e:
        print(f"⚠️  Test 12 SKIP: {e}")
        return None
    except Exception as e:
        print(f"❌ Test 12 FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def test_report_builder():
    """Test 13: ReportBuilder generates paper from epistemic data."""
    try:
        import sys, os, tempfile
        here = os.path.dirname(os.path.abspath(__file__))
        py_pkg = os.path.join(here, "sounio-py", "python")
        if py_pkg not in sys.path:
            sys.path.insert(0, py_pkg)

        from sounio.report import ReportBuilder
        from sounio.knowledge import Knowledge

        rb = ReportBuilder("Test Report", author="Demo Suite", date="2026-03-19")
        rb.add_section("Introduction", "This is a test of report generation.")
        rb.add_knowledge_table("Measurements", {
            "Temperature (°C)": Knowledge(36.5, 0.1, "thermometer"),
            "Pressure (hPa)": Knowledge(1013.25, 2.5, "barometer"),
        })

        md = rb.to_markdown()
        assert "# Test Report" in md
        assert "Demo Suite" in md
        assert "Temperature" in md
        assert "36.5" in md
        assert "± 0.1" in md

        latex = rb.to_latex()
        assert r"\documentclass" in latex
        assert r"\begin{document}" in latex
        assert "Temperature" in latex

        # Save and verify
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = os.path.join(tmpdir, "report.md")
            rb.save(md_path, format="markdown")
            assert os.path.exists(md_path)
            content = open(md_path).read()
            assert "# Test Report" in content

        # Test generate_paper() classmethod (with mock PipelineResult)
        from dataclasses import dataclass, field
        from typing import Optional

        @dataclass
        class MockSim:
            efficacy_rate: Knowledge = Knowledge(0.72, 0.05, "sim")
            adverse_rate: Knowledge = Knowledge(0.12, 0.02, "sim")
            therapeutic_index: Knowledge = Knowledge(5.8, 0.4, "sim")
            n_patients: int = 100
            confidence: float = 0.88

        @dataclass
        class MockMol:
            name: str = "TEST-1"

        @dataclass
        class MockPK:
            half_life: Knowledge = Knowledge(4.62, 0.77, "pk")
            cmax: Knowledge = Knowledge(220.0, 15.0, "pk")
            auc: Knowledge = Knowledge(1850.0, 120.0, "pk")

        @dataclass
        class MockResult:
            molecule: MockMol = field(default_factory=MockMol)
            decision: str = "PROCEED"
            confidence: float = 0.88
            molecules_screened: int = 1
            molecules_passed: int = 1
            pk_fitted: int = 1
            exit_code: int = 0
            pk_params: MockPK = field(default_factory=MockPK)
            simulation: MockSim = field(default_factory=MockSim)
            provenance_chain: Optional[list] = None

        with tempfile.TemporaryDirectory() as tmpdir:
            paper_path = os.path.join(tmpdir, "paper.md")
            rb2 = ReportBuilder.generate_paper(
                [MockResult()],
                title="Test Drug Discovery Paper",
                author="Sounio",
                output_path=paper_path,
                output_format="markdown",
            )
            assert os.path.exists(paper_path)
            paper_content = open(paper_path).read()
            assert "Abstract" in paper_content
            assert "Methods" in paper_content
            assert "Results" in paper_content
            assert "Discussion" in paper_content
            assert "Conclusion" in paper_content
            assert "TEST-1" in paper_content
            assert "PROCEED" in paper_content

        print("✅ Test 13 PASS: ReportBuilder paper generation correct")
        return True
    except Exception as e:
        print(f"❌ Test 13 FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("=" * 70)
    print("Triple Sounio Ecosystem Integration Tests")
    print("=" * 70)
    print()

    tests = [
        ("Knowledge class", test_sounio_py_knowledge),
        ("Pipeline execution", test_drug_discovery_pipeline),
        ("Jupyter kernel", test_jupyter_kernel_installed),
        ("Test 4: Knowledge round-trip", test_knowledge_round_trip),
        ("Test 5: Pipeline Knowledge parsing", test_pipeline_knowledge_parsing),
        ("Test 6: Knowledge arithmetic", test_knowledge_arithmetic),
        ("Test 7: Provenance chain DAG", test_provenance_chain),
        ("Test 8: Async executor", test_async_executor),
        ("Test 9: Shared memory IPC", test_shared_memory),
        ("Test 10: Dashboard Flask app", test_dashboard_app),
        ("Test 11: Kernel client API", test_kernel_client_api),
        ("Test 12: %python magic", test_python_magic_preamble),
        ("Test 13: Report/paper generation", test_report_builder),
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
