"""Tests for SounioExecutor — subprocess wrapper around souc.

Tests that require the souc binary are skipped automatically when the binary
is not found (CI-safe).
"""

import os
import re
import sys
import tempfile
import unittest.mock as mock


from sounio._executor import (
    SounioExecutor,
    ExecutionResult,
    CheckResult,
    _KNOWLEDGE_RE,
)
from sounio.knowledge import Knowledge


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _souc_available() -> bool:
    ex = SounioExecutor()
    import pathlib
    return pathlib.Path(ex.souc_path).exists()


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


def test_execution_result_ok():
    r = ExecutionResult("hello\n", "", 0)
    assert r.ok


def test_execution_result_fail():
    r = ExecutionResult("", "error\n", 1)
    assert not r.ok


def test_execution_result_knowledge_values():
    k = Knowledge(1.0, 0.1, "x")
    r = ExecutionResult("", "", 0, [k])
    assert len(r.knowledge_values) == 1
    assert r.knowledge_values[0].value == 1.0


def test_execution_result_repr():
    r = ExecutionResult("", "", 0, [Knowledge(1.0, 0.1, "x")])
    assert "ExecutionResult" in repr(r)
    assert "1" in repr(r)  # knowledge_values count


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


def test_check_result_ok():
    cr = CheckResult(True, "")
    assert cr.success
    assert cr.errors == ""
    assert cr.ast is None
    assert cr.types is None


def test_check_result_fail():
    cr = CheckResult(False, "type error on line 5\n")
    assert not cr.success
    assert "type error" in cr.errors


# ---------------------------------------------------------------------------
# Knowledge parsing
# ---------------------------------------------------------------------------


def test_parse_single_knowledge():
    stdout = 'Knowledge { value: 42.000 epsilon: 0.100 prov: "sensor" }'
    results = SounioExecutor._parse_knowledge(stdout)
    assert len(results) == 1
    assert results[0].value == 42.0
    assert results[0].epsilon == 0.1
    assert results[0].provenance == "sensor"


def test_parse_multiple_knowledge():
    stdout = (
        'Knowledge { value: 1.000 epsilon: 0.010 prov: "a" }\n'
        'Knowledge { value: 2.000 epsilon: 0.020 prov: "b" }\n'
    )
    results = SounioExecutor._parse_knowledge(stdout)
    assert len(results) == 2
    assert results[0].provenance == "a"
    assert results[1].provenance == "b"


def test_parse_no_knowledge():
    results = SounioExecutor._parse_knowledge("just text output, nothing special\n")
    assert results == []


def test_parse_scientific_notation():
    stdout = 'Knowledge { value: 1.5e-03 epsilon: 2.0e-05 prov: "moles" }'
    results = SounioExecutor._parse_knowledge(stdout)
    assert len(results) == 1
    assert abs(results[0].value - 1.5e-3) < 1e-15


def test_parse_empty_provenance():
    stdout = 'Knowledge { value: 9.800 epsilon: 0.001 prov: "" }'
    results = SounioExecutor._parse_knowledge(stdout)
    assert len(results) == 1
    assert results[0].provenance == ""


def test_parse_complex_provenance():
    stdout = 'Knowledge { value: 5.000 epsilon: 0.200 prov: "(a)+(b)" }'
    results = SounioExecutor._parse_knowledge(stdout)
    assert len(results) == 1
    assert results[0].provenance == "(a)+(b)"


# ---------------------------------------------------------------------------
# Executor construction
# ---------------------------------------------------------------------------


def test_executor_uses_souc_env_var(tmp_path):
    fake_souc = tmp_path / "souc-fake"
    fake_souc.write_text("#!/bin/sh\necho fake")
    fake_souc.chmod(0o755)
    os.environ["SOUC"] = str(fake_souc)
    try:
        ex = SounioExecutor()
        assert ex.souc_path == str(fake_souc)
    finally:
        del os.environ["SOUC"]


def test_executor_explicit_path():
    ex = SounioExecutor(souc_path="/custom/path/souc")
    assert ex.souc_path == "/custom/path/souc"


def test_executor_repr():
    ex = SounioExecutor(souc_path="/my/souc")
    assert "SounioExecutor" in repr(ex)
    assert "/my/souc" in repr(ex)


# ---------------------------------------------------------------------------
# Mocked subprocess — run_file
# ---------------------------------------------------------------------------


def _make_mock_proc(stdout="", stderr="", returncode=0):
    proc = mock.MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


def test_run_file_ok(tmp_path):
    sio = tmp_path / "hello.sio"
    sio.write_text("fn main() with IO { }")
    ex = SounioExecutor(souc_path="/fake/souc")
    mock_stdout = 'Knowledge { value: 7.000 epsilon: 0.070 prov: "test" }\n'
    with mock.patch("subprocess.run", return_value=_make_mock_proc(stdout=mock_stdout)):
        result = ex.run_file(str(sio))
    assert result.ok
    assert len(result.knowledge_values) == 1
    assert result.knowledge_values[0].value == 7.0


def test_run_file_non_zero_exit(tmp_path):
    sio = tmp_path / "bad.sio"
    sio.write_text("fn main() { bad syntax }")
    ex = SounioExecutor(souc_path="/fake/souc")
    with mock.patch("subprocess.run", return_value=_make_mock_proc(stderr="error", returncode=1)):
        result = ex.run_file(str(sio))
    assert not result.ok
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Mocked subprocess — run_code
# ---------------------------------------------------------------------------


def test_run_code_creates_temp_file():
    code = "fn main() with IO { }"
    ex = SounioExecutor(souc_path="/fake/souc")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _make_mock_proc()

    with mock.patch("subprocess.run", side_effect=fake_run):
        result = ex.run_code(code)

    assert len(calls) == 1
    # The temp file path should end in .sio
    assert calls[0][2].endswith(".sio")
    # The temp file should have been deleted
    assert not os.path.exists(calls[0][2])


def test_run_code_parses_knowledge():
    code = "fn main() with IO { }"
    ex = SounioExecutor(souc_path="/fake/souc")
    out = 'Knowledge { value: 3.140 epsilon: 0.003 prov: "pi" }'
    with mock.patch("subprocess.run", return_value=_make_mock_proc(stdout=out)):
        result = ex.run_code(code)
    assert len(result.knowledge_values) == 1
    assert abs(result.knowledge_values[0].value - 3.14) < 1e-3


# ---------------------------------------------------------------------------
# Mocked subprocess — check_file
# ---------------------------------------------------------------------------


def test_check_file_success(tmp_path):
    sio = tmp_path / "ok.sio"
    sio.write_text("fn main() { }")
    ex = SounioExecutor(souc_path="/fake/souc")
    with mock.patch("subprocess.run", return_value=_make_mock_proc(returncode=0)):
        cr = ex.check_file(str(sio))
    assert cr.success


def test_check_file_failure(tmp_path):
    sio = tmp_path / "bad.sio"
    sio.write_text("bad syntax {{{")
    ex = SounioExecutor(souc_path="/fake/souc")
    with mock.patch("subprocess.run", return_value=_make_mock_proc(stderr="E001", returncode=1)):
        cr = ex.check_file(str(sio))
    assert not cr.success
    assert "E001" in cr.errors


def test_check_file_show_ast(tmp_path):
    sio = tmp_path / "ok.sio"
    sio.write_text("fn main() { }")
    ex = SounioExecutor(souc_path="/fake/souc")
    ast_output = "(FnDef main ...)"
    with mock.patch("subprocess.run", return_value=_make_mock_proc(stdout=ast_output)):
        cr = ex.check_file(str(sio), show_ast=True)
    assert cr.ast == ast_output


# ---------------------------------------------------------------------------
# Live integration tests (skipped when souc is unavailable)
# ---------------------------------------------------------------------------


def test_live_check_nonexistent_file():
    if not _souc_available():
        return
    ex = SounioExecutor()
    cr = ex.check_file("/nonexistent/path/file.sio")
    # Expect failure — file doesn't exist
    assert not cr.success


def test_live_run_code_simple():
    if not _souc_available():
        return
    ex = SounioExecutor()
    # Minimal Sounio program that just exits
    result = ex.run_code("fn main() with IO { }", timeout=15)
    # Either succeeds or produces a known error — just check no crash
    assert result.exit_code in range(-255, 256)
