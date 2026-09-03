"""Tests for magic commands."""

import json
import sys
import os
import importlib.util
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import magics module directly to avoid ipykernel dependency
spec = importlib.util.spec_from_file_location(
    "magics",
    os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "magics.py"),
)
magics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(magics_module)
SounioMagics = magics_module.SounioMagics


def test_time_magic_with_valid_code():
    """Test %time magic with valid code."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.run_cell = Mock(return_value=("output", "", 0))
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_time("let x = 1 + 1")

    assert "CPU time:" in result
    assert "Output:" in result
    mock_executor.run_cell.assert_called_once()


def test_time_magic_empty_input():
    """Test %time magic with empty input."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_time("")

    assert "Usage:" in result


def test_timeit_magic_default_iterations():
    """Test %timeit magic with default iterations."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.run_cell = Mock(return_value=("", "", 0))
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_timeit("let x = 1")

    assert "Min:" in result
    assert "Max:" in result
    assert "Avg:" in result
    assert mock_executor.run_cell.call_count == 3  # Default 3 runs


def test_timeit_magic_custom_iterations():
    """Test %timeit magic with custom iteration count."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.run_cell = Mock(return_value=("", "", 0))
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_timeit("-n 5 let x = 1")

    assert "Min:" in result
    assert "Max:" in result
    assert "Avg:" in result
    assert mock_executor.run_cell.call_count == 5  # 5 runs


def test_writefile_magic_writes_file():
    """Test %%writefile magic writes content to file."""
    import tempfile

    mock_kernel = Mock()
    mock_executor = Mock()
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sio", delete=False) as f:
        temp_file = f.name

    try:
        cell_content = "fn main() { 1 + 1 }"
        result = magics.magic_writefile(temp_file, cell=cell_content)

        assert "Wrote" in result
        assert temp_file in result

        # Verify file was written
        with open(temp_file, "r") as f:
            content = f.read()
        assert content == cell_content
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_writefile_magic_no_filename():
    """Test %%writefile magic with no filename."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_writefile("")

    assert "Usage:" in result


def test_sounio_info_magic():
    """Test %sounio info magic."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.stdlib_path = "/test/stdlib"
    mock_executor.souc_binary = "/test/souc"
    mock_executor._declarations = []
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_sounio("info")

    assert "Sounio Kernel" in result
    assert "v0.2.0" in result
    assert "epistemic" in result.lower()


def test_sounio_stdlib_magic():
    """Test %sounio stdlib magic."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.stdlib_path = "/test/stdlib"
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_sounio("stdlib")

    assert "/test/stdlib" in result


def test_sounio_souc_magic():
    """Test %sounio souc magic."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.souc_binary = "/test/souc"
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_sounio("souc")

    assert "/test/souc" in result


def test_sounio_stdlib_not_found():
    """Test %sounio stdlib when not found."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.stdlib_path = None
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_sounio("stdlib")

    assert "not found" in result


def test_sounio_souc_not_found():
    """Test %sounio souc when not found."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.souc_binary = None
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_sounio("souc")

    assert "not found" in result


def test_sounio_help_magic():
    """Test %sounio with no arguments defaults to info."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor._declarations = []
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    result = magics.magic_sounio("")

    # When called with empty args, it defaults to "info" command
    assert "Sounio Kernel" in result


def test_handle_magic_dispatch_valid():
    """Test handle_magic dispatches to correct handler."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.stdlib_path = "/stdlib"
    mock_executor.souc_binary = "/souc"
    mock_executor._declarations = []
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    handled, result = magics.handle_magic("%sounio info")

    assert handled is True
    assert "Sounio" in result


def test_handle_magic_dispatch_invalid():
    """Test handle_magic with invalid magic name."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    handled, result = magics.handle_magic("%invalid_magic")

    assert handled is False


def test_handle_magic_time():
    """Test handle_magic dispatches to time."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.run_cell = Mock(return_value=("", "", 0))
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    handled, result = magics.handle_magic("%time let x = 1")

    assert handled is True
    assert "CPU time:" in result


def test_reset_magic():
    """Test %reset magic clears the session."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_executor.reset_session = Mock()
    mock_kernel.executor = mock_executor

    magics = SounioMagics(mock_kernel)
    handled, result = magics.handle_magic("%reset")

    assert handled is True
    assert "reset" in result.lower()
    mock_executor.reset_session.assert_called_once()


def test_ontology_search_magic():
    """Test %ontology_search uses ontology API and formats JSON."""
    mock_kernel = Mock()
    mock_kernel.executor = Mock()
    magics = SounioMagics(mock_kernel)

    fake_term = Mock()
    fake_term.to_dict.return_value = {"curie": "HPO:0003074", "label": "Hyperglycemia"}
    fake_ontology = Mock()
    fake_ontology.search.return_value = [fake_term]

    with patch.object(magics, "_ontology_module", return_value=fake_ontology):
        result = magics.magic_ontology_search("hyperglycemia")

    payload = json.loads(result)
    assert payload[0]["curie"] == "HPO:0003074"
    fake_ontology.search.assert_called_once_with("hyperglycemia")


def test_ontology_resolve_magic():
    """Test %ontology_resolve renders canonical term JSON."""
    mock_kernel = Mock()
    mock_kernel.executor = Mock()
    magics = SounioMagics(mock_kernel)

    fake_term = Mock()
    fake_term.to_dict.return_value = {
        "curie": "SNOMED:44054006",
        "label": "Type 2 diabetes mellitus",
        "provenance": "local:snomed",
    }
    fake_ontology = Mock()
    fake_ontology.resolve.return_value = fake_term

    with patch.object(magics, "_ontology_module", return_value=fake_ontology):
        result = magics.magic_ontology_resolve("SNOMED:44054006")

    payload = json.loads(result)
    assert payload["curie"] == "SNOMED:44054006"
    assert payload["provenance"] == "local:snomed"


def test_clinical_normalize_magic_inline_json():
    """Test %clinical_normalize accepts inline JSON payloads."""
    mock_kernel = Mock()
    mock_kernel.executor = Mock()
    magics = SounioMagics(mock_kernel)

    fake_ontology = Mock()
    fake_ontology.clinical_normalize.return_value = {
        "patient_id": "pt-001",
        "provenance": "clinical_normalize[test]",
    }

    line = '{"patient_id":"pt-001","diagnoses":["SNOMED:44054006"]}'
    with patch.object(magics, "_ontology_module", return_value=fake_ontology):
        result = magics.magic_clinical_normalize(line)

    payload = json.loads(result)
    assert payload["patient_id"] == "pt-001"
    fake_ontology.clinical_normalize.assert_called_once()


def test_drug_pipeline_magic_format_output():
    """Test drug pipeline output formatting with HTML."""
    mock_kernel = Mock()
    mock_executor = Mock()
    mock_kernel.executor = mock_executor
    magics = SounioMagics(mock_kernel)

    # Test the output formatting helper
    stdout = "Test output\n"
    knowledge_values = []

    result = magics._format_drug_pipeline_output(stdout, knowledge_values)

    assert "drug-pipeline-output" in result
    assert "Test output" in result
    assert "div" in result


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
