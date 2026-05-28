"""Tests for CellExecutor."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import only the executor directly to avoid ipykernel dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "executor",
    os.path.join(os.path.dirname(__file__), '..', 'sounio_kernel', 'executor.py')
)
executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor_module)
CellExecutor = executor_module.CellExecutor


def test_executor_initialization():
    """Test that CellExecutor initializes correctly."""
    executor = CellExecutor()
    assert executor.souc_binary is not None, "souc binary should be found"
    assert executor.stdlib_path is not None, "stdlib path should be found"
    executor.cleanup()


def test_code_wrapping_expressions():
    """Test that expressions are wrapped in main function."""
    executor = CellExecutor()

    # Test simple expression
    wrapped = executor._wrap_code("let x = 1 + 1")
    assert "fn main()" in wrapped
    assert "let x = 1 + 1" in wrapped

    # Test variable declaration
    wrapped = executor._wrap_code("var y = 5")
    assert "fn main()" in wrapped

    executor.cleanup()


def test_code_wrapping_preserves_functions():
    """Test that function definitions are not wrapped."""
    executor = CellExecutor()

    code = """fn helper() -> i32 {
    42
}"""
    wrapped = executor._wrap_code(code)
    assert wrapped == code, "Functions should not be wrapped"

    executor.cleanup()


def test_code_wrapping_preserves_type_defs():
    """Test that type definitions are not wrapped."""
    executor = CellExecutor()

    code = "type MyInt = i32"
    wrapped = executor._wrap_code(code)
    assert wrapped == code, "Type definitions should not be wrapped"

    executor.cleanup()


def test_session_persistence():
    """Test that declarations accumulate across run_cell calls."""
    executor = CellExecutor()

    # First cell: a declaration
    out1, err1, rc1 = executor.run_cell("fn add(x: i64, y: i64) -> i64 { x + y }")
    assert rc1 == 0, f"Declaration should compile: {err1}"
    assert len(executor._declarations) == 1

    # Second cell: another declaration
    out2, err2, rc2 = executor.run_cell("fn sub(x: i64, y: i64) -> i64 { x - y }")
    assert rc2 == 0, f"Second declaration should compile: {err2}"
    assert len(executor._declarations) == 2

    # Session source should contain both
    src = executor.get_session_source()
    assert "fn add" in src
    assert "fn sub" in src

    executor.cleanup()


def test_expression_uses_session():
    """Test that expressions can reference previously declared functions."""
    executor = CellExecutor()

    # Declare a function
    out1, err1, rc1 = executor.run_cell("fn double(x: i64) -> i64 { x * 2 }")
    assert rc1 == 0, f"Declaration failed: {err1}"

    # Evaluate an expression using the function
    out2, err2, rc2 = executor.run_cell("double(21)")
    assert rc2 == 0, f"Expression failed: {err2}"
    assert out2.strip() == "42", f"Expected 42, got {out2!r}"

    executor.cleanup()


def test_reset_session():
    """Test that reset_session clears accumulated declarations."""
    executor = CellExecutor()

    executor.run_cell("fn id(x: i64) -> i64 { x }")
    assert len(executor._declarations) == 1

    executor.reset_session()
    assert len(executor._declarations) == 0
    assert executor.get_session_source() == ""

    executor.cleanup()


if __name__ == "__main__":
    test_executor_initialization()
    test_code_wrapping_expressions()
    test_code_wrapping_preserves_functions()
    test_code_wrapping_preserves_type_defs()
    test_session_persistence()
    test_expression_uses_session()
    test_reset_session()
    print("All executor tests passed!")
