"""Tests for autocompletion."""

import sys
import os
import importlib.util

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import completion module directly to avoid ipykernel dependency
spec = importlib.util.spec_from_file_location(
    "completion",
    os.path.join(os.path.dirname(__file__), "..", "sounio_kernel", "completion.py"),
)
completion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion_module)
SounioCompleter = completion_module.SounioCompleter


def test_completer_initialization():
    """Test SounioCompleter initializes correctly."""
    completer = SounioCompleter()
    assert completer.user_variables == {}
    assert completer.user_functions == {}


def test_complete_empty_input():
    """Test completion with empty input."""
    completer = SounioCompleter()
    matches = completer.complete("", 0)
    assert matches == []


def test_complete_keyword_let():
    """Test completion finds 'let' keyword."""
    completer = SounioCompleter()
    matches = completer.complete("le", 2)
    assert "let" in matches


def test_complete_keyword_var():
    """Test completion finds 'var' keyword."""
    completer = SounioCompleter()
    matches = completer.complete("va", 2)
    assert "var" in matches


def test_complete_keyword_fn():
    """Test completion finds 'fn' keyword."""
    completer = SounioCompleter()
    matches = completer.complete("f", 1)
    assert "fn" in matches or "for" in matches


def test_complete_keyword_struct():
    """Test completion finds 'struct' keyword."""
    completer = SounioCompleter()
    matches = completer.complete("stru", 4)
    assert "struct" in matches


def test_complete_keyword_match():
    """Test completion finds 'match' keyword."""
    completer = SounioCompleter()
    matches = completer.complete("mat", 3)
    assert "match" in matches


def test_complete_stdlib_print():
    """Test completion finds 'print' function."""
    completer = SounioCompleter()
    matches = completer.complete("pri", 3)
    assert "print" in matches or "println" in matches


def test_complete_stdlib_sqrt():
    """Test completion finds 'sqrt' function."""
    completer = SounioCompleter()
    matches = completer.complete("sqr", 3)
    assert "sqrt" in matches


def test_complete_user_variable():
    """Test completion finds user-defined variables."""
    completer = SounioCompleter()
    completer.user_variables["my_var"] = True
    matches = completer.complete("my", 2)
    assert "my_var" in matches


def test_complete_user_function():
    """Test completion finds user-defined functions."""
    completer = SounioCompleter()
    completer.user_functions["helper_fn"] = True
    matches = completer.complete("help", 4)
    assert "helper_fn" in matches


def test_complete_returns_sorted():
    """Test completion returns sorted results."""
    completer = SounioCompleter()
    matches = completer.complete("", 0)  # This should return nothing for empty string
    # But when we test with actual matches, they should be sorted
    completer.user_variables["zebra"] = True
    completer.user_variables["apple"] = True
    matches = completer.complete("", 0)  # Still empty since no prefix
    # Test with a prefix that matches both
    all_matches = completer.complete("a", 1)
    if all_matches:
        assert all_matches == sorted(all_matches)


def test_complete_no_duplicates():
    """Test completion removes duplicates."""
    completer = SounioCompleter()
    completer.user_variables["test"] = True
    # Even if 'test' appears in keywords, it should only appear once
    matches = completer.complete("te", 2)
    assert matches == list(set(matches))  # No duplicates


def test_extract_let_definition():
    """Test extracting let variable definitions."""
    completer = SounioCompleter()
    code = "let my_var = 5"
    completer.extract_definitions(code)
    assert "my_var" in completer.user_variables


def test_extract_var_definition():
    """Test extracting var variable definitions."""
    completer = SounioCompleter()
    code = "var mutable_var = 10"
    completer.extract_definitions(code)
    assert "mutable_var" in completer.user_variables


def test_extract_function_definition():
    """Test extracting function definitions."""
    completer = SounioCompleter()
    code = "fn my_function() { 1 }"
    completer.extract_definitions(code)
    assert "my_function" in completer.user_functions


def test_extract_multiple_definitions():
    """Test extracting multiple definitions from code."""
    completer = SounioCompleter()
    code = """
    let x = 1
    var y = 2
    fn foo() { x + y }
    fn bar() { x - y }
    """
    completer.extract_definitions(code)
    assert "x" in completer.user_variables
    assert "y" in completer.user_variables
    assert "foo" in completer.user_functions
    assert "bar" in completer.user_functions


def test_extract_word_at_cursor():
    """Test extracting current word from text."""
    completer = SounioCompleter()
    text = "let my_variable = "
    word = completer._extract_current_word(text)
    assert word == ""  # Should be empty since we're at whitespace


def test_extract_word_mid_word():
    """Test extracting word when cursor is mid-word."""
    completer = SounioCompleter()
    text = "let my_var"
    word = completer._extract_current_word(text)
    assert word == "my_var"


def test_extract_word_with_underscore():
    """Test extracting word with underscores."""
    completer = SounioCompleter()
    text = "let __private"
    word = completer._extract_current_word(text)
    assert word == "__private"


def test_complete_case_sensitive():
    """Test that completion is case-sensitive."""
    completer = SounioCompleter()
    matches = completer.complete("Let", 3)  # Capital L
    assert "let" not in matches  # Should not match lowercase keyword


def test_snippet_get_fn():
    """Test retrieving fn snippet."""
    completer = SounioCompleter()
    snippet = completer.get_snippet("fn")
    assert snippet is not None
    assert "${name}" in snippet


def test_snippet_get_let():
    """Test retrieving let snippet."""
    completer = SounioCompleter()
    snippet = completer.get_snippet("let")
    assert snippet is not None
    assert "${name}" in snippet


def test_snippet_get_struct():
    """Test retrieving struct snippet."""
    completer = SounioCompleter()
    snippet = completer.get_snippet("struct")
    assert snippet is not None
    assert "${field}" in snippet


def test_snippet_not_found():
    """Test getting non-existent snippet."""
    completer = SounioCompleter()
    snippet = completer.get_snippet("nonexistent_snippet")
    assert snippet is None


def test_complete_knowledge_keyword():
    """Test completion finds Knowledge keyword."""
    completer = SounioCompleter()
    matches = completer.complete("Know", 4)
    assert "Knowledge" in matches


def test_complete_io_effect():
    """Test completion finds IO effect."""
    completer = SounioCompleter()
    matches = completer.complete("I", 1)
    assert "IO" in matches or "if" in matches


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
