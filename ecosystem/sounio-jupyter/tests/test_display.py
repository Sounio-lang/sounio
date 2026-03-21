"""Tests for display module."""

import sys
import os
import importlib.util

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import only the display module directly to avoid ipykernel dependency
spec = importlib.util.spec_from_file_location(
    "display",
    os.path.join(os.path.dirname(__file__), '..', 'sounio_kernel', 'display.py')
)
display_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(display_module)
format_knowledge_html = display_module.format_knowledge_html
format_knowledge_text = display_module.format_knowledge_text


def test_format_knowledge_html_high_confidence():
    """Test HTML formatting for high confidence values."""
    html = format_knowledge_html(100.0, 0.95, "measurement")
    assert "100" in html
    assert "#2ecc71" in html  # Green color
    assert "95%" in html
    assert "measurement" in html


def test_format_knowledge_html_medium_confidence():
    """Test HTML formatting for medium confidence values."""
    html = format_knowledge_html(50.0, 0.75, "estimation")
    assert "50" in html
    assert "#f39c12" in html  # Orange color
    assert "75%" in html


def test_format_knowledge_html_low_confidence():
    """Test HTML formatting for low confidence values."""
    html = format_knowledge_html(25.0, 0.6, "guess")
    assert "25" in html
    assert "#e74c3c" in html  # Red color
    assert "60%" in html


def test_format_knowledge_text():
    """Test plain text formatting."""
    text = format_knowledge_text(100.0, 0.95, "measurement")
    assert "100" in text
    assert "95%" in text
    assert "measurement" in text


def test_html_contains_required_elements():
    """Test that HTML output contains required semantic elements."""
    html = format_knowledge_html(42.0, 0.88, "calibration")
    assert "<div" in html
    assert "epistemic-value" in html
    assert "Value:" in html
    assert "Confidence:" in html
    assert "Provenance:" in html


if __name__ == "__main__":
    test_format_knowledge_html_high_confidence()
    test_format_knowledge_html_medium_confidence()
    test_format_knowledge_html_low_confidence()
    test_format_knowledge_text()
    test_html_contains_required_elements()
    print("All display tests passed!")
