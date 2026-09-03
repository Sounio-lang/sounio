"""Tests for the ReportBuilder module.

Run with:  pytest tests/test_report.py -v
"""

import sys
import os
import tempfile


from sounio.report import ReportBuilder
from sounio.knowledge import Knowledge
from sounio.types import PipelineResult, SimulationResult


# ---------------------------------------------------------------------------
# Basic report construction
# ---------------------------------------------------------------------------


def test_empty_report():
    """Test creating an empty report."""
    rb = ReportBuilder("Test Report", author="Test Author")
    assert rb.title == "Test Report"
    assert rb.author == "Test Author"
    assert len(rb.sections) == 0


def test_report_with_text_section():
    """Test adding text sections."""
    rb = ReportBuilder("Test Report")
    rb.add_section("Introduction", "This is an introduction.")
    rb.add_section("Methods", "This describes the methods.")
    assert len(rb.sections) == 2
    assert rb.sections[0].heading == "Introduction"
    assert rb.sections[0].content == "This is an introduction."


def test_method_chaining():
    """Test that add_* methods return self for chaining."""
    rb = ReportBuilder("Test")
    result = rb.add_section("S1", "content").add_section("S2", "more")
    assert result is rb
    assert len(rb.sections) == 2


# ---------------------------------------------------------------------------
# Knowledge tables
# ---------------------------------------------------------------------------


def test_knowledge_table_markdown():
    """Test Knowledge table rendering in Markdown."""
    data = {
        "Half-life": Knowledge(4.62, 0.767, "pk_fit"),
        "Clearance": Knowledge(12.5, 1.5, "pk_fit"),
    }
    rb = ReportBuilder("Report")
    rb.add_knowledge_table("PK Parameters", data)

    md = rb.to_markdown()
    assert "PK Parameters" in md
    assert "Half-life" in md
    assert "Clearance" in md
    assert "| Parameter | Value | Uncertainty | Rel. Unc. | Provenance |" in md
    assert "4.62" in md
    assert "12.5" in md
    assert "pk_fit" in md


def test_knowledge_table_empty():
    """Test empty Knowledge table."""
    rb = ReportBuilder("Report")
    rb.add_knowledge_table("Empty", {})
    md = rb.to_markdown()
    assert "Empty" in md
    assert "(no data)" in md


def test_knowledge_table_relative_uncertainty():
    """Test that relative uncertainty is calculated correctly."""
    data = {
        "Measurement": Knowledge(100.0, 5.0, "source"),  # 5% rel unc
    }
    rb = ReportBuilder("Report")
    rb.add_knowledge_table("Test", data)
    md = rb.to_markdown()
    # Should show 5.00%
    assert "5.00%" in md


def test_knowledge_table_zero_value():
    """Test Knowledge with zero value (rel unc = 0)."""
    data = {
        "Zero": Knowledge(0.0, 1.0, "source"),  # rel unc = 0
    }
    rb = ReportBuilder("Report")
    rb.add_knowledge_table("Test", data)
    md = rb.to_markdown()
    assert "0.00%" in md


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def test_markdown_header():
    """Test that Markdown includes title, author, date."""
    rb = ReportBuilder(
        "My Report",
        author="John Doe",
        date="2026-03-18",
    )
    md = rb.to_markdown()
    assert "# My Report" in md
    assert "John Doe" in md
    assert "2026-03-18" in md
    assert "---" in md


def test_markdown_without_author():
    """Test Markdown generation without author."""
    rb = ReportBuilder("My Report", date="2026-03-18")
    md = rb.to_markdown()
    assert "# My Report" in md
    assert "2026-03-18" in md
    assert "**Author:**" not in md


def test_markdown_multiple_sections():
    """Test Markdown with multiple sections."""
    rb = ReportBuilder("Report")
    rb.add_section("Intro", "Introduction text.")
    rb.add_section("Methods", "Methods text.")
    rb.add_section("Results", "Results text.")

    md = rb.to_markdown()
    assert "## Intro" in md
    assert "## Methods" in md
    assert "## Results" in md
    assert "Introduction text." in md
    assert "Methods text." in md
    assert "Results text." in md


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------


def test_latex_header():
    """Test LaTeX includes document structure."""
    rb = ReportBuilder("Test Report", author="Author", date="2026-03-18")
    latex = rb.to_latex()
    assert r"\documentclass" in latex
    assert r"\usepackage{booktabs}" in latex
    assert r"\title{Test Report}" in latex
    assert r"\author{Author}" in latex
    assert r"\date{2026-03-18}" in latex
    assert r"\begin{document}" in latex
    assert r"\maketitle" in latex
    assert r"\end{document}" in latex


def test_latex_section_heading():
    """Test LaTeX renders section headings."""
    rb = ReportBuilder("Report")
    rb.add_section("Introduction", "Intro text.")
    latex = rb.to_latex()
    assert r"\section{Introduction}" in latex


def test_latex_without_author():
    """Test LaTeX generation without author."""
    rb = ReportBuilder("Report", date="2026-03-18")
    latex = rb.to_latex()
    assert r"\title{Report}" in latex
    # Author should not be present if not provided
    assert r"\author{" not in latex or r"\author{}" not in latex


# ---------------------------------------------------------------------------
# Pipeline summary
# ---------------------------------------------------------------------------


def test_pipeline_summary_basic():
    """Test adding PipelineResult summary."""
    result = PipelineResult(
        molecules_screened=100,
        molecules_passed=25,
        pk_fitted=15,
        exit_code=0,
    )
    rb = ReportBuilder("Report")
    rb.add_pipeline_summary(result)

    md = rb.to_markdown()
    assert "Pipeline Summary" in md
    assert "100" in md
    assert "25" in md
    assert "15" in md
    assert "Exit code: 0" in md


def test_pipeline_summary_with_simulation():
    """Test pipeline summary with simulation results."""
    sim = SimulationResult(
        efficacy_rate=Knowledge(0.85, 0.05, "simulation"),
        adverse_rate=Knowledge(0.10, 0.02, "simulation"),
        therapeutic_index=Knowledge(8.5, 1.2, "derived"),
        confidence=0.95,
        n_patients=1000,
    )
    result = PipelineResult(
        molecules_screened=100,
        molecules_passed=25,
        pk_fitted=15,
        simulation=sim,
        exit_code=0,
    )
    rb = ReportBuilder("Report")
    rb.add_pipeline_summary(result)

    md = rb.to_markdown()
    assert "Simulation Results" in md
    assert "0.85" in md  # efficacy rate
    assert "0.10" in md  # adverse rate
    assert "8.5" in md   # therapeutic index
    assert "1000" in md  # n_patients
    assert "95.00%" in md  # confidence


def test_pipeline_summary_with_provenance():
    """Test pipeline summary includes provenance chain."""
    result = PipelineResult(
        molecules_screened=50,
        molecules_passed=10,
        pk_fitted=5,
        exit_code=0,
        provenance_chain=["screening", "pk_fitting", "simulation"],
    )
    rb = ReportBuilder("Report")
    rb.add_pipeline_summary(result)

    md = rb.to_markdown()
    assert "Provenance Chain" in md
    assert "screening" in md
    assert "pk_fitting" in md
    assert "simulation" in md
    assert "→" in md


# ---------------------------------------------------------------------------
# Figure references
# ---------------------------------------------------------------------------


def test_add_figure():
    """Test adding figure reference."""
    rb = ReportBuilder("Report")
    rb.add_figure("Concentration-Time Profile", "figures/ct_profile.png", "PK profile")
    md = rb.to_markdown()
    assert "## Concentration-Time Profile" in md
    assert "![Concentration-Time Profile](figures/ct_profile.png)" in md
    assert "PK profile" in md


def test_add_figure_without_caption():
    """Test figure without caption."""
    rb = ReportBuilder("Report")
    rb.add_figure("Chart", "chart.png")
    md = rb.to_markdown()
    assert "![Chart](chart.png)" in md


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def test_save_markdown():
    """Test saving report as Markdown."""
    rb = ReportBuilder("Test", date="2026-03-18")
    rb.add_section("Intro", "Content.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        path = f.name

    try:
        rb.save(path, format="markdown")
        with open(path, "r") as f:
            content = f.read()
        assert "# Test" in content
        assert "## Intro" in content
        assert "Content." in content
    finally:
        os.unlink(path)


def test_save_latex():
    """Test saving report as LaTeX."""
    rb = ReportBuilder("Test Report", date="2026-03-18")
    rb.add_section("Methods", "Method description.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
        path = f.name

    try:
        rb.save(path, format="latex")
        with open(path, "r") as f:
            content = f.read()
        assert r"\documentclass" in content
        assert r"\section{Methods}" in content
    finally:
        os.unlink(path)


def test_save_invalid_format():
    """Test that invalid format raises ValueError."""
    rb = ReportBuilder("Test")

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        path = f.name

    try:
        try:
            rb.save(path, format="invalid")
            assert False, "should have raised ValueError"
        except ValueError as e:
            assert "Unknown format" in str(e)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_complete_report():
    """Test a complete realistic report."""
    rb = ReportBuilder(
        "Drug Discovery Analysis",
        author="Research Team",
        date="2026-03-18",
    )

    rb.add_section(
        "Executive Summary",
        "This report summarizes screening and simulation results.",
    )

    rb.add_knowledge_table(
        "Pharmacokinetic Parameters",
        {
            "Half-life (h)": Knowledge(4.62, 0.767, "pk_fit"),
            "Clearance (L/h)": Knowledge(12.5, 1.5, "pk_fit"),
            "Volume of Distribution (L)": Knowledge(85.0, 5.0, "pk_fit"),
        },
    )

    sim = SimulationResult(
        efficacy_rate=Knowledge(0.82, 0.06, "simulation"),
        adverse_rate=Knowledge(0.12, 0.03, "simulation"),
        therapeutic_index=Knowledge(6.8, 1.1, "derived"),
        confidence=0.92,
        n_patients=5000,
    )
    result = PipelineResult(
        molecules_screened=150,
        molecules_passed=45,
        pk_fitted=28,
        simulation=sim,
        provenance_chain=["initial_screen", "pk_modeling", "simulation"],
        exit_code=0,
    )
    rb.add_pipeline_summary(result)

    rb.add_figure(
        "PK Profile",
        "figures/pk_profile.png",
        "Concentration vs. time with 95% uncertainty band",
    )

    md = rb.to_markdown()
    latex = rb.to_latex()

    # Markdown checks
    assert "# Drug Discovery Analysis" in md
    assert "Executive Summary" in md
    assert "Pharmacokinetic Parameters" in md
    assert "Pipeline Summary" in md
    assert "PK Profile" in md
    assert "4.62" in md
    assert "150" in md

    # LaTeX checks
    assert r"\documentclass" in latex
    assert r"\section{Executive Summary}" in latex
    assert r"\section{Pharmacokinetic Parameters}" in latex
