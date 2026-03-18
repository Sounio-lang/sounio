"""Report generation module for epistemic data and pipeline results.

Generates reproducible reports in Markdown and LaTeX formats with tables
of Knowledge values, uncertainty summaries, and provenance tracking.
"""

from __future__ import annotations

import datetime
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

from .knowledge import Knowledge
from .types import PipelineResult, SimulationResult


@dataclass
class _Section:
    """Internal representation of a report section."""
    heading: str
    content: str
    kind: str  # "text", "table", "figure", "summary"


class ReportBuilder:
    """Build reproducible reports from epistemic data.

    Parameters
    ----------
    title : str
        Report title.
    author : str, optional
        Author name (default: "").
    date : str, optional
        Report date in ISO format (default: today's date).

    Examples
    --------
    >>> rb = ReportBuilder("Drug Discovery Report", author="Dr. Smith")
    >>> rb.add_section("Introduction", "This is a drug discovery study.")
    >>> rb.add_knowledge_table("PK Parameters", {
    ...     "Half-life": Knowledge(4.62, 0.767, "pk_fit"),
    ...     "Clearance": Knowledge(12.5, 1.5, "pk_fit")
    ... })
    >>> print(rb.to_markdown())
    """

    def __init__(
        self,
        title: str,
        author: str = "",
        date: str = "",
    ) -> None:
        self.title = title
        self.author = author
        self.date = date or datetime.date.today().isoformat()
        self.sections: List[_Section] = []

    def add_section(self, heading: str, content: str) -> ReportBuilder:
        """Add a text section.

        Parameters
        ----------
        heading : str
            Section heading.
        content : str
            Text content.

        Returns
        -------
        ReportBuilder
            Self for method chaining.
        """
        self.sections.append(_Section(heading, content, "text"))
        return self

    def add_knowledge_table(
        self,
        heading: str,
        data: Dict[str, Knowledge],
    ) -> ReportBuilder:
        """Add a table of Knowledge values with uncertainty.

        Parameters
        ----------
        heading : str
            Table heading.
        data : Dict[str, Knowledge]
            Mapping of parameter names to Knowledge values.

        Returns
        -------
        ReportBuilder
            Self for method chaining.

        Examples
        --------
        >>> rb = ReportBuilder("Report")
        >>> rb.add_knowledge_table("Parameters", {
        ...     "Half-life": Knowledge(4.62, 0.767, "pk_fit"),
        ... })
        """
        content = self._format_knowledge_table_markdown(data)
        self.sections.append(_Section(heading, content, "table"))
        return self

    def add_figure(
        self,
        heading: str,
        fig_path: str,
        caption: str = "",
    ) -> ReportBuilder:
        """Reference a saved figure.

        Parameters
        ----------
        heading : str
            Figure section heading.
        fig_path : str
            Path to figure file (relative or absolute).
        caption : str, optional
            Figure caption (default: "").

        Returns
        -------
        ReportBuilder
            Self for method chaining.
        """
        content = f"![{heading}]({fig_path})"
        if caption:
            content += f"\n\n{caption}"
        self.sections.append(_Section(heading, content, "figure"))
        return self

    def add_pipeline_summary(self, result: PipelineResult) -> ReportBuilder:
        """Add PipelineResult summary.

        Parameters
        ----------
        result : PipelineResult
            Pipeline execution result.

        Returns
        -------
        ReportBuilder
            Self for method chaining.
        """
        lines = [
            f"**Execution Summary**\n",
            f"- Molecules screened: {result.molecules_screened}",
            f"- Molecules passed: {result.molecules_passed}",
            f"- PK fitted: {result.pk_fitted}",
            f"- Exit code: {result.exit_code}",
        ]

        if result.simulation:
            sim = result.simulation
            lines.extend([
                f"\n**Simulation Results**\n",
                f"- Efficacy rate: {sim.efficacy_rate.value:.4f} ± {sim.efficacy_rate.epsilon:.4f}",
                f"- Adverse rate: {sim.adverse_rate.value:.4f} ± {sim.adverse_rate.epsilon:.4f}",
                f"- Therapeutic index: {sim.therapeutic_index.value:.2f}",
                f"- Virtual patients: {sim.n_patients}",
                f"- Confidence: {sim.confidence:.2%}",
            ])

        if result.provenance_chain:
            lines.append(f"\n**Provenance Chain**\n")
            lines.append(f"```\n{' → '.join(result.provenance_chain)}\n```")

        content = "\n".join(lines)
        self.sections.append(_Section("Pipeline Summary", content, "summary"))
        return self

    def add_epistemic_summary(self, edf: Any) -> ReportBuilder:
        """Add EpistemicDataFrame summary statistics.

        Parameters
        ----------
        edf : EpistemicDataFrame
            DataFrame with Knowledge-valued columns.

        Returns
        -------
        ReportBuilder
            Self for method chaining.

        Notes
        -----
        This is a placeholder for future integration with pandas-based
        epistemic dataframes. Currently accepts any object with a summary()
        method or dict-like interface.
        """
        if hasattr(edf, "summary"):
            content = str(edf.summary())
        elif isinstance(edf, dict):
            content = self._format_knowledge_table_markdown(edf)
        else:
            content = str(edf)

        self.sections.append(_Section("Epistemic Summary", content, "summary"))
        return self

    def to_markdown(self) -> str:
        """Generate full Markdown report.

        Returns
        -------
        str
            Complete report in Markdown format.
        """
        lines = [
            f"# {self.title}",
            "",
        ]

        if self.author:
            lines.append(f"**Author:** {self.author}  ")

        lines.extend([
            f"**Date:** {self.date}",
            "",
            "---",
            "",
        ])

        for section in self.sections:
            lines.append(f"## {section.heading}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
            lines.append("")

        return "\n".join(lines)

    def to_latex(self) -> str:
        """Generate LaTeX article with booktabs tables.

        Returns
        -------
        str
            Complete report in LaTeX format.
        """
        lines = [
            r"\documentclass[11pt]{article}",
            r"\usepackage[utf-8]{inputenc}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            "",
            r"\title{" + self.title + r"}",
        ]

        if self.author:
            lines.append(r"\author{" + self.author + r"}")

        lines.extend([
            r"\date{" + self.date + r"}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
        ])

        for section in self.sections:
            lines.append(r"\section{" + section.heading + r"}")
            lines.append("")

            if section.kind == "table":
                lines.append(section.content)
            elif section.kind == "figure":
                # Extract ![alt](path) format
                import re
                match = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", section.content)
                if match:
                    alt, path = match.groups()
                    lines.append(r"\begin{figure}[h]")
                    lines.append(r"  \centering")
                    lines.append(rf"  \includegraphics[width=0.7\textwidth]{{{path}}}")
                    lines.append(rf"  \caption{{{alt}}}")
                    lines.append(r"\end{figure}")
            else:
                # Text content: convert ** to \textbf, etc.
                text = section.content
                text = text.replace(r"**", r"\textbf{")
                text = text.replace(r"**", r"}")  # Note: naive, assumes pairs
                lines.append(text)

            lines.append("")

        lines.extend([
            r"\end{document}",
        ])

        return "\n".join(lines)

    def save(self, path: str, format: str = "markdown") -> None:
        """Save report to file.

        Parameters
        ----------
        path : str
            Output file path.
        format : str, optional
            Output format: "markdown" (default) or "latex".

        Raises
        ------
        ValueError
            If format is not "markdown" or "latex".
        """
        if format == "markdown":
            content = self.to_markdown()
        elif format == "latex":
            content = self.to_latex()
        else:
            raise ValueError(f"Unknown format: {format}")

        with open(path, "w") as f:
            f.write(content)

    # ---- Helpers ----

    @staticmethod
    def _format_knowledge_table_markdown(data: Dict[str, Knowledge]) -> str:
        """Format Knowledge dictionary as Markdown table.

        Table columns: Parameter | Value | Uncertainty | Rel. Unc. | Provenance
        """
        if not data:
            return "(no data)"

        lines = [
            "| Parameter | Value | Uncertainty | Rel. Unc. | Provenance |",
            "|-----------|-------|-------------|-----------|------------|",
        ]

        for name, k in data.items():
            rel_unc_pct = k.relative_uncertainty * 100
            lines.append(
                f"| {name} | {k.value:.6g} | ± {k.epsilon:.6g} | "
                f"{rel_unc_pct:.2f}% | {k.provenance} |"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_knowledge_table_latex(data: Dict[str, Knowledge]) -> str:
        """Format Knowledge dictionary as LaTeX booktabs table."""
        if not data:
            return "% (no data)"

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{llrrrl}",
            r"\toprule",
            r"Parameter & Value & Uncertainty & Rel. Unc. & Provenance \\",
            r"\midrule",
        ]

        for name, k in data.items():
            rel_unc_pct = k.relative_uncertainty * 100
            lines.append(
                f"{name} & {k.value:.6g} & ±{k.epsilon:.6g} & "
                f"{rel_unc_pct:.2f}\\% & {k.provenance} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)
