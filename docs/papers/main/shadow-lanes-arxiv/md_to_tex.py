#!/usr/bin/env python3
"""Convert docs/papers/main/preprint.md to shadow-lanes-arxiv/main.tex (arXiv article)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREPRINT = ROOT.parent / "preprint.md"
OUT_TEX = ROOT / "main.tex"

CITE_KEYS = [
    "jcgm2008gum",
    "enzyme2021",
    "uncertaint2014",
    "uncertainties",
    "measurementsjl",
    "uncertainties_issue57",
    "measurements_issue25",
    "streamkpp2024",
    "kernelfoundry2026",
    "triton2019",
    "pyro2019",
    "stan2017",
    "blanchard2020",
    "fasi2021",
    "gtc",
    "puffin2021",
    "lithos2025",
    "jcgm101",
    "gbeesgpu2025",
    "khattak2025",
]


def strip_docs_meta(text: str) -> str:
    if text.startswith("<!--"):
        end = text.find("-->")
        if end != -1:
            text = text[end + 3 :].lstrip("\n")
    return text


def cite_replace(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        if 1 <= idx <= len(CITE_KEYS):
            return f"\\cite{{{CITE_KEYS[idx - 1]}}}"
        return m.group(0)

    return re.sub(r"\[(\d+)\]", repl, text)


def md_inline(text: str) -> str:
    text = cite_replace(text)
    text = text.replace("Uncertain\\<T\\>", r"Uncertain$\langle T \rangle$")
    text = text.replace("`Knowledge<T>`", r"\texttt{Knowledge}$\langle T \rangle$")
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", text)
    text = text.replace("—", "---")
    text = text.replace("×", r"$\times$")
    text = text.replace("≤", r"$\leq$")
    text = text.replace("≥", r"$\geq$")
    text = text.replace("≠", r"$\neq$")
    text = text.replace("→", r"$\rightarrow$")
    return text


def section_title(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^\d+\.\s*", "", raw)
    return md_inline(raw)


def convert_table(lines: list[str]) -> list[str]:
    rows = [ln.strip() for ln in lines if ln.strip()]
    if len(rows) < 2:
        return ["\\begin{verbatim}", *lines, "\\end{verbatim}"]
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    body = []
    for row in rows[2:]:
        if set(row.replace("|", "").replace("-", "").strip()) == set():
            continue
        body.append([c.strip() for c in row.strip("|").split("|")])
    colspec = "l" * len(header)
    out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    out.append(" & ".join(md_inline(h) for h in header) + r" \\")
    out.append("\\midrule")
    for row in body:
        out.append(" & ".join(md_inline(c) for c in row) + r" \\")
    out.extend(["\\bottomrule", "\\end{tabular}"])
    return out


def extract_display_math(line: str) -> tuple[str | None, str]:
    s = line.strip()
    if not s.startswith("$$"):
        return None, line
    if s.endswith("$$") and len(s) > 4:
        inner = s[2:-2].strip()
        return inner, ""
    return "", line


def convert(md: str) -> str:
    md = strip_docs_meta(md)
    lines = md.splitlines()
    body: list[str] = []
    i = 0
    abstract = ""
    title = "Epistemic Shadow Lanes"

    while i < len(lines):
        line = lines[i]

        if line.startswith("# "):
            title = line[2:].strip()
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue

        if line.startswith("## Abstract"):
            i += 1
            abs_lines = []
            while i < len(lines) and not lines[i].startswith("## "):
                if lines[i].strip() and lines[i].strip() != "---":
                    abs_lines.append(md_inline(lines[i].strip()))
                i += 1
            abstract = " ".join(abs_lines)
            continue

        if line.startswith("## References"):
            break

        if line.strip() == "---":
            i += 1
            continue

        if line.startswith("## "):
            body.append(f"\\section{{{section_title(line[3:])}}}")
            i += 1
            continue

        if line.startswith("### "):
            body.append(f"\\subsection{{{section_title(line[4:])}}}")
            i += 1
            continue

        if line.strip().startswith("$$"):
            inner, _ = extract_display_math(line)
            if inner is not None and inner != "":
                body.append(f"\\[{inner}\\]")
                i += 1
                continue
            i += 1
            math = []
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                math.append(lines[i])
                i += 1
            i += 1
            body.append("\\[" + " ".join(math) + "\\]")
            continue

        if line.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            body.append("\\begin{lstlisting}[basicstyle=\\ttfamily\\small]")
            body.extend(block)
            body.append("\\end{lstlisting}")
            continue

        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            body.append("\\begin{table}[t]\\centering")
            body.extend(convert_table(table_lines))
            body.append("\\end{table}")
            continue

        if line.strip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"\\item {md_inline(lines[i].strip()[2:])}")
                i += 1
            body.append("\\begin{itemize}")
            body.extend(items)
            body.append("\\end{itemize}")
            continue

        if not line.strip():
            body.append("")
            i += 1
            continue

        body.append(md_inline(line))
        i += 1

    preamble = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{url}
\usepackage{listings}
\lstset{basicstyle=\ttfamily\small,breaklines=true}

\title{""" + title + r"""}
\author{Demetrios C. Agourakis\\
Biomaterials and Regenerative Medicine Post-Graduate Program,\\
Pontif\'{i}cia Universidade Cat\'{o}lica de S\~{a}o Paulo (PUC-SP), Brazil\\
Faculdade S\~{a}o Leopoldo Mandic, Campinas, Brazil\\
ORCID: 0009-0001-8671-8878}
\date{July 2026}

\begin{document}
\maketitle
\begin{abstract}
""" + abstract + r"""
\end{abstract}

"""
    postamble = r"""
\bibliographystyle{plain}
\bibliography{refs}
\end{document}
"""
    return preamble + "\n".join(body) + postamble


def main() -> int:
    src = PREPRINT
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    text = src.read_text(encoding="utf-8")
    OUT_TEX.write_text(convert(text), encoding="utf-8")
    print(f"Wrote {OUT_TEX} ({OUT_TEX.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())