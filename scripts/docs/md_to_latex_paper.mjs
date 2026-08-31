#!/usr/bin/env node
// scripts/docs/md_to_latex_paper.mjs — deterministic Markdown → LaTeX for the research papers.
//
//   node scripts/docs/md_to_latex_paper.mjs <paper.md> <out.tex> [--class acmart|article]
//
// Converts the restricted Markdown the paper drafts use (ATX headings, paragraphs, **bold**,
// *italic*, `code`, fenced code blocks, > blockquotes, - / 1. lists, pipe tables, --- rules,
// `$…$`-free inline math written as unicode, and the docs:meta / status-note HTML comments the
// governance sync injects) into a compilable LaTeX body. No LLM in the loop: the conversion is a
// pure function of the source, so the .tex is auditable against the .md line by line.
//
// It does NOT try to be a full CommonMark implementation. Anything it does not recognise is
// emitted as an escaped paragraph, never dropped.

import fs from 'node:fs';

const [,, inPath, outPath, ...rest] = process.argv;
if (!inPath || !outPath) {
  console.error('usage: md_to_latex_paper.mjs <paper.md> <out.tex> [--class acmart|article]');
  process.exit(2);
}
const cls = (rest.indexOf('--class') >= 0) ? rest[rest.indexOf('--class') + 1] : 'acmart';

let src = fs.readFileSync(inPath, 'utf8');
// strip governance frontmatter / status notes
src = src.replace(/<!--\s*docs:meta[\s\S]*?-->\s*/g, '')
         .replace(/<!--\s*docs:status-note:start\s*-->[\s\S]*?<!--\s*docs:status-note:end\s*-->\s*/g, '');

// ---------- inline ----------
const escapeTex = (s) => s
  .replace(/\\/g, '\\textbackslash{}')
  .replace(/([&%$#_{}])/g, '\\$1')
  .replace(/~/g, '\\textasciitilde{}')
  .replace(/\^/g, '\\textasciicircum{}');

// unicode that LaTeX (pdfTeX/tectonic with T1) chokes on → macros; keep the rest (xelatex-safe
// fallback is documented in the preamble: we compile with tectonic/xelatex so most unicode passes).
const uni = (s) => s
  .replace(/⟹/g, '$\\Rightarrow$').replace(/⟸/g, '$\\Leftarrow$').replace(/⟺/g, '$\\Leftrightarrow$')
  .replace(/→/g, '$\\rightarrow$').replace(/←/g, '$\\leftarrow$').replace(/↔/g, '$\\leftrightarrow$')
  .replace(/≤/g, '$\\leq$').replace(/≥/g, '$\\geq$').replace(/≠/g, '$\\neq$').replace(/≈/g, '$\\approx$')
  .replace(/∈/g, '$\\in$').replace(/∉/g, '$\\notin$').replace(/∪/g, '$\\cup$').replace(/∩/g, '$\\cap$')
  .replace(/⊆/g, '$\\subseteq$').replace(/⊑/g, '$\\sqsubseteq$').replace(/⊤/g, '$\\top$').replace(/∅/g, '$\\emptyset$')
  .replace(/∀/g, '$\\forall$').replace(/∃/g, '$\\exists$').replace(/⇒/g, '$\\Rightarrow$').replace(/⇒\*/g, '$\\Rightarrow^{*}$')
  .replace(/·/g, '$\\cdot$').replace(/×/g, '$\\times$').replace(/√/g, '$\\surd$').replace(/∞/g, '$\\infty$')
  .replace(/²/g, '$^{2}$').replace(/³/g, '$^{3}$').replace(/⁻/g, '$^{-}$').replace(/¹/g, '$^{1}$')
  .replace(/₀/g, '$_{0}$').replace(/₁/g, '$_{1}$').replace(/₂/g, '$_{2}$').replace(/₋/g, '$_{-}$').replace(/₂₄/g, '$_{24}$')
  .replace(/α/g, '$\\alpha$').replace(/β/g, '$\\beta$').replace(/ρ/g, '$\\rho$').replace(/σ/g, '$\\sigma$').replace(/ε/g, '$\\varepsilon$').replace(/Γ/g, '$\\Gamma$').replace(/τ/g, '$\\tau$').replace(/λ/g, '$\\lambda$').replace(/Σ/g, '$\\Sigma$').replace(/δ/g, '$\\delta$').replace(/μ/g, '$\\mu$')
  .replace(/⟨/g, '$\\langle$').replace(/⟩/g, '$\\rangle$').replace(/⊢/g, '$\\vdash$').replace(/‖/g, '$\\|$').replace(/ᵢ/g, '$_{i}$').replace(/ₐ/g, '$_{a}$').replace(/ₛ/g, '$_{s}$').replace(/ᵦ/g, '$_{b}$')
  .replace(/—/g, '---').replace(/–/g, '--').replace(/…/g, '\\ldots{}')
  .replace(/“|”/g, '"').replace(/‘|’/g, "'")
  .replace(/✅/g, '\\checkmark{}').replace(/◻/g, '$\\square$').replace(/✗/g, '$\\times$');

// collapse "$x$$y$" runs produced by adjacent replacements
// box-drawing (U+2500–U+257F) has no glyph in the listings font: rules become ASCII
const asciiBox = (s) => s.replace(/[─━═]/g, "-").replace(/[│┃║]/g, "|").replace(/[┌┐└┘├┤┬┴┼╭╮╰╯]/g, "+")
  .replace(/⊢/g, "|-").replace(/⟨/g, "<").replace(/⟩/g, ">").replace(/‖/g, "||").replace(/ᵢ/g, "_i").replace(/ₐ/g, "_a").replace(/₁/g, "_1").replace(/₂/g, "_2").replace(/₀/g, "_0")
  .replace(/ε/g, "eps").replace(/Γ/g, "G").replace(/∪/g, "U").replace(/∩/g, "^").replace(/⊤/g, "TOP").replace(/∅/g, "{}").replace(/≠/g, "!=").replace(/≤/g, "<=").replace(/≥/g, ">=").replace(/→/g, "->").replace(/⟹/g, "=>").replace(/⇒/g, "=>").replace(/⟺/g, "<=>")
  .replace(/∀/g, "forall ").replace(/∃/g, "exists ").replace(/ρ/g, "rho").replace(/σ/g, "sigma").replace(/α/g, "alpha").replace(/β/g, "beta").replace(/λ/g, "lambda").replace(/τ/g, "tau").replace(/·/g, "*").replace(/×/g, "x").replace(/√/g, "sqrt").replace(/²/g, "^2").replace(/∈/g, " in ").replace(/∉/g, " notin ").replace(/⊆/g, " subset ").replace(/⊑/g, " <: ").replace(/—/g, "--").replace(/–/g, "-");
const tidyMath = (s) => s.replace(/\$\s*\$/g, '');

function inline(text) {
  // protect code spans first
  const codes = [];
  let t = text.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  t = escapeTex(t);
  t = uni(t);
  t = t.replace(/\*\*([^*]+)\*\*/g, '\\textbf{$1}')
       .replace(/(^|[^*\w])\*([^*\n]+)\*(?=[^*\w]|$)/g, '$1\\emph{$2}')
       .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, a, u) => `${a}\\footnote{\\url{${u}}}`);
  t = t.replace(/\u0000(\d+)\u0000/g, (_, i) => {
    const c = codes[+i];
    // \texttt with escaped specials; keep unicode via uni() for operators inside code
    return /[\/.]/.test(c) && !/[\\{}]/.test(c) ? `\\path{${c}}` : `\\texttt{${tidyMath(uni(escapeTex(c)))}}`;
  });
  return tidyMath(t);
}

// ---------- blocks ----------
const lines = src.split('\n');
const out = [];
let i = 0;
let title = null, subtitle = null;
const para = [];
const flushPara = () => { if (para.length) { out.push(inline(para.join(' ')) + '\n'); para.length = 0; } };

while (i < lines.length) {
  const L = lines[i];
  // fenced code
  if (/^```/.test(L)) {
    flushPara();
    const lang = L.slice(3).trim();
    const buf = [];
    i++;
    while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
    i++;
    out.push(`\\begin{lstlisting}\n${asciiBox(buf.join('\n'))}\n\\end{lstlisting}\n`);
    continue;
  }
  // headings
  let m;
  if ((m = /^(#{1,4})\s+(.*)$/.exec(L))) {
    flushPara();
    const lvl = m[1].length, txt = m[2].trim();
    if (lvl === 1 && !title) { title = txt; i++; continue; }
    if (lvl === 2 && title && !subtitle && !out.length && /^[A-Z]/.test(txt) && !/^\d/.test(txt) && !/^Abstract/.test(txt)) { subtitle = txt; i++; continue; }
    const clean = txt.replace(/^\d+(\.\d+)*\.?\s*/, '');
    if (/^Abstract$/i.test(clean)) { out.push('%%ABSTRACT-START\n'); i++; continue; }
    const cmd = lvl <= 2 ? 'section' : lvl === 3 ? 'subsection' : 'subsubsection';
    if (lvl <= 2 && out.length && out[out.length - 1] !== '%%ABSTRACT-END\n' && out.some(x => x === '%%ABSTRACT-START\n') && !out.some(x => x === '%%ABSTRACT-END\n')) out.push('%%ABSTRACT-END\n');
    out.push(`\\${cmd}{${inline(clean)}}\n`);
    i++; continue;
  }
  // rule
  if (/^-{3,}\s*$/.test(L)) { flushPara(); i++; continue; }
  // blockquote
  if (/^>/.test(L)) {
    flushPara();
    const buf = [];
    while (i < lines.length && /^>/.test(lines[i])) { buf.push(lines[i].replace(/^>\s?/, '')); i++; }
    out.push(`\\begin{quote}\n${inline(buf.join(' '))}\n\\end{quote}\n`);
    continue;
  }
  // table
  if (/^\|/.test(L) && i + 1 < lines.length && /^\|?\s*:?-{2,}/.test(lines[i + 1])) {
    flushPara();
    const header = L.split('|').slice(1, -1).map(s => s.trim());
    i += 2;
    const rows = [];
    while (i < lines.length && /^\|/.test(lines[i])) { rows.push(lines[i].split('|').slice(1, -1).map(s => s.trim())); i++; }
    const n = header.length;
    const spec = 'p{' + (0.95 / n).toFixed(2) + '\\linewidth}';
    out.push(`\\begin{table}[htbp]\\centering\\small\n\\begin{tabular}{${Array(n).fill(spec).join('')}}\n\\toprule\n${header.map(inline).join(' & ')} \\\\\n\\midrule\n${rows.map(r => r.map(inline).join(' & ') + ' \\\\').join('\n')}\n\\bottomrule\n\\end{tabular}\n\\end{table}\n`);
    continue;
  }
  // lists
  if (/^\s*([-*]|\d+\.)\s+/.test(L)) {
    flushPara();
    const ordered = /^\s*\d+\./.test(L);
    const items = [];
    while (i < lines.length && (/^\s*([-*]|\d+\.)\s+/.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
      if (/^\s*([-*]|\d+\.)\s+/.test(lines[i])) items.push(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ''));
      else items[items.length - 1] += ' ' + lines[i].trim();
      i++;
    }
    const env = ordered ? 'enumerate' : 'itemize';
    out.push(`\\begin{${env}}\n${items.map(it => '  \\item ' + inline(it)).join('\n')}\n\\end{${env}}\n`);
    continue;
  }
  // display code indented 4 spaces (the paper's typing rules)
  if (/^    \S/.test(L) && !para.length) {
    const buf = [];
    while (i < lines.length && (/^    /.test(lines[i]) || lines[i].trim() === '')) { if (lines[i].trim() === '' && !(i + 1 < lines.length && /^    /.test(lines[i + 1]))) break; buf.push(lines[i].replace(/^    /, '')); i++; }
    out.push(`\\begin{lstlisting}\n${asciiBox(buf.join('\n'))}\n\\end{lstlisting}\n`);
    continue;
  }
  if (L.trim() === '') { flushPara(); i++; continue; }
  para.push(L.trim()); i++;
}
flushPara();
if (out.some(x => x === '%%ABSTRACT-START\n') && !out.some(x => x === '%%ABSTRACT-END\n')) out.push('%%ABSTRACT-END\n');

// assemble: abstract between markers
let body = out.join('\n');
body = body.replace(/%%ABSTRACT-START\n([\s\S]*?)%%ABSTRACT-END\n/, (_, a) => `\\begin{abstract}\n${a}\\end{abstract}\n\\maketitle\n`);

const preambleAcm = `\\documentclass[sigplan,review,anonymous,10pt]{acmart}
\\usepackage{listings,booktabs,url}
\\lstset{basicstyle=\\ttfamily\\footnotesize,breaklines=true,columns=fullflexible,keepspaces=true}
\\settopmatter{printfolios=true}
\\title{${inline(title || 'Untitled')}}
${subtitle ? `\\subtitle{${inline(subtitle)}}` : ''}
\\author{Anonymous Author(s)}
\\begin{document}
`;
const preambleArt = `\\documentclass[10pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{amsmath,amssymb,listings,booktabs,url,hyperref}
\\lstset{basicstyle=\\ttfamily\\footnotesize,breaklines=true,columns=fullflexible,keepspaces=true}
\\title{${inline(title || 'Untitled')}${subtitle ? `\\\\ \\large ${inline(subtitle)}` : ''}}
\\author{}
\\date{}
\\begin{document}
`;
let doc = (cls === 'acmart' ? preambleAcm : preambleArt) + body + '\n\\end{document}\n';
// article has no \subtitle; abstract must come after \maketitle for article
if (cls !== 'acmart') doc = doc.replace(/\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}\n\\maketitle\n/, (_, a) => `\\maketitle\n\\begin{abstract}${a}\\end{abstract}\n`);
fs.writeFileSync(outPath, doc);
console.log(`wrote ${outPath}: ${doc.split('\n').length} lines, class=${cls}, title=${JSON.stringify(title)}`);
