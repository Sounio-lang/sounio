<!-- docs:meta
topic_id: repo.docs.ledger.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ledger.readme
-->

# docs/ledger — the Sounio work, as one spreadsheet

```bash
python3 tools/ledger/build_ledger.py --out sounio_ledger.xlsx \
    --uhs-repo ../sounio-uhs-coupled \
    --gri-repo ../sounio-gri30-crossvalidation \
    --verify          # re-runs every producer command (about 4 min)
```

| sheet | source | edit by hand? |
|---|---|---|
| Resumo | formulas over the other sheets | no |
| Linhas de pesquisa | `research_lines.tsv`, plus a check that each path exists | the TSV |
| Issues e PRs | GitHub REST, open items of `Sounio-lang/sounio` | the "Prioridade" column only |
| Limitações | `defects.tsv` + KL-xx headings of `docs/compiler/KNOWN_LIMITATIONS.md` + G-xx headings of the uhs-coupled `LANGUAGE_GAPS.md` | `defects.tsv` |
| Resultados | `results.tsv`; `--verify` runs each `comando` and looks for `espera` in its output | `results.tsv` |

**Rules for `results.tsv`.** One row per published number. `comando` must be the
command that produced it, run from the repository root named in `repositorio`,
and `espera` a string it prints. A row is never edited in place when the number
changes: mark the old row `corrigido` or `retirado`, say in `citado_em` what
replaced it, and add a new row. With `--verify`, a missing producer file is
reported as **Ausente**, never as valid. A row is **Verificado** only when its
producer exits 0 *and* prints `espera`; a non-zero exit is **Falhou** (exit code
in the note) even if the string was printed before the failure. The
`Lean`-backed row needs `lake` on `PATH` (`~/.elan/bin`), or it fails with
code 127.

The generator only marshals data into cells; it computes no scientific
number. Yellow cells in the workbook are for a person to fill in.

CI (`.github/workflows/ledger.yml`) checks the generator, not the science: the
verdict unit tests (`python3 -m unittest tools/ledger/test_build_ledger.py`) and
an offline `--no-github` smoke build that must produce the five sheets. It does
not run `--verify`.
