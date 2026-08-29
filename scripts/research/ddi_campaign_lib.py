#!/usr/bin/env python3
"""Shared helpers for the DDI O-SSM DrugBank benchmark.

This module centralizes:
- the canonical CYP / Fano mapping
- merged row loading from the DrugBank FAERS, demographics, and concomitant CSVs
- manifest generation
- lightweight statistics helpers used by multiple scripts
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CYP_ORDER: list[str] = [
    "CYP1A2",
    "CYP2C9",
    "CYP2C8",
    "CYP2B6",
    "CYP2C19",
    "CYP2D6",
    "CYP3A4",
]

CYP_TO_INDEX: dict[str, int] = {name: idx + 1 for idx, name in enumerate(CYP_ORDER)}

FANO_TRIPLES: set[tuple[str, str, str]] = {
    ("CYP1A2", "CYP2C9", "CYP2C8"),
    ("CYP1A2", "CYP2B6", "CYP2C19"),
    ("CYP1A2", "CYP2D6", "CYP3A4"),
    ("CYP2C9", "CYP2B6", "CYP2D6"),
    ("CYP2C9", "CYP2C19", "CYP3A4"),
    ("CYP2C8", "CYP2B6", "CYP3A4"),
    ("CYP2C8", "CYP2C19", "CYP2D6"),
}


@dataclass(frozen=True)
class DDISample:
    triple_id: str
    cyp_a: str
    cyp_b: str
    cyp_c: str
    basis_a: int
    basis_b: int
    basis_c: int
    anchor_group: str
    fold_id: int
    fano: bool
    assoc_norm: float
    n_drug_triples: int
    n_positive_triples: int
    cases: int
    a_first: int
    b_first: int
    temporal: int
    asymmetry: float
    mean_age: float | None
    frac_female: float | None
    frac_hcp: float | None
    frac_us: float | None
    concomitant_total: int
    concomitant_by_cyp: tuple[int, ...]
    sequence: tuple[tuple[float, ...], ...]


def write_json(path: str | Path, payload: object) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_tsv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def basis_sequence(cyps: tuple[str, str, str]) -> tuple[tuple[float, ...], ...]:
    sequence: list[tuple[float, ...]] = []
    for cyp in cyps:
        index = CYP_TO_INDEX[cyp]
        step = [0.0] * 8
        step[index] = 1.0
        sequence.append(tuple(step))
    return tuple(sequence)


def assoc_norm_from_indices(i: int, j: int, k: int) -> float:
    triple = tuple(sorted((CYP_ORDER[i - 1], CYP_ORDER[j - 1], CYP_ORDER[k - 1])))
    return 0.0 if triple in FANO_TRIPLES else 2.0


def _optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number < 0.0:
        return None
    return number


def _load_csv_map(path: Path, key_fields: tuple[str, str, str]) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = tuple(row[field] for field in key_fields)
            rows[key] = row
    return rows


def assign_folds(
    triples: list[tuple[str, str, str]],
    fold_count: int = 5,
) -> dict[tuple[str, str, str], int]:
    """Deterministic 5-fold assignment balancing Fano and target spread."""
    indexed = []
    for triple in triples:
        fano = triple in FANO_TRIPLES
        indexed.append((fano, triple))
    indexed.sort(key=lambda item: (0 if item[0] else 1, item[1]))

    folds: dict[tuple[str, str, str], int] = {}
    strata: dict[bool, list[tuple[str, str, str]]] = {True: [], False: []}
    for fano, triple in indexed:
        strata[fano].append(triple)

    for triples_in_stratum in strata.values():
        for idx, triple in enumerate(triples_in_stratum):
            folds[triple] = idx % fold_count if triple not in folds else folds[triple]
    return folds


def load_ddi_samples(
    repo_root: str | Path,
    dataset_name: str = "faers_drugbank.csv",
    fold_count: int = 5,
) -> list[DDISample]:
    repo = Path(repo_root)
    data_dir = repo / "data"
    base_rows = _load_csv_map(
        data_dir / dataset_name,
        ("cyp_a", "cyp_b", "cyp_c"),
    )
    demographics = _load_csv_map(
        data_dir / "faers_demographics.csv",
        ("cyp_a", "cyp_b", "cyp_c"),
    )
    concomitants = _load_csv_map(
        data_dir / "faers_concomitants.csv",
        ("cyp_a", "cyp_b", "cyp_c"),
    )

    triples = sorted(base_rows.keys())
    fold_map = assign_folds(triples, fold_count=fold_count)

    samples: list[DDISample] = []
    for triple in triples:
        base = base_rows[triple]
        demo = demographics.get(triple, {})
        conc = concomitants.get(triple, {})

        basis = tuple(CYP_TO_INDEX[name] for name in triple)
        assoc_norm = assoc_norm_from_indices(*basis)
        sequence = basis_sequence(triple)
        conc_by_cyp = tuple(int(conc.get(f"conc_cyp{i}", 0) or 0) for i in range(1, 8))

        sample = DDISample(
            triple_id="__".join(triple),
            cyp_a=triple[0],
            cyp_b=triple[1],
            cyp_c=triple[2],
            basis_a=basis[0],
            basis_b=basis[1],
            basis_c=basis[2],
            anchor_group=triple[0],
            fold_id=fold_map[triple],
            fano=triple in FANO_TRIPLES,
            assoc_norm=assoc_norm,
            n_drug_triples=int(base["n_drug_triples"]),
            n_positive_triples=int(base["n_positive_triples"]),
            cases=int(base["cases"]),
            a_first=int(base["a_first"]),
            b_first=int(base["b_first"]),
            temporal=int(base["temporal"]),
            asymmetry=float(base["asymmetry"]),
            mean_age=_optional_float(demo.get("mean_age")),
            frac_female=_optional_float(demo.get("frac_female")),
            frac_hcp=_optional_float(demo.get("frac_hcp")),
            frac_us=_optional_float(demo.get("frac_us")),
            concomitant_total=sum(conc_by_cyp),
            concomitant_by_cyp=conc_by_cyp,
            sequence=sequence,
        )
        samples.append(sample)
    return samples


def write_ddi_manifest(samples: list[DDISample], path: str | Path, fold_count: int = 5) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_columns: list[str] = []
    for time_idx in range(3):
        for feature_idx in range(8):
            feature_columns.append(f"t{time_idx}_f{feature_idx}")

    scalar_columns = [
        "n_drug_triples",
        "n_positive_triples",
        "cases",
        "a_first",
        "b_first",
        "temporal",
        "mean_age",
        "frac_female",
        "frac_hcp",
        "frac_us",
        "concomitant_total",
        "assoc_norm",
    ] + [f"conc_cyp{i}" for i in range(1, 8)]

    header = [
        "triple_id",
        "target",
        "fold_id",
        "anchor_group",
        "fano",
        "cyp_a",
        "cyp_b",
        "cyp_c",
    ] + scalar_columns + feature_columns

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# schema=ddi_ossm.drugbank.v1\n")
        handle.write("# seq_len=3\n")
        handle.write("# input_dim=8\n")
        handle.write("# feature_layout=explicit_seq\n")
        handle.write("# label_space=asymmetry_regression\n")
        handle.write(f"# split_mode=stratified_{fold_count}fold\n")
        handle.write("# sequence_semantics=cyp_basis_triplet\n")
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for sample in samples:
            row = [
                sample.triple_id,
                f"{sample.asymmetry:.8f}",
                sample.fold_id,
                sample.anchor_group,
                "1" if sample.fano else "0",
                sample.cyp_a,
                sample.cyp_b,
                sample.cyp_c,
                sample.n_drug_triples,
                sample.n_positive_triples,
                sample.cases,
                sample.a_first,
                sample.b_first,
                sample.temporal,
                "" if sample.mean_age is None else f"{sample.mean_age:.8f}",
                "" if sample.frac_female is None else f"{sample.frac_female:.8f}",
                "" if sample.frac_hcp is None else f"{sample.frac_hcp:.8f}",
                "" if sample.frac_us is None else f"{sample.frac_us:.8f}",
                sample.concomitant_total,
                f"{sample.assoc_norm:.8f}",
                *sample.concomitant_by_cyp,
            ]
            for step in sample.sequence:
                row.extend(f"{value:.8f}" for value in step)
            writer.writerow(row)


def load_ddi_manifest(path: str | Path) -> list[DDISample]:
    manifest_path = Path(path)
    rows: list[DDISample] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        table_lines = [line for line in handle if line.strip() and not line.startswith("#")]
    reader = csv.DictReader(table_lines, delimiter="\t")
    for row in reader:
        triple = (row["cyp_a"], row["cyp_b"], row["cyp_c"])
        sequence: list[tuple[float, ...]] = []
        for time_idx in range(3):
            step = []
            for feature_idx in range(8):
                step.append(float(row[f"t{time_idx}_f{feature_idx}"]))
            sequence.append(tuple(step))
        concomitant_by_cyp = tuple(int(row[f"conc_cyp{i}"]) for i in range(1, 8))
        rows.append(
            DDISample(
                triple_id=row["triple_id"],
                cyp_a=triple[0],
                cyp_b=triple[1],
                cyp_c=triple[2],
                basis_a=CYP_TO_INDEX[triple[0]],
                basis_b=CYP_TO_INDEX[triple[1]],
                basis_c=CYP_TO_INDEX[triple[2]],
                anchor_group=row["anchor_group"],
                fold_id=int(row["fold_id"]),
                fano=row["fano"] in {"1", "True", "true"},
                assoc_norm=float(row["assoc_norm"]),
                n_drug_triples=int(row["n_drug_triples"]),
                n_positive_triples=int(row["n_positive_triples"]),
                cases=int(row["cases"]),
                a_first=int(row["a_first"]),
                b_first=int(row["b_first"]),
                temporal=int(row["temporal"]),
                asymmetry=float(row["target"]),
                mean_age=_optional_float(row.get("mean_age")),
                frac_female=_optional_float(row.get("frac_female")),
                frac_hcp=_optional_float(row.get("frac_hcp")),
                frac_us=_optional_float(row.get("frac_us")),
                concomitant_total=int(row["concomitant_total"]),
                concomitant_by_cyp=concomitant_by_cyp,
                sequence=tuple(sequence),
            )
        )
    return rows


def mean_or_zero(values: Iterable[float]) -> float:
    values_list = list(values)
    return statistics.fmean(values_list) if values_list else 0.0


def stdev_or_zero(values: Iterable[float]) -> float:
    values_list = list(values)
    return statistics.pstdev(values_list) if len(values_list) > 1 else 0.0


def pearson_r(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 3:
        return 0.0, 1.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0, 1.0
    r = cov / (sx * sy)
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    p = 2 * (1 - t_cdf_approx(abs(t), n - 2))
    return r, max(0.0, p)


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][1] == indexed[pos][1]:
            end += 1
        avg_rank = (pos + end - 1) / 2.0 + 1.0
        for idx in range(pos, end):
            ranks[indexed[idx][0]] = avg_rank
        pos = end
    return ranks


def spearman_r(xs: list[float], ys: list[float]) -> tuple[float, float]:
    return pearson_r(rankdata(xs), rankdata(ys))


def t_cdf_approx(t: float, df: int) -> float:
    if df > 30:
        return norm_cdf(t)
    x = t * (1 - 1 / max(1.0, 4.0 * df))
    return norm_cdf(x)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def solve_linear(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(a)
    matrix = [row[:] + [b[idx]] for idx, row in enumerate(a)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda row_idx: abs(matrix[row_idx][col]))
        matrix[col], matrix[max_row] = matrix[max_row], matrix[col]
        if abs(matrix[col][col]) < 1e-12:
            return None
        for row_idx in range(col + 1, n):
            factor = matrix[row_idx][col] / matrix[col][col]
            for j in range(col, n + 1):
                matrix[row_idx][j] -= factor * matrix[col][j]
    out = [0.0] * n
    for idx in range(n - 1, -1, -1):
        out[idx] = (
            matrix[idx][n] - sum(matrix[idx][j] * out[j] for j in range(idx + 1, n))
        ) / matrix[idx][idx]
    return out


def partial_correlation(
    x: list[float],
    y: list[float],
    covariates: list[list[float]],
) -> tuple[float, float]:
    def residuals(values: list[float], controls: list[list[float]]) -> list[float]:
        n = len(values)
        if not controls or n < len(controls[0]) + 2:
            return values
        p = len(controls) + 1
        design = [[1.0] + [controls[j][i] for j in range(len(controls))] for i in range(n)]
        xtx = [
            [sum(design[i][a] * design[i][b] for i in range(n)) for b in range(p)]
            for a in range(p)
        ]
        xty = [sum(design[i][a] * values[i] for i in range(n)) for a in range(p)]
        beta = solve_linear(xtx, xty)
        if beta is None:
            return values
        return [
            values[i] - sum(design[i][a] * beta[a] for a in range(p))
            for i in range(n)
        ]

    rx = residuals(x, covariates)
    ry = residuals(y, covariates)
    return pearson_r(rx, ry)
