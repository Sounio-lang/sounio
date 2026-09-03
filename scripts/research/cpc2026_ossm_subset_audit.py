#!/usr/bin/env python3
"""Independent normative/anxious O-SSM subset audit for the CPC 2026 dossier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


A = np.array([
    [0.82, 0.14, 0.04, 0.00, 0.05, 0.02, 0.00, 0.01],
    [0.78, 0.00, 0.16, 0.06, 0.00, 0.03, 0.02, 0.00],
    [0.80, 0.03, 0.00, 0.18, 0.04, 0.00, 0.02, 0.01],
    [0.76, 0.05, 0.04, 0.00, 0.14, 0.02, 0.03, 0.02],
], dtype=np.float32)
B = np.array([
    [0.91, 0.03, 0.02, 0.00, 0.00, 0.01, 0.00, 0.00],
    [0.89, 0.00, 0.04, 0.02, 0.00, 0.00, 0.01, 0.00],
    [0.90, 0.01, 0.00, 0.05, 0.02, 0.00, 0.00, 0.01],
    [0.88, 0.02, 0.01, 0.00, 0.04, 0.01, 0.00, 0.00],
], dtype=np.float32)
C = np.array([
    [0.45, 0.08, 0.03, 0.12, 0.04, 0.04, 0.02, 0.01],
    [0.42, 0.02, 0.09, 0.08, 0.03, 0.04, 0.04, 0.02],
    [0.40, 0.03, 0.02, 0.14, 0.05, 0.02, 0.04, 0.03],
    [0.44, 0.05, 0.04, 0.04, 0.11, 0.03, 0.03, 0.02],
], dtype=np.float32)
D = np.array([
    [0.55, 0.10, 0.04, 0.18, 0.02, 0.01, 0.00, 0.00],
    [0.50, 0.02, 0.12, 0.14, 0.03, 0.01, 0.01, 0.00],
    [0.53, 0.03, 0.02, 0.20, 0.02, 0.00, 0.01, 0.01],
    [0.48, 0.04, 0.03, 0.10, 0.11, 0.01, 0.00, 0.02],
], dtype=np.float32)

REGIMES = {
    "normative": {
        "temperature": 0.5,
        "valence_gate": 1.0,
        "initial": np.array([
            [0.28, 0.08, 0.08, 0.04, 0.04, 0.02, 0.02, 0.01],
            [0.24, 0.02, 0.09, 0.06, 0.04, 0.03, 0.02, 0.01],
            [0.26, 0.03, 0.04, 0.09, 0.03, 0.02, 0.03, 0.01],
            [0.25, 0.04, 0.03, 0.03, 0.09, 0.02, 0.02, 0.02],
        ], dtype=np.float32),
    },
    "anxious": {
        "temperature": 2.0,
        "valence_gate": 2.0,
        "initial": np.array([
            [-0.12, 0.02, 0.04, -0.28, 0.02, 0.00, 0.00, 0.00],
            [-0.10, 0.01, 0.03, -0.24, 0.01, 0.00, 0.01, 0.00],
            [-0.11, 0.02, 0.02, -0.26, 0.01, 0.01, 0.00, 0.00],
            [-0.09, 0.01, 0.02, -0.22, 0.02, 0.00, 0.00, 0.01],
        ], dtype=np.float32),
    },
}


def oct_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3, a4, a5, a6, a7 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3, b4, b5, b6, b7 = np.moveaxis(b, -1, 0)
    return np.stack([
        a0*b0-a1*b1-a2*b2-a3*b3-a4*b4-a5*b5-a6*b6-a7*b7,
        a0*b1+a1*b0+a2*b3-a3*b2+a4*b5-a5*b4-a6*b7+a7*b6,
        a0*b2-a1*b3+a2*b0+a3*b1+a4*b6+a5*b7-a6*b4-a7*b5,
        a0*b3+a1*b2-a2*b1+a3*b0+a4*b7-a5*b6+a6*b5-a7*b4,
        a0*b4-a1*b5-a2*b6-a3*b7+a4*b0+a5*b1+a6*b2+a7*b3,
        a0*b5+a1*b4-a2*b7+a3*b6-a4*b1+a5*b0-a6*b3+a7*b2,
        a0*b6+a1*b7+a2*b4-a3*b5-a4*b2+a5*b3+a6*b0-a7*b1,
        a0*b7-a1*b6+a2*b5+a3*b4-a4*b3-a5*b2+a6*b1+a7*b0,
    ], axis=-1)


def hidden_entropy(hidden: np.ndarray) -> np.ndarray:
    magnitudes = np.abs(hidden).reshape(hidden.shape[0], -1)
    probabilities = magnitudes / magnitudes.sum(axis=1, keepdims=True)
    return -np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities), 0), axis=1) / np.log(32.0)


def run_regime(input_root: Path, regime: str, n: int, steps: int, chunk_size: int) -> dict[str, np.ndarray]:
    config = REGIMES[regime]
    features = np.loadtxt(input_root / "node_features.csv", delimiter=",", skiprows=1, usecols=range(2, 10), dtype=np.float32)
    nodes = np.loadtxt(
        input_root / f"trajectories_{regime}_nodes.csv",
        delimiter=",", skiprows=1, usecols=range(1, steps + 1), max_rows=n, dtype=np.int32,
    )
    output: dict[str, list[np.ndarray]] = {key: [] for key in ("entropy_production", "associator", "c_ent_variance", "h_entropy")}

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        x_all = features[nodes[start:stop]]
        batch = stop - start
        hidden = np.broadcast_to(config["initial"], (batch, 4, 8)).copy()
        entropies = np.empty((batch, steps), dtype=np.float32)
        associators = np.empty((batch, steps), dtype=np.float32)
        readouts = np.empty((batch, steps), dtype=np.float32)

        for step in range(steps):
            x_t = x_all[:, step].copy()
            x_t[:, 3] *= np.float32(config["valence_gate"])
            previous = hidden
            mixed = hidden + np.float32(0.18) * hidden.mean(axis=1, keepdims=True)
            preactivation = oct_mul(oct_mul(A[None], mixed), B[None])
            preactivation += oct_mul(C[None], x_t[:, None])
            preactivation += np.float32(0.22) * hidden
            scaled = preactivation / np.float32(config["temperature"])
            hidden = scaled / (np.float32(1.0) + np.abs(scaled))

            assoc = oct_mul(oct_mul(hidden, x_t[:, None]), previous) - oct_mul(hidden, oct_mul(x_t[:, None], previous))
            assoc_norm = np.linalg.norm(assoc, axis=2).mean(axis=1)
            entropy = hidden_entropy(hidden).astype(np.float32)
            read_oct = oct_mul(hidden, D[None])
            signal = read_oct[..., 0].mean(axis=1) + np.float32(0.25) * read_oct[..., 3].mean(axis=1)
            readout = x_t[:, 2].copy()
            readout += np.float32(0.18) * np.tanh(signal)
            readout += np.float32(0.08) * assoc_norm * np.sign(x_t[:, 3])
            readout -= np.float32(0.05) * (entropy - np.float32(0.5))

            entropies[:, step] = entropy
            associators[:, step] = assoc_norm
            readouts[:, step] = np.clip(readout, -1.0, 1.0)

        output["entropy_production"].append(np.abs(np.diff(entropies, axis=1)).mean(axis=1))
        output["associator"].append(associators.mean(axis=1))
        output["c_ent_variance"].append(readouts.var(axis=1, ddof=1))
        output["h_entropy"].append(entropies.mean(axis=1))

    return {key: np.concatenate(parts).astype(np.float64) for key, parts in output.items()}


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.sqrt(((len(x)-1)*x.var(ddof=1) + (len(y)-1)*y.var(ddof=1)) / (len(x)+len(y)-2))
    return float((y.mean() - x.mean()) / pooled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scientific-repo", type=Path, default=Path("/workspace/hyperbolic-semantic-networks"))
    parser.add_argument("--native-json", type=Path, default=Path("examples/cognitive_ossm/results/ossm_sounio_native_n1000.json"))
    parser.add_argument("--trajectories", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_root = args.scientific_repo / "data" / "cpc2026" / "sounio_input"
    reference = {name: run_regime(input_root, name, args.trajectories, args.steps, args.chunk_size) for name in REGIMES}
    native = json.loads(args.native_json.read_text(encoding="utf-8"))
    metrics = {
        "entropy_production": "mean_hidden_entropy_production_rate",
        "associator": "mean_associator_norm",
        "c_ent_variance": "mean_c_ent_variance",
        "h_entropy": "mean_h_entropy",
    }
    means = {name: {metric: float(values.mean()) for metric, values in values_by_metric.items()} for name, values_by_metric in reference.items()}
    relative_errors = {
        name: {
            metric: abs(float(native[name][native_key]) - means[name][metric]) / abs(means[name][metric])
            for metric, native_key in metrics.items()
        }
        for name in REGIMES
    }
    comparisons = {
        "anxious_minus_normative_hidden_entropy_production_rate_cohens_d": cohens_d(reference["normative"]["entropy_production"], reference["anxious"]["entropy_production"]),
        "anxious_minus_normative_mean_associator_norm_cohens_d": cohens_d(reference["normative"]["associator"], reference["anxious"]["associator"]),
    }
    max_relative_error = max(value for regime in relative_errors.values() for value in regime.values())
    payload = {
        "scope": {"regimes": list(REGIMES), "trajectories": args.trajectories, "steps": args.steps, "source": str(input_root)},
        "reference_means": means,
        "reference_comparisons": comparisons,
        "legacy_native_relative_errors": relative_errors,
        "max_legacy_native_relative_error": max_relative_error,
        "classification": "not_numerical_parity" if max_relative_error > 0.05 else "bounded_numerical_agreement",
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
