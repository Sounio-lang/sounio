#!/usr/bin/env python3
"""Export dynamic-FC switching events to the Brain O-SSM ABIDE manifest format.

The exported manifest is a bridge artifact for the existing compiled Sounio
`examples/brain_ossm_abide.sio` benchmark. Each row is a switch-event example,
not an ASD diagnostic subject. Features are built only from dynamic-state
history available before the target transition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEQ_LEN = 8
INPUT_DIM = 8
FEATURE_COUNT = SEQ_LEN * INPUT_DIM
DEFAULT_CONFIG = {
    "global_train_epochs": "8",
    "global_train_lr": "0.015",
    "global_core_lr_scale": "0.25",
    "oct_train_epochs": "8",
    "oct_train_lr": "0.015",
    "oct_core_lr_scale": "0.25",
    "oct_warmup_epochs": "2",
    "h_train_epochs": "8",
    "h_train_lr": "0.015",
    "h_core_lr_scale": "0.25",
    "h_warmup_epochs": "2",
    "trace_hidden_state": "0",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value: str, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def parse_binary(value: str) -> int | None:
    if value == "":
        return None
    parsed = int(float(value))
    if parsed not in (0, 1):
        raise ValueError(f"switch_event must be binary, got {value!r}")
    return parsed


def event_sort_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (row["fold_id"], row["subject_id"], parse_int(row["window_index"]))


def compute_history_vectors(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_subject: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=event_sort_key):
        by_subject[(row["fold_id"], row["subject_id"])].append(row)

    events: list[dict[str, Any]] = []
    for (fold_id, subject_id), subject_rows in by_subject.items():
        subject_rows.sort(key=lambda row: parse_int(row["window_index"]))
        run_lengths: list[int] = []
        current_run = 0
        previous_state: int | None = None
        for row in subject_rows:
            state = parse_int(row["state"])
            if previous_state is None or state != previous_state:
                current_run = 1
            else:
                current_run += 1
            run_lengths.append(current_run)
            previous_state = state

        window_count = max(parse_int(subject_rows[0].get("window_count", "0")), len(subject_rows))
        n_timepoints = parse_int(subject_rows[0].get("n_timepoints", "0"))
        for idx, row in enumerate(subject_rows):
            target = parse_binary(row.get("switch_event", ""))
            if target is None:
                continue
            features: list[float] = []
            for step in range(SEQ_LEN):
                hist_idx = idx - SEQ_LEN + step
                vec = [0.0] * INPUT_DIM
                if hist_idx >= 0:
                    hist_row = subject_rows[hist_idx]
                    state = parse_int(hist_row["state"])
                    if 0 <= state < 4:
                        vec[state] = 1.0
                    else:
                        vec[state % 4] = 0.5
                    previous_switch = parse_binary(hist_row.get("switch_event", "")) or 0
                    vec[4] = float(previous_switch)
                    vec[5] = min(run_lengths[hist_idx], SEQ_LEN) / float(SEQ_LEN)
                    vec[6] = parse_int(hist_row["window_index"]) / max(window_count - 1, 1)
                    vec[7] = min(math.log1p(max(n_timepoints, 1)) / 6.0, 1.0)
                features.extend(vec)
            if len(features) != FEATURE_COUNT:
                raise AssertionError(f"expected {FEATURE_COUNT} features, got {len(features)}")
            event_id = f"dynfc_{fold_id}_{subject_id}_{parse_int(row['window_index']):03d}"
            events.append(
                {
                    "event_id": event_id,
                    "manifest_subject_id": event_id,
                    "label": str(target),
                    "site": row["site"],
                    "fold_id": fold_id,
                    "holdout_site": row["holdout_site"],
                    "subject_id": subject_id,
                    "source_label": row.get("label", ""),
                    "window_index": parse_int(row["window_index"]),
                    "target_switch_event": target,
                    "features": features,
                }
            )
    return events


def balanced_sample(events: list[dict[str, Any]], max_events: int) -> list[dict[str, Any]]:
    if max_events <= 0 or len(events) <= max_events:
        return events
    selected: list[dict[str, Any]] = []
    by_label = {
        0: [event for event in events if event["target_switch_event"] == 0],
        1: [event for event in events if event["target_switch_event"] == 1],
    }
    first_quota = max_events // 2
    quotas = {1: first_quota, 0: max_events - first_quota}
    for label in (1, 0):
        selected.extend(by_label[label][: quotas[label]])
    if len(selected) < max_events:
        used = {event["event_id"] for event in selected}
        for event in events:
            if event["event_id"] not in used:
                selected.append(event)
                if len(selected) >= max_events:
                    break
    return sorted(selected, key=lambda event: event["event_id"])


def write_manifest(path: Path, events: list[dict[str, Any]], source: Path) -> None:
    header = ["subject_id", "label", "site"] + [f"f{i}" for i in range(FEATURE_COUNT)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# schema=abide_dynamic_fc_switching_sounio_manifest.v1\n")
        handle.write("# target=switch_event; labels are dynamic-FC switching events, not ASD/control diagnoses\n")
        handle.write("# features=8x8 prior-state-history; no current-window state or clinical label is encoded\n")
        handle.write(f"# source={source}\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for event in events:
            writer.writerow(
                [event["manifest_subject_id"], event["label"], event["site"]]
                + [f"{value:.8f}" for value in event["features"]]
            )


def write_config(path: Path, overrides: list[str]) -> dict[str, str]:
    config = dict(DEFAULT_CONFIG)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"config override must be key=value, got {override!r}")
        key, value = override.split("=", 1)
        config[key] = value
    with path.open("w", encoding="utf-8") as handle:
        for key, value in config.items():
            handle.write(f"{key}={value}\n")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-table", required=True, help="dynamic_fc_window_table.tsv")
    parser.add_argument("--output-dir", required=True, help="Directory for Sounio manifest/config outputs")
    parser.add_argument("--split", default="test", help="Window-table split to export")
    parser.add_argument("--max-events", type=int, default=320, help="Deterministic cap for compiled smoke runs")
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--min-sites", type=int, default=2)
    parser.add_argument("--config", action="append", default=[], help="Extra abide_run_config.tsv key=value line")
    args = parser.parse_args()

    window_table = Path(args.window_table)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in read_tsv(window_table)
        if row.get("split") == args.split and row.get("switch_event", "") != ""
    ]
    if not rows:
        raise SystemExit(f"no rows with split={args.split!r} and binary switch_event in {window_table}")

    events = compute_history_vectors(rows)
    events = balanced_sample(events, args.max_events)
    label_counts = Counter(event["target_switch_event"] for event in events)
    site_counts = Counter(event["site"] for event in events)
    if len(events) < args.min_events:
        raise SystemExit(f"too few exported events: {len(events)} < {args.min_events}")
    if len(site_counts) < args.min_sites:
        raise SystemExit(f"too few exported sites: {len(site_counts)} < {args.min_sites}")
    if min(label_counts.values()) <= 0 or len(label_counts) < 2:
        raise SystemExit(f"manifest needs both switch classes, got {dict(label_counts)}")

    manifest_path = output_dir / "abide_roi_manifest.tsv"
    config_path = output_dir / "abide_run_config.tsv"
    event_map_path = output_dir / "dynamic_fc_sounio_event_map.tsv"
    audit_path = output_dir / "dynamic_fc_sounio_manifest_audit.json"
    audit_md_path = output_dir / "dynamic_fc_sounio_manifest_audit.md"

    write_manifest(manifest_path, events, window_table)
    config = write_config(config_path, args.config)
    write_tsv(
        event_map_path,
        [
            {
                key: event[key]
                for key in (
                    "event_id",
                    "manifest_subject_id",
                    "label",
                    "site",
                    "fold_id",
                    "holdout_site",
                    "subject_id",
                    "source_label",
                    "window_index",
                    "target_switch_event",
                )
            }
            for event in events
        ],
        [
            "event_id",
            "manifest_subject_id",
            "label",
            "site",
            "fold_id",
            "holdout_site",
            "subject_id",
            "source_label",
            "window_index",
            "target_switch_event",
        ],
    )
    audit = {
        "schema": "abide_dynamic_fc_switching_sounio_manifest_audit.v1",
        "verdict": "PASS",
        "source_window_table": str(window_table),
        "split": args.split,
        "event_count": len(events),
        "site_count": len(site_counts),
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "feature_shape": {"seq_len": SEQ_LEN, "input_dim": INPUT_DIM, "feature_count": FEATURE_COUNT},
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
        "event_map_path": str(event_map_path),
        "claim_boundary": (
            "Switch-event labels are dynamic-FC temporal events, not ASD/control diagnoses; "
            "this bridge does not establish a clinical, biomarker, mechanism, or O-SSM superiority claim."
        ),
        "config": config,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_md_path.write_text(
        "\n".join(
            [
                "# Dynamic-FC Sounio Manifest Audit",
                "",
                f"- Verdict: `{audit['verdict']}`",
                f"- Events: `{len(events)}`",
                f"- Sites: `{len(site_counts)}`",
                f"- Label counts: `{dict(label_counts)}`",
                f"- Feature shape: `{SEQ_LEN}x{INPUT_DIM}`",
                "",
                "The exported labels are dynamic-FC switch events. They are not ASD/control diagnoses, biomarkers,",
                "clinical-decision outputs, biological mechanisms, or evidence of O-SSM superiority.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"ABIDE_DYNAMIC_FC_SOUNIO_MANIFEST_PASS output_dir={output_dir} events={len(events)} sites={len(site_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
