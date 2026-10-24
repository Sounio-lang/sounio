#!/usr/bin/env python3
"""Audit both-Spark material/timing evidence and replay frozen native decisions."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cycle import digest, encoded, verify
from resume_pilot import completed_accounting

NODES = ["gpuorangefs-multi-spark-3c59", "gpuorangefs-multi-spark-8e54"]


def audit(root):
    manifest = verify(root)
    end = json.loads((root / "execution-exit.json").read_text())
    assert end["returncode"] == 0
    assert end["log_sha256"] == digest((root / "execution.log").read_bytes())
    provenance = json.loads((root / "control-analysis.json").read_text())
    engines = provenance["engine_identities"]
    for key in ("engine", "fixture_engine", "parity_engine", "gain_engine"):
        assert digest(Path(engines[key]["path"]).read_bytes()) == engines[key]["sha256"]
    membership = json.loads((root / "control-membership.json").read_text())["members"]
    assert len(membership) == 64
    assert {(r["source_job"], r["request_index"]) for r in membership} == {
        (job, i) for job in (11957, 11958) for i in range(32)}
    gpu = json.loads((root / "gpu-manifest.json").read_text())
    bm = json.loads((root / "benchmark-manifest.json").read_text())
    result = json.loads((root / "benchmark-report.json").read_text())
    expected_ids = [f"{i:03d}" for i in range(18)]
    assert [r["id"] for r in bm["candidates"]] == expected_ids
    assert [r["id"] for r in bm["controls"]] == ["direct", "shuffle"]
    assert gpu["candidates"] == bm["controls"] + bm["candidates"]
    assert len({(r["plan_id"], r["ptx_sha256"]) for r in membership}) == 18
    for row in membership:
        candidate = next(c for c in bm["candidates"] if c["id"] == row["representative"])
        receipt = json.loads((root / (candidate["id"] + ".receipt.json")).read_text())
        assert receipt["plan_id"] == row["plan_id"]
        assert digest((root / candidate["ptx"]).read_bytes()) == row["ptx_sha256"]
    assert (bm["blocks"], bm["vectors"], bm["replays"]) == (30, 16384, 32)
    assert bm["thresholds"] == {"median_ppm": 50000, "low_ci_ppm_exclusive": 0}
    assert bm["method"] == "paired-median-ppm-bootstrap4000-v1"
    assert bm["layout_conversion_included"] is False
    fixtures = (root / "numeric-fixtures.jsonl").read_bytes()
    assert digest(fixtures) == gpu["fixtures_sha256"] == bm["fixtures_sha256"]
    rows = [json.loads(l) for l in fixtures.splitlines()]
    reference = "".join("".join(f"{int(x) & ((1 << 64) - 1):016x}" for x in row["output_bits"]) + "\n"
                        for row in rows).encode()
    jobs = {"probe": set(), "timing": set()}
    parity_checks = 0
    timing_rows = 0
    for node in NODES:
        observed = root / ("observed-" + node)
        timings = root / ("timings-" + node)
        for phase, folder in (("probe", observed), ("timing", timings)):
            inv = json.loads((folder / "inventory.json").read_text())
            assert inv["node"] == node and inv["job"].isdecimal()
            jobs[phase].add(inv["job"])
        assert (observed / "reference.hex").read_bytes() == reference
        for c in gpu["candidates"]:
            saved = observed / (c["id"] + ".parity.json")
            replay = subprocess.run([engines["parity_engine"]["path"], str(observed / "reference.hex"),
                                     str(observed / (c["id"] + ".hex"))],
                                     capture_output=True, timeout=120, check=True)
            assert replay.stdout == saved.read_bytes()
            r = json.loads(replay.stdout)
            assert r["decision"] == "PASS" and r["mismatches"] == 0 and r["exact_bits"] == 5120
            parity_checks += 1
        for c in bm["candidates"]:
            for control in bm["controls"]:
                data = (timings / (c["id"] + "." + control["id"] + ".tsv")).read_text()
                values = [[int(x) for x in line.split()] for line in data.splitlines()]
                assert len(values) == 30
                assert all(len(v) == 4 and v[0] == i and v[1] == i % 2 and
                           0 < v[2] < 1000000000 and 0 < v[3] < 1000000000 for i, v in enumerate(values))
                timing_rows += len(values)
    assert all(len(v) == 1 for v in jobs.values())
    jobs = {k: next(iter(v)) for k, v in jobs.items()}
    for phase, stage in (("probe", "MATERIAL_OBSERVATION"), ("timing", "PAIRED_TIMING_OBSERVED")):
        events = []
        for line in (root / (phase + ".log")).read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("stage") == stage:
                assert e["job"] == jobs[phase] and e["node"] in NODES
                folder = root / (("observed-" if phase == "probe" else "timings-") + e["node"])
                name = e["candidate"] + (".hex" if phase == "probe" else "." + e["control"] + ".tsv")
                assert digest((folder / name).read_bytes()) == e["sha256"]
                events.append((e["node"], name))
        expected = {(node, c["id"] + ".hex") for node in NODES for c in gpu["candidates"]} if phase == "probe" else {
            (node, c["id"] + "." + control["id"] + ".tsv") for node in NODES for c in bm["candidates"] for control in bm["controls"]}
        assert set(events) == expected and len(events) == len(expected)
    medians = []
    assert [d["candidate"] for d in result["decisions"]] == expected_ids
    for d in result["decisions"]:
        paths = [root / ("timings-" + node) / (d["candidate"] + "." + control["id"] + ".tsv")
                 for node in NODES for control in bm["controls"]]
        replay = subprocess.run([engines["gain_engine"]["path"], *map(str, paths)],
                                capture_output=True, timeout=120, check=True)
        assert replay.stdout == (root / (d["candidate"] + ".gain.json")).read_bytes()
        assert json.loads(replay.stdout) == d["receipt"]
        medians += [v["median_gain_ppm"] for v in d["receipt"]["comparisons"]]
    assert result["material_candidates"] == result["material_pass"] == 18
    assert result["material_refused"] == 0
    assert result["gain_eligible"] == sum(d["receipt"]["gain_gate"] == "PASS" for d in result["decisions"])
    command = ["kubectl", "-n", "slurm-pilot", "exec", "slurm-pilot-controller-0", "-c", "slurmctld",
               "--", "sacct", "-j", ",".join(jobs.values()),
               "--format=JobIDRaw,State,ExitCode,NodeList,Start,End", "-n", "-P", "--noconvert"]
    raw = subprocess.check_output(command, text=True, timeout=60)
    accounting = {phase: completed_accounting(raw, job) for phase, job in jobs.items()}
    # Verify the untouched original pilot and its failed attempt independently.
    pilot = Path("/workspace/.cache/pireus-continuity/pilot-3x3x32-20260907-v4")
    original = json.loads((pilot / "round-0-deterministic/pilot-cell-complete.json").read_text())
    assert original["generated"] == original["hardware_benchmarked"] == 32
    assert len(list(pilot.glob("round-*/pilot-cell-complete.json"))) == 1
    negative = json.loads((HERE / "validation/pilot-memory-stop-11956.json").read_text())
    assert digest(Path(negative["log_path"]).read_bytes()) == negative["log_sha256"]
    output = dict(schema="pireus-control-hardware-audit-v1",
                  audited_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  jobs=jobs, accounting=accounting, accounting_command=command, accounting_raw=raw,
                  unique_materials=18, preserved_occurrences=64,
                  candidate_node_parity_pass=36, control_node_parity_pass=4,
                  native_parity_replays=parity_checks, exact_components_per_comparison=5120,
                  timing_comparisons=72, timing_blocks=timing_rows,
                  native_gain_replays=18, gain_eligible=result["gain_eligible"],
                  median_gain_ppm_range=[min(medians), max(medians)],
                  scope="resident-layout-kernel; layout conversion excluded",
                  pilot_cells_completed=1, pilot_proposals_accepted=32,
                  failed11956_unchanged=True, pilot_acceptance=False,
                  memory_root_cause_established=False,
                  source_code_sha256=digest(Path(__file__).read_bytes()),
                  file_hashes={str(p.relative_to(root)): digest(p.read_bytes())
                               for p in root.rglob("*") if p.is_file() and p.name not in (".lock", "hardware-audit.json")})
    with (root / "hardware-audit.json").open("xb") as f:
        f.write(encoded(output))
    print(json.dumps({k: v for k, v in output.items() if k != "file_hashes"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    audit(p.parse_args().run)
