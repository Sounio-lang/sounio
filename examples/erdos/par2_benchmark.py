#!/usr/bin/env python3
"""
S5 PAR-2 benchmark: souc_sat vs kissat 4.0.4 on graph-colouring.

Runs souc_sat once per instance (produces both result + CNF),
then runs kissat on the same CNF. PAR-2 scoring.
"""
import subprocess, sys, os, time, json, shutil

SOLVER  = sys.argv[1] if len(sys.argv) > 1 else "/tmp/souc_sat_s3.elf"
KISSAT  = sys.argv[2] if len(sys.argv) > 2 else "tools/bin/kissat"
BENCHDIR = sys.argv[3] if len(sys.argv) > 3 else "/tmp/colour_bench"
DATADIR = sys.argv[4] if len(sys.argv) > 4 else "examples/erdos/data"
TIMEOUT = 120

instances = [
    # name, edge_file, k, expected
    ("degrey_529_k4",  "degrey_529.edge", 4, "UNSAT"),
    ("parts_510_k4",   "parts_510.edge",  4, "UNSAT"),
    ("mycielski_3_k2", "mycielski_3.edge", 2, "UNSAT"),
    ("mycielski_4_k3", "mycielski_4.edge", 3, "UNSAT"),
    ("mycielski_5_k3", "mycielski_5.edge", 3, "UNSAT"),
    ("mycielski_6_k4", "mycielski_6.edge", 4, "UNSAT"),
    ("mycielski_7_k5", "mycielski_7.edge", 5, "UNSAT"),
    ("queen_5_k4",     "queen_5.edge",     4, "UNSAT"),
    ("queen_6_k6",     "queen_6.edge",     6, "UNSAT"),
    ("queen_7_k7",     "queen_7.edge",     7, "UNSAT"),
    ("queen_8_k8",     "queen_8.edge",     8, "UNSAT"),
    ("complete_5_k4",  "complete_5.edge",  4, "UNSAT"),
    ("complete_6_k5",  "complete_6.edge",  5, "UNSAT"),
    ("complete_7_k6",  "complete_7.edge",  6, "UNSAT"),
]

def find_edge(name):
    for d in [BENCHDIR, DATADIR, os.path.join(DATADIR, "degrey")]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def run_souc_sat(edge_path, k, workdir):
    """Run souc_sat, return (time, result, conflicts). Saves CNF as instance.cnf."""
    # Clean any stale CNF
    for f in os.listdir(workdir):
        os.remove(os.path.join(workdir, f))

    start = time.time()
    try:
        r = subprocess.run(
            [SOLVER, "42", str(k), "1", "1", edge_path],
            cwd=workdir, capture_output=True, text=True, timeout=TIMEOUT
        )
        elapsed = time.time() - start
        out = r.stdout + r.stderr
        if "UNSAT" in out:
            result = "UNSAT"
        elif "SAT colouring" in out:
            result = "SAT"
        else:
            result = "UNKNOWN"
        conflicts = 0
        for line in out.split("\n"):
            if "conflicts=" in line:
                c = line.split("conflicts=")[1].strip().split()[0]
                conflicts = int(c.replace(",", ""))
        # Rename CNF for kissat
        src = os.path.join(workdir, "souc_sat_worker.cnf")
        dst = os.path.join(workdir, "instance.cnf")
        if os.path.exists(src):
            shutil.move(src, dst)
        # Clean DRAT
        drat = os.path.join(workdir, "souc_sat_worker.drat")
        if os.path.exists(drat):
            os.remove(drat)
        return elapsed, result, conflicts
    except subprocess.TimeoutExpired:
        return 2 * TIMEOUT, "TIMEOUT", 0

def run_kissat(cnf_path):
    """Run kissat on pre-generated CNF, return (time, result, conflicts)."""
    if not os.path.exists(cnf_path):
        return 2 * TIMEOUT, "NO_CNF", 0
    start = time.time()
    try:
        r = subprocess.run(
            [KISSAT, cnf_path],
            capture_output=True, text=True, timeout=TIMEOUT
        )
        elapsed = time.time() - start
        out = r.stdout + r.stderr
        if "exit 20" in out:
            result = "UNSAT"
        elif "exit 10" in out:
            result = "SAT"
        else:
            result = "UNKNOWN"
        conflicts = 0
        for line in out.split("\n"):
            if "conflicts:" in line:
                c = line.split("conflicts:")[1].strip().split()[0]
                conflicts = int(c.replace(",", ""))
                break
        return elapsed, result, conflicts
    except subprocess.TimeoutExpired:
        return 2 * TIMEOUT, "TIMEOUT", 0

print(f"{'Instance':<22} {'k':>2} {'Exp':>5} | {'souc_sat':>10} {'conf':>8} {'res':>6} | {'kissat':>10} {'conf':>8} {'res':>6} | {'ratio':>6}")
print("-" * 100)

results = []
souc_par2 = 0
kiss_par2 = 0
souc_correct = 0
kiss_correct = 0
n_run = 0

for name, edge, k, expected in instances:
    edge_path = find_edge(edge)
    if not edge_path:
        print(f"{name:<22} SKIP (file not found: {edge})")
        continue

    workdir = f"/tmp/par2_{name}"
    os.makedirs(workdir, exist_ok=True)

    # 1. Run souc_sat (produces result + CNF)
    st, sr, sc = run_souc_sat(edge_path, k, workdir)

    # 2. Run kissat on the same CNF
    cnf_path = os.path.join(workdir, "instance.cnf")
    kt, kr, kc = run_kissat(cnf_path)

    ratio = st / kt if kt > 0 and kt < 2*TIMEOUT else 0

    souc_par2 += st
    kiss_par2 += kt
    n_run += 1
    if sr == expected: souc_correct += 1
    if kr == expected: kiss_correct += 1

    results.append({
        "instance": name, "k": k, "expected": expected,
        "souc_time": round(st, 2), "souc_result": sr, "souc_conflicts": sc,
        "kiss_time": round(kt, 2), "kiss_result": kr, "kiss_conflicts": kc,
        "ratio": round(ratio, 1) if ratio > 0 else None,
    })

    rt = f"{ratio:.1f}x" if ratio > 0 else "N/A"
    print(f"{name:<22} {k:>2} {expected:>5} | {st:>9.1f}s {sc:>8} {sr:>6} | {kt:>9.1f}s {kc:>8} {kr:>6} | {rt:>6}")

print("-" * 100)
print(f"{'PAR-2 TOTAL':<22} {'':>2} {'':>5} | {souc_par2:>9.1f}s {'':>8} {'':>6} | {kiss_par2:>9.1f}s {'':>8} {'':>6} | {souc_par2/kiss_par2 if kiss_par2>0 else 0:>5.1f}x")
print(f"\nCorrectness: souc_sat {souc_correct}/{n_run}, kissat {kiss_correct}/{n_run}")

with open("/tmp/par2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results: /tmp/par2_results.json")
