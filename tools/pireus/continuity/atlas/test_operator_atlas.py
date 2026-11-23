#!/usr/bin/env python3
"""Transport-only checks for eisa_h.pireus_operator_atlas.v1.

Python decides nothing about novelty here. It (a) re-reads the 62 class words
from the frozen V13 transcript bytes and compares them with what the Sounio
atlas prints, and (b) runs the admission engine and checks the receipt
contract. Every classification comes from the Sounio executables.

usage: test_operator_atlas.py ATLAS_TEST.elf ADMISSION.elf
"""
import hashlib, json, pathlib, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
PIREUS = HERE.parents[1]
TRANSCRIPT = PIREUS / "operator_orbit_canonicalization.first.v13"
FREEZE = PIREUS / "operator_orbit_canonicalization.freeze.v13"


def frozen_classes():
    data = TRANSCRIPT.read_bytes()
    freeze = dict(line.split("=", 1) for line in FREEZE.read_text().splitlines() if "=" in line)
    assert hashlib.sha256(data).hexdigest() == freeze["first_transcript_sha256"], "V13 transcript bytes changed"
    text = data.decode().replace("\n:", ":").replace("\n ", " ").replace("\n;", ";")
    lines = [line for line in text.split("\n") if line.startswith("PIREUS_POC")]
    fields = lambda line: dict(t.split("=", 1) for t in line.split()[1:] if "=" in t)
    classes = []
    for line in lines:
        if line.startswith("PIREUS_POC_BASELINE "):
            d = fields(line)
            words = [int(x) for x in d["words"].split(":")]
            cid = int(d["class"])
            if cid == len(classes):
                classes.append(words)
            assert classes[cid] == words, ("baseline class words differ", cid)
    assert len(classes) == 30, len(classes)
    for line in lines:
        if line.startswith("PIREUS_POC_CANON_BITS "):
            d = fields(line)
            assert int(d["admitted"]) == len(classes) - 30
            bits = [int(x) for x in d["bits"].split(":")]
            assert len(bits) == 256 and set(bits) <= {0, 1}
            words = [0] * 8
            for cell, bit in enumerate(bits):
                words[cell // 32] |= bit << (cell % 32)
            classes.append(words)
    assert len(classes) == 62 and len({tuple(w) for w in classes}) == 62
    return classes


def atlas_controls(engine, classes):
    run = subprocess.run([engine], capture_output=True, text=True, timeout=1800)
    out = run.stdout
    assert run.returncode == 0 and "ATLAS_RESULT PASS" in out, out[-3000:]
    for marker in ("ATLAS_PASS reproduction references=65 v13=62", "ATLAS_PASS invariance transforms=35",
                   "ATLAS_PASS phase_equivalences sedenion_class=63", "ATLAS_PASS negatives=6"):
        assert marker in out, marker
    printed = {}
    for line in out.splitlines():
        if line.startswith("ATLAS_CLASS "):
            _, cid, words = line.split()
            printed[int(cid)] = [int(x) for x in words.split(":")]
    assert sorted(printed) == list(range(65)), sorted(printed)
    for cid, words in enumerate(classes):
        assert printed[cid] == words, ("Sounio atlas table differs from the frozen V13 transcript", cid)
    print("ATLAS_TABLE_MATCHES_FROZEN_V13 classes=62", flush=True)


def raw(value):
    return (json.dumps(value, separators=(",", ":")) + "\n").encode()


def admission_contract(engine):
    context = dict(schema=1, target=701202, dimension=16, precision=64, order=1, fma=0,
                   capabilities=55, lane_width=32, facts_state=1)
    proposal = dict(schema=1, target=701202, dimension=16, precision=64, order=1, fma=0, kind=1,
                    lane_stride=1, lane_offset=0, load=1, layout=0, unroll=1)
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        cb = raw(context)
        (root / "context.json").write_bytes(cb)

        def admit(**changes):
            p = proposal | {"context": hashlib.sha256(cb).hexdigest()} | changes
            (root / "p.json").write_bytes(raw(p))
            run = subprocess.run([engine, str(root / "context.json"), str(root / "p.json")],
                                 capture_output=True, text=True, timeout=60)
            assert run.returncode == 0, run.stdout
            return json.loads(run.stdout)

        lowering = admit()
        assert lowering["schema"] == 1 and "novelty" not in lowering and "atlas_class" not in lowering, lowering
        seen = {}
        for phase, level, cls, origin in ((0, "N0", 63, "V1_CORPUS"), (18, "N1", 63, "V1_CORPUS"),
                                          (2, None, None, None), (16, None, None, None),
                                          (36077, None, None, None), (36007, None, None, None)):
            r = admit(kind=2, phase=phase)
            assert r["schema"] == 2 and r["kind"] == "operator" and r["phase_code"] == phase, r
            assert r["classifier"] == "eisa_h.pireus_operator_atlas.v1", r
            assert r["novelty"] in ("N0", "N1", "N2"), r
            assert r["claim_ready"] is False and r["scientific_novelty"] is False and r["historical_novelty"] is False, r
            assert r["atlas_v13_semantics_sha256"] == "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c"
            assert r["atlas_class_digest"] == "5d13a22658f9fd1b5c44dafa62afc7839b2de0969e903c0aab7379e32f0df09c", r
            assert len(r["canonical_table"]) == 64, r
            assert (r["novelty"] == "N2") == (r["atlas_class"] == -1 and r["atlas_origin"] == "NONE"), r
            if level:
                assert (r["novelty"], r["atlas_class"], r["atlas_origin"]) == (level, cls, origin), r
            seen[phase] = r
            print("ATLAS_RECEIPT phase=%d novelty=%s class=%d" % (phase, r["novelty"], r["atlas_class"]), flush=True)
        # Transposed and gauge-related phases name the same class.
        assert seen[2]["canonical_table"] == seen[16]["canonical_table"]
        assert seen[36077]["canonical_table"] == seen[36007]["canonical_table"]
        assert seen[0]["canonical_table"] == seen[18]["canonical_table"]
        # The materialized kernel does not depend on the atlas.
        p = proposal | {"context": hashlib.sha256(cb).hexdigest(), "kind": 2, "phase": 1128}
        (root / "p.json").write_bytes(raw(p))
        run = subprocess.run([engine, str(root / "context.json"), str(root / "p.json"), "ptx"],
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 0 and "novelty" not in run.stdout and ".target sm_121" in run.stdout
    print("ATLAS_ADMISSION_CONTRACT_PASS", flush=True)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    atlas_controls(sys.argv[1], frozen_classes())
    admission_contract(sys.argv[2])
    print("PIREUS_OPERATOR_ATLAS_GATE_PASS", flush=True)


if __name__ == "__main__":
    main()
