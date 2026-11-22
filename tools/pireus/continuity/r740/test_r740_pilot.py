#!/usr/bin/env python3
"""R740 pilot controls. No Slurm submission and no Spark-path edit."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTINUITY = HERE.parent
ROOT = CONTINUITY.parents[2]
sys.path.insert(0, str(HERE))
import gpu_device_key as keys
import paired_timing
import phase_a
import proposals
import r740_launch as launch
import run_inputs

REVIEW_NAME = "20261117T120000Z-1-0123456789abcdef"
SM121_KIND1 = "c55daadb94b2050ca88047316f5f0317ccbe3fe48744b26dae13ad7afed659b3"
SM121_PHASE1128 = "5ddecfab36cd8c5a3dd67bc9772e8c3c44d6399f1691dc34e582e2f9657941dc"
SM121_DET = (
    "90b9f46cbaa8c15ff7312e39ca0373842bf1524bb9d07518b53b48c5bf9e5a23",
    "8c9f44ce584a2d371470b407c7a7d2f47009f5e81f99b0c414e39fc70067b4b0",
    "5dd9d865f8d2fa9d557e9de4ae611e32eed857fcd0a3c4d0d1a02d15b3288244",
    "3264e2ee264ca11636c929e112886474fb5ac9f25ef3a767703ad73847ac3471",
    "a2862eab17c76c00a7285d5a18bbf5dbc4292d315b47efa37b08b81a4b43e541",
    "b8c57ead5078c7ac6451d74e68dedc4c7c28dcfbdba85a14c41b938b6d8332c5",
    "08e4f8f34162659f3e1b83dd951ff8786a5637be746fad25b6d16507ef4ce002",
    "39eab9416e2eb0a815cb71a27336a9e0e9042dff02f669b9f9f156c8b4c4f791",
)
RECEIPT = (
    "e8c2794b561f8383911aaee1ed879b33beb23965b3c13fb0a0612bb5f1fb36b9",
    "26e850d4740ee77a0b81842307a3b4168ae172c5f86e9692172e312771f31f5d",
    "7ef40d36509078c9aacf928368616ec4f3c59af160dc9c64d184cbcea580b43f",
    "a5add43a11c6fd1d5a1e94ad6e5029cab12391b66eb856665ddb1590d06b83fd",
    "4f8f45f1ed4d1940a2340855860fe6b3fe33d3dbd114cf8b50b8503535cf316f",
    "84a9d97f938ef9e898a88cf30bc3b5a7bfaa4260fbbe6afb3f63cb69afe5e00b",
    "99d6507d2d0ed6c5d06f937ca22f3a19928ab6e996be9ef33bce5b5244093441",
    "5916bbf83b6e9a2edcea3cad91c4b1676dd42329212ed7ddd9c2eecbded22e90",
)
UNCHANGED = {
    "tools/pireus/continuity/benchmark_pair.py": "b5f1284a3ed45dff3c6981180d677863de52c98a1a2d05f5e24d4a6eae27702c",
    "tools/pireus/continuity/benchmark_decision.sio": "7bac55c6566591ade6e32a6eef33af3d260a9f481b81e987bb49b80f83dc9846",
    "tools/pireus/continuity/gpu_probe.py": "d9a01f4c4c3e9a3b33cc7150fc11e72cd8627aa3b143e11e4942a245c75ec959",
    "tools/pireus/continuity/gpu_benchmark.py": "b8b398c12b88f8e326948cd4539270368dca93e6a388ca214dcf789ed3d5405e",
    "tools/pireus/continuity/runtime/preflight.py": "7a5e08e0fa6df8d94566fef92a9a84821d7117bc71fcc3a00d867bd633a1a408",
    "tools/pireus/continuity/runtime/launch_pair.py": "0876d596f6f1fe50c342d120825b22ceefd297a288383138cc85bdc25c07dcde",
    "tools/pireus/continuity/runtime/memory_guard.py": "7a5f24ef3863cc1b12c9a219ec25bf7b1a952a21624721ce0c62f58f777abbd8",
    "tools/pireus/continuity/runtime/worker_prerequisites.py": "07130801b51eba7e42c8578ffaa74ddebf50d7bb314e32fa44dcc8b26a4d60d8",
    "tools/pireus/continuity/runtime/run_in_container.sh": "4a1a5cc6825752c864b27ea6669f380215f35a45f66907ab1a5db743cabafdc1",
    "tools/pireus/continuity/runtime/stage_image.py": "eb4cf2b56f36e170a26e090a22d68d09bb7dba0b1fabca3347753d81066d4565",
    "tools/pireus/continuity/runtime/runtime-lock.json": "9df753aeccbfdf250049b33f963a851921424ee49876eb103080a897894ffa51",
    "tools/pireus/continuity/cycle.py": "9481e316666a8f2f78d378f87b31aba3a1a18fbcfe76bf5ca8721e88d562c2a8",
    "tools/pireus/continuity/pilot.py": "d29da918577852c4450275fa99a2b64af24173b0ddd05c9d0c27d4c9b7c53e97",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw(value):
    return (json.dumps(value, separators=(",", ":")) + "\n").encode()


def series(candidate, control=1000000):
    return "".join(f"{i} {i % 2} {candidate} {control}\n" for i in range(30))


def write_inputs(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "controls/direct.ptx": b".version 9.0\n.target sm_86\n// direct\n",
        "controls/shuffle.ptx": b".version 9.0\n.target sm_86\n// shuffle\n",
        "000.ptx": b".version 9.0\n.target sm_75\n// candidate\n",
    }
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    fixtures = b'{"a_bits":[0],"b_bits":[0],"output_bits":[0]}\n'
    (root / "numeric-fixtures.jsonl").write_bytes(fixtures)
    fixture_sha = hashlib.sha256(fixtures).hexdigest()

    def entry(name, path):
        return {"id": name, "layout": 0, "ptx": path, "ptx_sha256": hashlib.sha256(files[path]).hexdigest()}

    candidates = [
        entry("direct", "controls/direct.ptx"),
        entry("shuffle", "controls/shuffle.ptx"),
        entry("000", "000.ptx"),
    ]
    gpu = {"schema": 1, "fixtures_sha256": fixture_sha, "candidates": candidates}
    bench = {
        "method": "paired-median-ppm-bootstrap4000-v1",
        "blocks": 30, "vectors": 16384, "replays": 32,
        "fixtures_sha256": fixture_sha,
        "controls": candidates[:2],
        "candidates": candidates[2:],
    }
    (root / "gpu-manifest.json").write_text(json.dumps(gpu) + "\n")
    (root / "benchmark-manifest.json").write_text(json.dumps(bench) + "\n")
    return root


def declared_image(path, payload=b"declared-sif"):
    path = Path(path)
    path.write_bytes(payload)
    return {
        "status": "DECLARED",
        "derived_sif_sha256": hashlib.sha256(payload).hexdigest(),
        "sif_bytes": len(payload),
        "sif_path": str(path),
        "definition_sha256": hashlib.sha256((HERE / "pireus-r740-amd64.def").read_bytes()).hexdigest(),
        "independently_verified": False,
    }


class Keys(unittest.TestCase):
    def test_two_gpus_on_one_host_do_not_share_a_directory(self):
        a = keys.observation_dir("timing", "GPU-" + "ab" * 16, 8, 6)
        b = keys.observation_dir("timing", "GPU-" + "cd" * 16, 7, 5)
        self.assertNotEqual(a, b)
        self.assertNotIn("gpuorangefs", a)
        self.assertTrue(a.startswith("timings-gpu-"))
        self.assertTrue(a.endswith("-sm86"))
        self.assertTrue(b.endswith("-sm75"))

    def test_uuid_and_capability_round_trip(self):
        uuid = "GPU-00112233-4455-6677-8899-aabbccddeeff"
        for major, minor, sm in ((8, 6, 86), (7, 5, 75), (12, 1, 121)):
            record = keys.identity_record(uuid, major, minor, "name")
            self.assertEqual(record["sm"], sm)
            self.assertEqual(record["compute_capability"], f"{major}.{minor}")
            self.assertEqual(record["key_scope"], "per-gpu")
            self.assertEqual(keys.device_key(uuid, major, minor), record["key"])

    def test_refuses_a_capability_outside_the_pilot(self):
        with self.assertRaises(ValueError):
            keys.device_key("GPU-" + "ab" * 16, 8, 0)
        with self.assertRaises(ValueError):
            keys.device_key("not-a-uuid", 8, 6)

    def test_cuuuid_object_becomes_the_nvidia_smi_form(self):
        raw = bytes.fromhex("00112233445566778899aabbccddeeff")

        class CUuuid:
            def __init__(self, blob, text):
                self.bytes = blob
                self.text = text

            def __str__(self):
                return self.text

        # PyTorch's __str__ is the dashed hex and is not a str uuid until we prefix it.
        printed = CUuuid(list(raw), "00112233-4455-6677-8899-aabbccddeeff")
        # A binding that only exposes .bytes, with an unusable __str__, still converts.
        opaque = CUuuid(raw, "<_CUuuid object>")
        expected = "GPU-00112233-4455-6677-8899-aabbccddeeff"
        self.assertEqual(keys.canonical_gpu_uuid(printed), expected)
        self.assertEqual(keys.canonical_gpu_uuid(opaque), expected)
        self.assertEqual(keys.canonical_gpu_uuid(raw), expected)
        self.assertEqual(keys.canonical_gpu_uuid(expected.upper()), expected)

        class Integral:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Props:
            uuid = opaque
            major = Integral(8)
            minor = Integral(6)
            name = b"RTX A5000\x00junk"
            total_memory = Integral(256 * 1024 * 1024)

        record = keys.device_identity(Props(), None, 86)
        self.assertEqual(record["gpu_uuid"], expected)
        self.assertEqual(record["gpu"], "RTX A5000")
        self.assertEqual(record["sm"], 86)
        self.assertEqual(record["total_memory_bytes"], 256 * 1024 * 1024)
        self.assertEqual(record["key"], keys.device_key(expected, 8, 6))
        with self.assertRaises(RuntimeError):
            keys.device_identity(Props(), None, 75)
        for bad in (None, "", b"", raw[:15], "GPU-zz", "not-a-uuid", object()):
            with self.assertRaises(ValueError):
                keys.canonical_gpu_uuid(bad)
        with self.assertRaises(ValueError):
            keys.device_identity(Props(), True, 86)

    def test_timing_row_matches_the_spark_grammar(self):
        self.assertEqual(paired_timing.row(1, 940000, 1000000), "1 1 940000 1000000\n")
        spark = (CONTINUITY / "gpu_benchmark.py").read_text()
        self.assertIn('manifest["blocks"]==30', spark.replace(" ", ""))
        self.assertIn("16384", spark)
        self.assertIn('manifest["replays"]==32', spark.replace(" ", ""))
        observe = (HERE / "r740_observe.py").read_text()
        self.assertIn("observation_dir(", observe)
        self.assertIn("device_count() != 1", observe)
        self.assertIn("write_inventory(", observe)
        self.assertIn("append_timing_block(", observe)
        self.assertNotIn("**identity", observe)
        self.assertNotIn("check_pair", observe)
        self.assertNotIn("timings-\"+socket.gethostname()", observe)

    def test_inventory_and_timing_records_accept_a_cuuuid(self):
        class CUuuid:
            def __init__(self, blob, text):
                self.bytes = blob
                self.text = text

            def __str__(self):
                return self.text

        class Integral:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Version:
            def __str__(self):
                return "13.2"

        def props(raw, text, major, minor, name):
            class Props:
                uuid = CUuuid(raw, text)
            Props.major = Integral(major)
            Props.minor = Integral(minor)
            Props.name = name
            Props.total_memory = Integral(8 * 1024 * 1024)
            return Props()

        a5000 = props(
            bytes.fromhex("00112233445566778899aabbccddeeff"),
            "<_CUuuid object>", 8, 6, b"NVIDIA RTX A5000\x00pad",
        )
        rtx8000 = props(
            list(bytes.fromhex("aabbccddeeff00112233445566778899")),
            "aabbccdd-eeff-0011-2233-445566778899", 7, 5, b"Quadro RTX 8000\x00",
        )
        identities = (
            keys.device_identity(a5000, None, 86),
            keys.device_identity(rtx8000, None, 75),
        )
        self.assertEqual(identities[0]["gpu_uuid"], "GPU-00112233-4455-6677-8899-aabbccddeeff")
        self.assertEqual(identities[1]["gpu_uuid"], "GPU-aabbccdd-eeff-0011-2233-445566778899")
        self.assertEqual([item["sm"] for item in identities], [86, 75])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for identity, major, minor in (
                (identities[0], 8, 6), (identities[1], 7, 5),
            ):
                probed = root / keys.observation_dir("probe", identity["gpu_uuid"], major, minor)
                timed = root / keys.observation_dir("timing", identity["gpu_uuid"], major, minor)
                probed.mkdir()
                timed.mkdir()
                messy = dict(identity)
                messy["total_memory_bytes"] = Integral(identity["total_memory_bytes"])
                inventory = keys.write_inventory(
                    probed, messy, **keys.probe_record_fields("24827", "r740", Version()),
                )
                text = inventory.read_text()
                record = json.loads(text)
                self.assertEqual(text.count('"key_scope"'), 1)
                self.assertEqual(record["key_scope"], "per-gpu")
                self.assertEqual(record["cuda"], "13.2")
                self.assertEqual(record["total_memory_bytes"], identity["total_memory_bytes"])
                self.assertEqual(record["gpu_uuid"], identity["gpu_uuid"])
                self.assertEqual(record["observation_only"], True)
                timing_inventory = keys.write_inventory(
                    timed, identity,
                    **keys.timing_record_fields("24828", "r740", "13.2", "2.15.0"),
                )
                timing_record = json.loads(timing_inventory.read_text())
                self.assertEqual(timing_record["key_scope"], "per-gpu")
                self.assertEqual(timing_record["sm"], identity["sm"])
                self.assertEqual(timing_record["vectors"], paired_timing.VECTORS)
                self.assertEqual(timing_record["layout_conversion_included"], False)
                self.assertEqual(timing_record["torch"], "2.15.0")
                trial = timed / "000.direct.tsv"
                with trial.open("w") as log:
                    for block in range(paired_timing.BLOCKS):
                        paired_timing.append_timing_block(
                            log, block, paired_timing.duration_ns(Integral(0.5)),
                            paired_timing.duration_ns(1.0),
                        )
                lines = trial.read_text().splitlines()
                self.assertEqual(len(lines), paired_timing.BLOCKS)
                self.assertEqual(lines[0], "0 0 500000 1000000")
                self.assertEqual(lines[1], "1 1 500000 1000000")
                parsed = json.loads(keys.log_line(
                    stage="PAIRED_TIMING_OBSERVED", sm=Integral(identity["sm"]),
                    key_scope=identity["key_scope"], blocks=paired_timing.BLOCKS,
                ))
                self.assertEqual(parsed["sm"], identity["sm"])
                self.assertEqual(parsed["key_scope"], "per-gpu")
                marker = keys.write_marker(root, "probe", identity, "24827")
                marker_record = json.loads(marker.read_text())
                self.assertEqual(marker_record["key_scope"], "per-gpu")
                self.assertEqual(marker_record["phase"], "probe")
                self.assertEqual(marker_record["sm"], identity["sm"])
            not_json = dict(identities[0])
            not_json["total_memory_bytes"] = float("nan")
            with self.assertRaises(ValueError):
                keys.observation_document(not_json)
            foreign = dict(identities[0])
            foreign["key_scope"] = "host"
            with self.assertRaises(RuntimeError) as caught:
                keys.write_inventory(root / "rejected", foreign, **keys.probe_record_fields("1", "n", "13.2"))
            self.assertIn("per-gpu", str(caught.exception))
            self.assertFalse((root / "rejected" / "inventory.json").exists())
            clash = dict(identities[0])
            clash["job"] = "other"
            with self.assertRaises(RuntimeError):
                keys.write_inventory(root / "clash", clash, **keys.probe_record_fields("24827", "r740", "13.2"))


class Plans(unittest.TestCase):
    def test_phase_a_is_round_zero_budget_eight(self):
        rows = proposals.phase_a()
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["round"] == 0 and row["index"] == i for i, row in enumerate(rows)))
        cycle = (CONTINUITY / "cycle.py").read_text()
        for fragment in (
            "lane_stride=1 + 2 * (index % 8)",
            'lane_offset=(index // 8 + manifest["round"]) % 16',
            "load=(index // 2) % 2, layout=index % 2",
            "unroll=(1, 2, 4, 8, 16)[index % 5]",
        ):
            self.assertIn(fragment, cycle)
        self.assertEqual(rows[2]["lane_stride"], 5)
        self.assertEqual(rows[2]["load"], 1)
        self.assertEqual(rows[4]["unroll"], 16)

    def test_phase_b_is_thirty_two_across_three_rounds_and_does_not_repeat_phase_a(self):
        rows = proposals.phase_b()
        self.assertEqual(len(rows), 32)
        self.assertEqual({row["round"] for row in rows}, {0, 1, 2})
        phase_a = {(row["round"], row["index"]) for row in proposals.phase_a()}
        phase_b = {(row["round"], row["index"]) for row in rows}
        self.assertTrue(phase_a.isdisjoint(phase_b))
        lanes = {(row["lane_stride"], row["lane_offset"], row["load"], row["layout"], row["unroll"]) for row in rows + proposals.phase_a()}
        self.assertEqual(len(lanes), 40)


class Launcher(unittest.TestCase):
    def test_review_commands_pin_the_node_one_gres_and_memory(self):
        plan = launch.build_plan(launch.DEFAULT_SCRATCH, REVIEW_NAME)
        self.assertEqual([step["gres"] for step in plan["steps"]], [
            "gpu:a5000:1", "gpu:quadro_rtx_8000:1", "gpu:a5000:1", "gpu:quadro_rtx_8000:1"])
        self.assertEqual([step["phase"] for step in plan["steps"]], ["probe", "probe", "timing", "timing"])
        doc = (HERE / "R740_PILOT.md").read_text()
        locked = json.loads((HERE / "r740-runtime-lock.json").read_text())
        emitted = launch.build_plan(launch.DEFAULT_SCRATCH, REVIEW_NAME, image_token_value=launch.image_token(locked))
        for step in emitted["steps"]:
            self.assertIn(locked["derived_sif_sha256"], step["script"])
            self.assertNotIn("UNPINNED", step["script"])
            self.assertIn(locked["derived_sif_sha256"], doc)
        self.assertEqual(doc.count("# PAUSE LLM services"), 1)
        self.assertIn("Probes above stay with the services up", doc)
        self.assertIn("Do this even if either timing command failed", doc)
        self.assertLess(doc.index("--step probe-sm75"), doc.index("# PAUSE LLM services"))
        self.assertLess(doc.index("# PAUSE LLM services"), doc.index("--step timing-sm86"))
        self.assertLess(doc.index("--step timing-sm75"), doc.index("# RESUME LLM services"))
        for step in plan["steps"]:
            argv = step["argv"]
            self.assertEqual(argv[0], "srun")
            self.assertEqual(argv[argv.index("-p") + 1], "gpu-orangefs")
            self.assertEqual(argv[argv.index("-w") + 1], "gpuorangefs-multi-r740-proxmox")
            self.assertIn("--nodes=1", argv)
            self.assertIn("--ntasks=1", argv)
            self.assertIn("--mem=16G", argv)
            self.assertNotIn("--exclusive", argv)
            self.assertEqual(argv.count("--gres=" + step["gres"]), 1)
            self.assertNotIn("sbatch", argv)
            self.assertFalse(step["sbatch_submitted"])
            self.assertIn("#SBATCH --no-requeue", step["sbatch_review"])
            self.assertIn("submitted: false", step["sbatch_review"])
            self.assertIn(" ".join(argv[:12]), doc)
            self.assertIn("--job-name=" + step["job_name"], doc)
        self.assertFalse(plan["check_pair"])
        self.assertFalse(plan["spark_lease"])
        self.assertFalse(plan["host_fence"])
        source = (HERE / "r740_launch.py").read_text()
        self.assertNotIn("check_pair(", source)
        self.assertNotIn("import preflight", source)
        self.assertNotIn("benchmark_pair", source)
        self.assertNotIn("tools/cluster", source)

    def test_memory_cap_and_fresh_directory(self):
        launch.parse_mem("16G")
        launch.parse_mem("64G")
        for bad in ("65G", "110G", "0G", "16M", "016G", "", "16"):
            with self.assertRaises(ValueError):
                launch.parse_mem(bad)
        with tempfile.TemporaryDirectory() as directory:
            name = launch.new_run_name("20261117T120000Z", 7, "0123456789abcdef")
            first = launch.allocate_run(directory, name)
            with self.assertRaises(FileExistsError):
                launch.allocate_run(directory, name)
            self.assertTrue(str(first).startswith(directory))

    def test_declared_sif_is_hashed_before_srun(self):
        lock = json.loads((HERE / "r740-runtime-lock.json").read_text())
        self.assertEqual(lock["status"], "DECLARED")
        self.assertFalse(lock["independently_verified"])
        self.assertEqual(lock["derived_sif_sha256"], "e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25")
        self.assertEqual(lock["sif_bytes"], 3270336512)
        self.assertEqual(lock["definition_sha256"], "b8c273f00197c0046e802b86ef17ea016e0216c2f214c00e8c043a9f25f06f9a")
        self.assertEqual(hashlib.sha256((HERE / "pireus-r740-amd64.def").read_bytes()).hexdigest(), lock["definition_sha256"])
        self.assertNotIn(b"\r", (HERE / "pireus-r740-amd64.def").read_bytes())
        self.assertEqual(lock["apptainer"]["rpm"]["sha256"], "db786cb78a8bba81347e46ca676b1ad7409e4cd7225f2ee48e2915bdfedd535c")
        self.assertEqual(lock["apptainer"]["rpm"]["bytes"], 50171128)
        self.assertEqual(lock["apptainer"]["install_script"]["sha256"], "a097956eafa6ab3dd843ee429d0b774ed029753bfadf013b8046556228fac6f2")
        self.assertEqual(lock["apptainer"]["prefix"], "/scratch/pireus/r740/apptainer-1.5.3")
        self.assertEqual(lock["apptainer"]["reported_version"], "apptainer version 1.5.3-3.el8")
        self.assertNotIn("c85138e9", json.dumps(lock))
        calls = []

        def boom(argv):
            calls.append(argv)
            raise AssertionError("submitted")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError) as caught:
                launch.execute(directory, REVIEW_NAME, boom, image_lock=lock)
            self.assertIn("SIF is missing", str(caught.exception))
            self.assertEqual(calls, [])
            self.assertFalse((Path(directory) / "runs").exists())
            image = declared_image(Path(directory) / "image.sif")
            image["derived_sif_sha256"] = "ab" * 32
            with self.assertRaises(RuntimeError) as caught:
                launch.execute(directory, REVIEW_NAME, boom, image_lock=image)
            self.assertIn("sha256 does not match", str(caught.exception))
            self.assertEqual(calls, [])
            image = declared_image(Path(directory) / "image.sif")
            image["sif_bytes"] = 1
            with self.assertRaises(RuntimeError) as caught:
                launch.execute(directory, REVIEW_NAME, boom, image_lock=image)
            self.assertIn("size does not match", str(caught.exception))
            self.assertEqual(calls, [])

    def test_pinned_run_uses_one_directory_and_does_not_resubmit(self):
        with tempfile.TemporaryDirectory() as directory:
            image = declared_image(Path(directory) / "image.sif")
            bundle = write_inputs(Path(directory) / "bundle")
            calls = []

            def ok(argv):
                calls.append(argv)
                plan = launch.build_plan(directory, REVIEW_NAME, image_token_value=image["derived_sif_sha256"])
                marker = Path(plan["steps"][len(calls) - 1]["marker"])
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}\n")
                return 0

            root = launch.execute(directory, REVIEW_NAME, ok, image_lock=image, inputs=bundle)
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0][calls[0].index("--gres=gpu:a5000:1")], "--gres=gpu:a5000:1")
            self.assertIn("--gres=gpu:quadro_rtx_8000:1", calls[1])
            self.assertIn(image["derived_sif_sha256"], calls[0][-1])
            self.assertEqual((root / "bin" / "gpu_probe.py").read_bytes(), (CONTINUITY / "gpu_probe.py").read_bytes())
            with self.assertRaises(FileExistsError):
                launch.execute(directory, REVIEW_NAME, ok, image_lock=image, inputs=bundle)

            failed = []

            def fail(argv):
                failed.append(argv)
                return 1

            name = "20261117T120001Z-2-fedcba9876543210"
            with self.assertRaises(RuntimeError):
                launch.execute(directory, name, fail, image_lock=image, inputs=bundle)
            self.assertEqual(len(failed), 1)
            again = []
            plan = launch.build_plan(directory, name, image_token_value=image["derived_sif_sha256"])
            with self.assertRaises(RuntimeError) as caught:
                launch.run_steps(Path(directory) / "runs" / name, plan["steps"], lambda argv: again.append(argv) or 0)
            self.assertIn("resubmit", str(caught.exception))
            self.assertEqual(again, [])

    def test_staged_inputs_are_readable_before_srun_and_a_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            image = declared_image(Path(directory) / "image.sif", b"staged-sif")
            bundle = write_inputs(Path(directory) / "bundle")
            seen = {}

            def inspect(argv):
                root = Path(directory) / "runs" / REVIEW_NAME
                probe_manifest, probe_fixtures, probe_digest, probe_ptx = run_inputs.require_probe_inputs(root)
                timing_manifest, timing_fixtures, timing_digest, timing_ptx = run_inputs.require_timing_inputs(root)
                self.assertEqual(probe_fixtures, (bundle / "numeric-fixtures.jsonl").read_bytes())
                self.assertEqual(probe_fixtures, timing_fixtures)
                self.assertEqual(probe_digest, timing_digest)
                self.assertEqual(probe_digest, probe_manifest["fixtures_sha256"])
                self.assertEqual(timing_digest, timing_manifest["fixtures_sha256"])
                referenced = probe_ptx | timing_ptx
                self.assertEqual(set(referenced), {
                    "controls/direct.ptx", "controls/shuffle.ptx", "000.ptx"})
                for name, digest in referenced.items():
                    payload = (root / name).read_bytes()
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
                    self.assertEqual(payload, (bundle / name).read_bytes())
                seen["argv"] = argv
                plan = launch.build_plan(directory, REVIEW_NAME, image_token_value=image["derived_sif_sha256"])
                for step in plan["steps"]:
                    marker = Path(step["marker"])
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text("{}\n")
                return 0

            root = launch.execute(directory, REVIEW_NAME, inspect, image_lock=image, inputs=bundle)
            self.assertIn("argv", seen)
            self.assertTrue((root / "input-inventory.json").is_file())
            observe = (HERE / "r740_observe.py").read_text()
            probe_body = observe.split("def probe", 1)[1].split("def timing", 1)[0]
            timing_body = observe.split("def timing", 1)[1].split("def main", 1)[0]
            self.assertLess(probe_body.index("require_probe_inputs"), probe_body.index("Driver()"))
            self.assertLess(timing_body.index("require_timing_inputs"), timing_body.index("Driver()"))

            calls = []

            def boom(argv):
                calls.append(argv)
                raise AssertionError("submitted")

            missing = write_inputs(Path(directory) / "missing")
            (missing / "000.ptx").unlink()
            with self.assertRaises(ValueError):
                launch.execute(directory, "20261117T120002Z-3-0011223344556677", boom,
                               image_lock=image, inputs=missing)
            self.assertEqual(calls, [])
            self.assertFalse((Path(directory) / "runs" / "20261117T120002Z-3-0011223344556677").exists())

            mismatched = write_inputs(Path(directory) / "mismatched")
            (mismatched / "000.ptx").write_bytes(b"not the recorded ptx\n")
            with self.assertRaises(ValueError):
                launch.execute(directory, "20261117T120003Z-4-0011223344556677", boom,
                               image_lock=image, inputs=mismatched)
            self.assertEqual(calls, [])

            with self.assertRaises(RuntimeError) as caught:
                launch.execute(directory, "20261117T120004Z-5-0011223344556677", boom, image_lock=image)
            self.assertIn("inputs are missing", str(caught.exception))
            self.assertEqual(calls, [])

    def test_cross_architecture_report_does_not_grant_a_gain(self):
        report = launch.descriptive_report([
            {"sm": 86, "gain_gate": "PASS"},
            {"sm": 75, "gain_gate": "PASS"},
        ])
        self.assertFalse(report["gain_counts"])
        self.assertIsNone(report["aggregate_gain_gate"])
        with self.assertRaises(ValueError):
            launch.descriptive_report([{"sm": 86, "gain_gate": "PASS"}, {"sm": 86, "gain_gate": "PASS"}])

    def test_cli_run_refuses_before_srun_when_the_sif_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(HERE / "r740_launch.py"), "run",
                 "--name", REVIEW_NAME, "--scratch", directory],
                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to submit", result.stderr)
        self.assertNotIn("srun:", result.stderr)

    def test_materialize_modes_leave_sm121_on_the_historical_spelling(self):
        self.assertEqual(launch.materialize_argv("engine", "c", "p", 121)[-1], "ptx")
        self.assertEqual(launch.materialize_argv("engine", "c", "p", 86)[-1], "ptx-sm86")
        self.assertEqual(launch.materialize_argv("engine", "c", "p", 75)[-1], "ptx-sm75")

    def test_container_script_is_amd64_and_fails_closed(self):
        text = (HERE / "run_in_container_r740.sh").read_text()
        for banned in ("aarch64", "inkling-spark", "ld-linux-aarch64", "gpu:gb10", "sm_121"):
            self.assertNotIn(banned, text)
        self.assertIn("pireus-r740-amd64.sif", text)
        self.assertIn("exit 2", text)


class FrozenRule(unittest.TestCase):
    def test_source_hash_matches_the_lock_and_no_r740_data_is_recorded(self):
        lock = json.loads((HERE / "arch_gain_rule.lock.json").read_text())
        launch.assert_rule_frozen(lock)
        self.assertEqual(lock["source_sha256"], sha(HERE / "arch_gain_decision.sio"))
        self.assertEqual(lock["r740_observation_files"], 0)
        self.assertEqual(lock["thresholds"]["median_gain_ppm"], 50000)
        spark = (CONTINUITY / "benchmark_decision.sio").read_text()
        arch = (HERE / "arch_gain_decision.sio").read_text()
        for needle in (
            "var state: i64 = 20260906",
            "state = (state * 48271) % 2147483647",
            "while repetition < 4000",
            "low: boot[99], high: boot[3899]",
            "result.median < 50000 || result.low <= 0",
            "((values[3] - values[2]) * 1000000) / values[3]",
        ):
            self.assertIn(needle, spark)
            self.assertIn(needle, arch)
        self.assertIn("arg_count() != 4", spark)
        self.assertIn("arg_count() != 2", arch)


class PhaseABundle(unittest.TestCase):
    def test_imported_bytes_match_the_baseline_commit(self):
        origin = phase_a.verify_imported_sources()
        self.assertEqual(origin["origin_commit"], "1a82c417e55c687748ee396d91457ec4b3fe15b4")
        self.assertEqual(origin["origin_branch"], "codex/pireus-inkling-cycle-20260906")
        self.assertEqual(origin["sources"]["proposals/000.proposal.json"]["sha256"],
                         origin["sources"]["controls/direct.proposal.json"]["sha256"])
        self.assertEqual(len(origin["sources"]), 12)
        self.assertEqual(origin["sm121_ptx_sha256"]["shuffle"],
                         "c55daadb94b2050ca88047316f5f0317ccbe3fe48744b26dae13ad7afed659b3")

    def test_bundle_is_deterministic_and_one_step_respects_probe_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "bundle-a"
            second = Path(directory) / "bundle-b"
            phase_a.bundle(first)
            phase_a.bundle(second)
            for arch in ("sm86", "sm75"):
                run_inputs.require_run_inputs(first / arch)
                for name in ("gpu-manifest.json", "benchmark-manifest.json", "numeric-fixtures.jsonl", "000.ptx"):
                    self.assertEqual((first / arch / name).read_bytes(), (second / arch / name).read_bytes())
            image = declared_image(Path(directory) / "image.sif")
            scratch = Path(directory) / "scratch"
            root = launch.prepare_run(scratch, REVIEW_NAME, first)
            self.assertTrue((root / "sm86" / "gpu-manifest.json").is_file())
            self.assertTrue((root / "sm75" / "000.ptx").is_file())
            calls = []

            def run(step, runner):
                def wrapped(argv):
                    calls.append(argv)
                    self.assertEqual(argv[0], "srun")
                    self.assertEqual([item for item in argv if item.startswith("--gres=")], [
                        "--gres=" + ("gpu:a5000:1" if step.endswith("86") else "gpu:quadro_rtx_8000:1")])
                    plan = launch.build_plan(scratch, REVIEW_NAME, image_token_value=image["derived_sif_sha256"])
                    chosen = next(item for item in plan["steps"] if item["name"] == step)
                    self.assertEqual(argv, chosen["argv"])
                    Path(chosen["marker"]).parent.mkdir(parents=True, exist_ok=True)
                    Path(chosen["marker"]).write_text("{}\n")
                    return runner()
                return wrapped

            launch.run_step(scratch, REVIEW_NAME, "probe-sm86", run("probe-sm86", lambda: 0), image_lock=image)
            self.assertEqual(len(calls), 1)
            missed = []
            with self.assertRaises(RuntimeError) as caught:
                launch.run_step(scratch, REVIEW_NAME, "timing-sm75", lambda argv: missed.append(argv), image_lock=image)
            self.assertIn("successful probe", str(caught.exception))
            self.assertEqual(missed, [])
            launch.run_step(scratch, REVIEW_NAME, "timing-sm86", run("timing-sm86", lambda: 0), image_lock=image)
            self.assertEqual(len(calls), 2)
            with self.assertRaises(RuntimeError):
                launch.run_step(scratch, REVIEW_NAME, "probe-sm75", run("probe-sm75", lambda: 1), image_lock=image)
            blocked = []
            with self.assertRaises(RuntimeError) as caught:
                launch.run_step(scratch, REVIEW_NAME, "timing-sm75", lambda argv: blocked.append(argv), image_lock=image)
            self.assertIn("did not succeed", str(caught.exception))
            self.assertEqual(blocked, [])
            self.assertEqual(run_inputs.data_root(root, 86), root / "sm86")
            self.assertEqual(run_inputs.data_root(root / "flat", 86).name, "flat")

def receipt_v1(stdout):
    """Receipt contract v2 adds the atlas block to kind=2 receipts only.

    The frozen digests were taken under contract v1. A schema-2 receipt maps
    back to v1 by restoring schema 1 and the v1 novelty/classifier pair; every
    other byte must be unchanged, so the frozen digest still binds the rest.
    """
    text = stdout.decode()
    if not text.startswith('{"schema":2,'):
        return stdout
    receipt = json.loads(text)
    assert receipt["kind"] == "operator" and receipt["novelty"] in ("N0", "N1", "N2"), receipt
    assert receipt["claim_ready"] is False and receipt["scientific_novelty"] is False, receipt
    head, tail = text.index(',"novelty":'), text.index(',"fp_parity"')
    return ('{"schema":1,' + text[len('{"schema":2,'):head] +
            ',"novelty":"NOT_CLASSIFIED","classifier":"eisa_h.pireus_operator_atlas.v1"' + text[tail:]).encode()

class Engines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="pireus-r740-"))
        souc = ROOT / "bin" / "souc"
        cls.admission = cls.work / "admission.elf"
        cls.spark = cls.work / "spark-gain.elf"
        cls.arch = cls.work / "arch-gain.elf"
        for source, dest in (
            (CONTINUITY / "admission.sio", cls.admission),
            (CONTINUITY / "benchmark_decision.sio", cls.spark),
            (HERE / "arch_gain_decision.sio", cls.arch),
        ):
            run = subprocess.run([str(souc), str(source), str(dest)], capture_output=True, text=True, timeout=180)
            if run.returncode != 0:
                raise AssertionError(run.stderr[-2000:])
            dest.chmod(0o755)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_proposals_dir_bundle_admits_materializes_and_drops(self):
        import inbox_bundle
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "inbox"
            src.mkdir()
            context_sha = sha(HERE / "phase-a" / "context.json")
            base = json.loads((HERE / "phase-a" / "proposals" / "004.proposal.json").read_text())
            self.assertEqual(base["context"], context_sha)
            fresh = dict(base, lane_offset=1)
            (src / "x-new.proposal.json").write_bytes(raw(fresh))
            (src / "x-again.proposal.json").write_bytes(raw(base))
            (src / "x-stale.proposal.json").write_bytes(raw(dict(fresh, context="0" * 64)))
            out = Path(directory) / "bundle"
            inbox_bundle.bundle(out, self.admission, src)
            summary = json.loads((out / "PROPOSALS.json").read_text())
            rows = {row["id"]: row for row in summary["proposals"]}
            self.assertEqual(summary["admitted"], ["x-new"])
            self.assertEqual(rows["x-stale"]["decision"], "REFUSE")
            self.assertEqual(rows["x-again"]["decision"], "ADMIT")
            self.assertEqual(rows["x-again"]["duplicate_of"], ["phase-a:004"])
            self.assertEqual(summary["context_sha256"], context_sha)
            self.assertEqual(summary["engine_sha256"], sha(self.admission))
            for arch in ("sm86", "sm75"):
                inputs = run_inputs.require_run_inputs(out / arch)
                bench = inputs["benchmark_manifest"]
                self.assertEqual([c["id"] for c in bench["controls"]], ["direct", "shuffle"])
                self.assertEqual([c["id"] for c in bench["candidates"]], ["x-new"])
                self.assertEqual((out / arch / "controls" / "direct.ptx").read_bytes(),
                                 (HERE / "phase-a" / "ptx" / arch / "direct.ptx").read_bytes())
                self.assertEqual(sha(out / arch / "x-new.ptx"), rows["x-new"]["ptx_sha256"][arch])
            self.assertTrue((out / "admission" / "x-stale.receipt.json").is_file())
            with self.assertRaises(ValueError):
                inbox_bundle.bundle(Path(directory) / "none", None, src)
            scratch = Path(directory) / "scratch"
            root = launch.prepare_run(scratch, REVIEW_NAME, out, str(Path(directory) / "runs-agent"))
            self.assertTrue((root / "proposals-bundle.json").is_file())
            self.assertTrue((root / "admission" / "x-new.receipt.json").is_file())

    def test_proposals_dir_bundle_takes_kind2_operators(self):
        import inbox_bundle
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "inbox"
            src.mkdir()
            base = json.loads((HERE / "phase-a" / "proposals" / "004.proposal.json").read_text())
            (src / "op-a.proposal.json").write_bytes(raw(dict(base, kind=2, phase=35)))
            (src / "op-b.proposal.json").write_bytes(raw(dict(base, kind=2, phase=36077)))
            (src / "op-bad.proposal.json").write_bytes(raw(dict(base, kind=2, phase=65536)))
            (src / "op-nophase.proposal.json").write_bytes(raw(dict(base, kind=2)))
            (src / "op-fixed.proposal.json").write_bytes(raw(dict(base, kind=2, phase=36077, unroll=16, sign_mode=1)))
            (src / "op-fixed-rolled.proposal.json").write_bytes(raw(dict(base, kind=2, phase=36077, unroll=4, sign_mode=1)))
            (src / "op-packed.proposal.json").write_bytes(raw(dict(base, kind=2, phase=36077, pack=1)))
            (src / "op-packed-bad.proposal.json").write_bytes(raw(dict(base, kind=2, phase=36077, pack=2)))
            (src / "op-staged.proposal.json").write_bytes(raw(dict(base, kind=2, phase=36077, load=2, layout=1, pack=1, vector_load=1)))
            (src / "op-staged-aos.proposal.json").write_bytes(raw(dict(base, kind=2, phase=36077, load=2, layout=0, pack=1, vector_load=1)))
            (src / "op-wide.proposal.json").write_bytes(raw(dict(base, kind=2, phase=36077, load=2, block_threads=64)))
            (src / "op-wide-pack.proposal.json").write_bytes(raw(dict(base, kind=2, phase=36077, pack=1, block_threads=32)))
            (src / "op-wide-bad.proposal.json").write_bytes(raw(dict(base, kind=2, phase=36077, block_threads=96)))
            out = Path(directory) / "bundle"
            inbox_bundle.bundle(out, self.admission, src)
            summary = json.loads((out / "PROPOSALS.json").read_text())
            rows = {row["id"]: row for row in summary["proposals"]}
            self.assertEqual(summary["admitted"], ["op-a", "op-b", "op-fixed", "op-packed", "op-staged", "op-wide"])
            self.assertEqual(rows["op-wide"]["block_threads"], 64)
            self.assertEqual(rows["op-wide-pack"]["reason"], "BLOCK_THREADS_PACK")
            self.assertEqual(rows["op-wide-bad"]["reason"], "BLOCK_THREADS")
            import r740_wide_launch as wide
            for arch in ("sm86", "sm75"):
                entries = {c["id"]: c for c in json.loads((out / arch / "gpu-manifest.json").read_text())["candidates"]}
                self.assertEqual(entries["op-wide"]["block_threads"], 64)
                for name, entry in entries.items():
                    if name != "op-wide":
                        self.assertNotIn("block_threads", entry, name)
                    self.assertEqual(wide.geometry(entry, (out / arch / entry["ptx"]).read_bytes()),
                                     64 if name == "op-wide" else 16, name)
            self.assertEqual(rows["op-staged"]["vector_load"], 1)
            self.assertEqual(rows["op-staged-aos"]["reason"], "VECTOR_LOAD_LAYOUT")
            self.assertEqual(rows["op-packed"]["pack"], 1)
            self.assertNotIn("pack", rows["op-fixed"])
            self.assertEqual(rows["op-packed-bad"]["reason"], "PACK")
            self.assertEqual(rows["op-fixed"]["sign_mode"], 1)
            self.assertNotIn("sign_mode", rows["op-a"])
            self.assertEqual(rows["op-fixed-rolled"]["reason"], "SIGN_MODE_UNROLL")
            self.assertEqual((rows["op-a"]["kind"], rows["op-a"]["phase"]), (2, 35))
            self.assertEqual(rows["op-bad"]["reason"], "PHASE_RANGE")
            self.assertEqual(rows["op-nophase"]["reason"], "KIND_MASK")
            for arch in ("sm86", "sm75"):
                lowering = (HERE / "phase-a" / "ptx" / arch / "004.ptx").read_bytes()
                twisted = (out / arch / "op-a.ptx").read_bytes()
                self.assertNotEqual(twisted, lowering)
                strip = lambda text: [line for line in text.splitlines() if b"pireus_signs[16]" not in line]
                self.assertEqual(strip(twisted), strip(lowering))

    def context_and_proposal(self, kind=1, phase=None, plan=None):
        context = dict(schema=1, target=701202, dimension=16, precision=64, order=1, fma=0,
                       capabilities=55, lane_width=32, facts_state=1)
        cb = raw(context)
        proposal = dict(schema=1, target=701202, dimension=16, precision=64, order=1, fma=0,
                        kind=kind, lane_stride=1, lane_offset=0, load=1, layout=0, unroll=1,
                        context=hashlib.sha256(cb).hexdigest())
        if phase is not None:
            proposal["phase"] = phase
        if plan:
            proposal.update(plan)
        return cb, raw(proposal)

    def emit(self, mode, kind=1, phase=None, plan=None, target=None):
        cb, pb = self.context_and_proposal(kind, phase, plan)
        if target is not None:
            body = json.loads(pb)
            body["target"] = target
            pb = raw(body)
        cpath = self.work / "context.json"
        ppath = self.work / "proposal.json"
        cpath.write_bytes(cb)
        ppath.write_bytes(pb)
        argv = [str(self.admission), str(cpath), str(ppath)] + ([mode] if mode else [])
        return subprocess.run(argv, capture_output=True, timeout=30)

    def test_sm121_bytes_match_the_prechange_engine(self):
        kind1 = self.emit("ptx")
        self.assertEqual(kind1.returncode, 0)
        self.assertEqual(hashlib.sha256(kind1.stdout).hexdigest(), SM121_KIND1)
        twisted = self.emit("ptx", kind=2, phase=1128)
        self.assertEqual(hashlib.sha256(twisted.stdout).hexdigest(), SM121_PHASE1128)
        self.assertIn(b".target sm_121\n", kind1.stdout)
        for index, digest in enumerate(SM121_DET):
            plan = {key: proposals.phase_a()[index][key]
                    for key in ("lane_stride", "lane_offset", "load", "layout", "unroll")}
            # Phase A proposals carry context before the lane fields, as cycle.py's template does.
            context = dict(schema=1, target=701202, dimension=16, precision=64, order=1, fma=0,
                           capabilities=55, lane_width=32, facts_state=1)
            cb = raw(context)
            proposal = dict(schema=1, target=701202, dimension=16, precision=64, order=1, fma=0,
                            kind=1, context=hashlib.sha256(cb).hexdigest(), **plan)
            cpath = self.work / "context.json"
            ppath = self.work / "proposal.json"
            cpath.write_bytes(cb)
            ppath.write_bytes(raw(proposal))
            ptx = subprocess.run([str(self.admission), str(cpath), str(ppath), "ptx"], capture_output=True, timeout=30)
            receipt = subprocess.run([str(self.admission), str(cpath), str(ppath)], capture_output=True, timeout=30)
            self.assertEqual(hashlib.sha256(ptx.stdout).hexdigest(), digest, index)
            self.assertEqual(hashlib.sha256(receipt.stdout).hexdigest(), RECEIPT[index], index)

    def test_phase_a_engine_matches_the_vendored_ptx_and_the_baseline_sm121(self):
        phase_a.check_engine(self.admission)

    def test_new_targets_differ_only_in_the_target_line(self):
        base = self.emit("ptx").stdout
        for mode, token in (("ptx-sm86", b"sm_86"), ("ptx-sm75", b"sm_75")):
            got = self.emit(mode).stdout
            self.assertEqual(got, base.replace(b"sm_121", token))
            self.assertEqual(got.count(token), 1)
            self.assertNotIn(b"sm_121", got)
            self.assertTrue(got.startswith(b".version 9.0\n"))

    def test_admission_policy_is_unchanged(self):
        refused = self.emit("ptx-sm86", target=999)
        receipt = json.loads(refused.stdout)
        self.assertEqual(refused.returncode, 1)
        self.assertEqual(receipt["reason"], "TARGET")
        self.assertNotIn(b".target", refused.stdout)
        mode = self.emit("ptx-sm80")
        self.assertEqual(json.loads(mode.stdout)["reason"], "MODE")
        self.assertEqual(json.loads(self.emit("PTX").stdout)["reason"], "MODE")

    def test_absent_or_zero_extensions_keep_the_b001_to_b004_bytes(self):
        frozen = json.loads((HERE / "frozen_inbox_outputs.json").read_text())["outputs"]
        self.assertEqual(len(frozen), 56)
        for batch in ("b001", "b002", "b003", "b004"):
            self.assertGreater(sum(1 for name in frozen if "-" + batch + "-" in name), 0, batch)
        context = HERE / "phase-a" / "context.json"
        zero = self.work / "zero.proposal.json"
        optional = ("sign_mode", "pack", "vector_load")
        for name, modes in sorted(frozen.items()):
            body = json.loads((CONTINUITY / "inbox" / name).read_text())
            self.assertIn(body["load"], (0, 1), name)
            self.assertNotIn("vector_load", body)
            if "-b003-" in name:
                self.assertEqual(body["sign_mode"], 1)
            elif "-b004-" not in name:
                self.assertNotIn("sign_mode", body)
            self.assertEqual("pack" in body, "-b004-" in name)
            absent = [key for key in optional if key not in body]
            explicit = [dict(body, **{key: 0}) for key in absent] + [dict(body, **{key: 0 for key in absent})]
            for mode, digest in modes.items():
                argv = [str(self.admission), str(context), str(CONTINUITY / "inbox" / name)]
                argv += [] if mode == "receipt" else [mode]
                out = subprocess.run(argv, capture_output=True, timeout=30).stdout
                self.assertEqual(hashlib.sha256(receipt_v1(out) if mode == "receipt" else out).hexdigest(),
                                 digest, (name, mode))
                if mode == "receipt":
                    continue
                for variant in explicit:
                    zero.write_bytes(raw(variant))
                    got = subprocess.run([str(self.admission), str(context), str(zero), mode],
                                         capture_output=True, timeout=30)
                    self.assertEqual(hashlib.sha256(got.stdout).hexdigest(), digest, (name, mode, variant))

    def test_sign_mode_is_admitted_only_fully_unrolled(self):
        full = dict(unroll=16, sign_mode=1)
        for kind, phase in ((1, None), (2, 36077)):
            receipt = json.loads(self.emit(None, kind, phase, full).stdout)
            self.assertEqual(receipt["decision"], "ADMIT", receipt)
            self.assertEqual(receipt["sign_mode"], 1)
            self.assertGreaterEqual(receipt["plan_id"], 65536)
            plain = json.loads(self.emit(None, kind, phase, dict(unroll=16)).stdout)
            self.assertNotIn("sign_mode", plain)
            self.assertEqual(receipt["plan_id"] - 65536, plain["plan_id"])
            self.assertEqual(receipt["tensor_sha256"], plain["tensor_sha256"])
        for plan, reason in ((dict(unroll=8, sign_mode=1), "SIGN_MODE_UNROLL"),
                             (dict(unroll=1, sign_mode=1), "SIGN_MODE_UNROLL"),
                             (dict(unroll=16, sign_mode=2), "SIGN_MODE")):
            run = self.emit("ptx-sm86", 2, 35, plan)
            self.assertEqual(run.returncode, 1)
            self.assertEqual(json.loads(run.stdout)["reason"], reason, plan)
            self.assertNotIn(b".target", run.stdout)
        ptx = self.emit("ptx-sm86", 2, 36077, full).stdout
        self.assertIn(b"pireus_lane_signs[16]", ptx)
        self.assertNotIn(b"pireus_signs[16]", ptx)
        self.assertNotIn(b"selp.f64", ptx)
        self.assertNotIn(b"fma", ptx)
        self.assertEqual(ptx.count(b"ld.const.u32"), 1)
        self.assertEqual(ptx.count(b"@%p1 sub.rn.f64"), 16)
        self.assertEqual(ptx.count(b"@!%p1 add.rn.f64"), 16)

    def test_sign_mode_ptx_computes_the_operator_reference_bit_for_bit(self):
        import operator_reference as ref
        import ptx_sim
        rows = ref.load_rows(HERE / "phase-a" / "numeric-fixtures.jsonl")

        def mismatches(ptx, phase, layout):
            got = ptx_sim.run_rows(ptx.decode(), rows, layout)
            bad = 0
            for v, row in enumerate(rows):
                for k, word in enumerate(ref.outputs(phase, row)):
                    other = got[v][k]
                    if word != other and not (ref.is_nan(word & ptx_sim.M64) and ref.is_nan(other & ptx_sim.M64)):
                        bad += 1
            return bad

        cases = ((1, None, dict(lane_stride=1, lane_offset=0, load=1, layout=0)),
                 (2, 36077, dict(lane_stride=3, lane_offset=5, load=0, layout=1)),
                 (2, 36076, dict(lane_stride=7, lane_offset=9, load=1, layout=1)),
                 (2, 33825, dict(lane_stride=15, lane_offset=13, load=1, layout=0)))
        for kind, phase, plan in cases:
            for sign_mode in (0, 1):
                ptx = self.emit("ptx-sm86", kind, phase, dict(plan, unroll=16, sign_mode=sign_mode)).stdout
                self.assertEqual(mismatches(ptx, phase or 0, plan["layout"]), 0, (kind, phase, plan, sign_mode))
        # Control: one wrong lane word must be visible, or a zero above means nothing.
        ptx = self.emit("ptx-sm86", 2, 36077, dict(unroll=16, sign_mode=1)).stdout.decode()
        head, rest = ptx.split("pireus_lane_signs[16] = {", 1)
        words, tail = rest.split("}", 1)
        words = words.split(",")
        words[3] = str(int(words[3]) ^ 1)
        broken = (head + "pireus_lane_signs[16] = {" + ",".join(words) + "}" + tail).encode()
        self.assertGreater(mismatches(broken, 36077, 0), 0)

    def test_pack_is_admitted_with_any_plan_and_refused_above_one(self):
        for kind, phase in ((1, None), (2, 33825)):
            for plan in (dict(unroll=1), dict(unroll=16, sign_mode=1)):
                receipt = json.loads(self.emit(None, kind, phase, dict(plan, pack=1)).stdout)
                plain = json.loads(self.emit(None, kind, phase, plan).stdout)
                self.assertEqual(receipt["decision"], "ADMIT", receipt)
                self.assertEqual(receipt["pack"], 1)
                self.assertNotIn("pack", plain)
                self.assertEqual(receipt["plan_id"] - 131072, plain["plan_id"])
                self.assertEqual(receipt["tensor_sha256"], plain["tensor_sha256"])
                self.assertEqual(receipt.get("sign_mode"), plain.get("sign_mode"))
        for plan, reason in ((dict(pack=2), "PACK"), (dict(pack=1, unroll=4, sign_mode=1), "SIGN_MODE_UNROLL")):
            run = self.emit("ptx-sm86", 2, 35, plan)
            self.assertEqual(run.returncode, 1)
            self.assertEqual(json.loads(run.stdout)["reason"], reason, plan)
            self.assertNotIn(b".target", run.stdout)
        ptx = self.emit("ptx-sm75", 2, 35, dict(pack=1, load=1, layout=1, unroll=16)).stdout
        # The launch is gpu_probe.py's: grid = vectors, 16 threads. pack keeps that contract.
        self.assertIn(b".maxntid 16,1,1\n", ptx)
        self.assertIn(b"shl.b32 %r1, %r1, 1;", ptx)
        self.assertIn(b"min.u32 %r15, %r15, %r16;", ptx)
        self.assertEqual(ptx.count(b"st.global.f64"), 2)
        self.assertEqual(ptx.count(b"shfl.sync.idx.b32"), 64)
        self.assertEqual(ptx.count(b"ld.const.u32"), 16)
        self.assertNotIn(b"fma", ptx)

    def test_pack_ptx_computes_the_operator_reference_bit_for_bit(self):
        import operator_reference as ref
        import ptx_sim
        rows = ref.load_rows(HERE / "phase-a" / "numeric-fixtures.jsonl")
        self.assertEqual(len(rows), 320)

        def mismatches(ptx, phase, layout, count):
            got = ptx_sim.run_rows(ptx.decode(), rows[:count], layout)
            bad = 0
            for v, row in enumerate(rows[:count]):
                for k, word in enumerate(ref.outputs(phase, row)):
                    other = got[v][k]
                    if word != other and not (ref.is_nan(word & ptx_sim.M64) and ref.is_nan(other & ptx_sim.M64)):
                        bad += 1
            return bad

        # 319 exercises the odd tail: the last block computes its one vector twice.
        cases = ((1, None, dict(lane_stride=1, lane_offset=0, load=1, layout=0, unroll=2), 320),
                 (2, 36077, dict(lane_stride=3, lane_offset=5, load=0, layout=1, unroll=16, sign_mode=1), 320),
                 (2, 36076, dict(lane_stride=7, lane_offset=9, load=1, layout=1, unroll=4), 319),
                 (2, 33825, dict(lane_stride=15, lane_offset=13, load=1, layout=0, unroll=16, sign_mode=1), 319),
                 (2, 1123, dict(lane_stride=11, lane_offset=2, load=0, layout=0, unroll=8), 320))
        for kind, phase, plan, count in cases:
            ptx = self.emit("ptx-sm86", kind, phase, dict(plan, pack=1)).stdout
            self.assertEqual(mismatches(ptx, phase or 0, plan["layout"], count), 0, (kind, phase, plan, count))
        # Controls: a one-bit sign flip, and a packed kernel that drops vector B's store.
        ptx = self.emit("ptx-sm86", 2, 36077, dict(unroll=16, pack=1)).stdout.decode()
        head, rest = ptx.split("pireus_signs[16] = {", 1)
        words, tail = rest.split("}", 1)
        words = words.split(",")
        words[5] = str(int(words[5]) ^ 8)
        broken = (head + "pireus_signs[16] = {" + ",".join(words) + "}" + tail).encode()
        self.assertGreater(mismatches(broken, 36077, 0, 320), 0)
        dropped = ptx.replace("    st.global.f64 [%rd6], %fd5;\n", "")
        self.assertNotEqual(dropped, ptx)
        self.assertGreater(mismatches(dropped.encode(), 36077, 0, 320), 0)

    def test_staged_load_and_vector_load_admission(self):
        for kind, phase in ((1, None), (2, 36076)):
            for plan in (dict(layout=0, unroll=1), dict(layout=1, unroll=16, sign_mode=1, pack=1)):
                staged = json.loads(self.emit(None, kind, phase, dict(plan, load=2)).stdout)
                direct = json.loads(self.emit(None, kind, phase, dict(plan, load=0)).stdout)
                self.assertEqual(staged["decision"], "ADMIT", staged)
                self.assertEqual(staged["load"], 2)
                self.assertNotIn("load", direct)
                self.assertEqual(staged["plan_id"] - 262144, direct["plan_id"])
                self.assertEqual(staged["tensor_sha256"], direct["tensor_sha256"])
            vec = json.loads(self.emit(None, kind, phase, dict(load=2, layout=1, pack=1, vector_load=1)).stdout)
            plain = json.loads(self.emit(None, kind, phase, dict(load=2, layout=1, pack=1)).stdout)
            self.assertEqual((vec["vector_load"], vec["plan_id"] - 524288), (1, plain["plan_id"]))
            self.assertNotIn("vector_load", plain)
        for plan, reason in ((dict(load=3), "GRAMMAR"),
                             (dict(load=2, layout=1, pack=1, vector_load=2), "VECTOR_LOAD"),
                             (dict(load=1, layout=1, pack=1, vector_load=1), "VECTOR_LOAD_LOAD"),
                             (dict(load=2, layout=0, pack=1, vector_load=1), "VECTOR_LOAD_LAYOUT"),
                             (dict(load=2, layout=1, vector_load=1), "VECTOR_LOAD_PACK"),
                             (dict(load=2, sign_mode=1, unroll=8), "SIGN_MODE_UNROLL")):
            run = self.emit("ptx-sm86", 2, 35, plan)
            self.assertEqual(run.returncode, 1)
            self.assertEqual(json.loads(run.stdout)["reason"], reason, plan)
            self.assertNotIn(b".target", run.stdout)
        for plan, smem, gloads, v2 in ((dict(load=2, layout=0, unroll=4), b"pireus_sa[16]", 2, 0),
                                       (dict(load=2, layout=1, pack=1, unroll=16), b"pireus_sa[32]", 4, 0),
                                       (dict(load=2, layout=1, pack=1, vector_load=1, unroll=16), b"pireus_sa[32]", 4, 2)):
            ptx = self.emit("ptx-sm75", 2, 35, plan).stdout
            self.assertIn(b".maxntid 16,1,1\n", ptx)
            self.assertIn(smem, ptx)
            self.assertEqual(ptx.count(b"bar.sync 0;"), 1)
            # The barrier sits after every shared store and before every shared load.
            barrier = ptx.index(b"bar.sync 0;")
            self.assertLess(ptx.rindex(b"st.shared.f64"), barrier)
            self.assertGreater(ptx.index(b"ld.shared.f64"), barrier)
            self.assertEqual(ptx.count(b"ld.global.f64"), gloads)
            self.assertEqual(ptx.count(b"ld.global.v2.f64 {"), v2)
            self.assertNotIn(b"shfl", ptx)
            self.assertNotIn(b"fma", ptx)
            self.assertNotIn(b".f32", ptx)

    def test_staged_ptx_computes_the_operator_reference_bit_for_bit(self):
        import operator_reference as ref
        import ptx_sim
        rows = ref.load_rows(HERE / "phase-a" / "numeric-fixtures.jsonl")
        self.assertEqual(len(rows), 320)

        def mismatches(ptx, phase, layout, count):
            got = ptx_sim.run_rows(ptx.decode() if isinstance(ptx, bytes) else ptx, rows[:count], layout)
            bad = 0
            for v, row in enumerate(rows[:count]):
                for k, word in enumerate(ref.outputs(phase, row)):
                    other = got[v][k]
                    if word != other and not (ref.is_nan(word & ptx_sim.M64) and ref.is_nan(other & ptx_sim.M64)):
                        bad += 1
            return bad

        # 320 takes the ld.global.v2.f64 path; 319 the odd-n scalar path and the packed tail.
        cases = ((1, None, dict(lane_stride=1, lane_offset=0, layout=0, unroll=1), 320),
                 (2, 36077, dict(lane_stride=3, lane_offset=5, layout=1, unroll=16, sign_mode=1), 319),
                 (2, 36076, dict(lane_stride=7, lane_offset=9, layout=0, unroll=4, pack=1), 319),
                 (2, 33825, dict(lane_stride=15, lane_offset=13, layout=1, unroll=16, sign_mode=1, pack=1, vector_load=1), 320),
                 (2, 33825, dict(lane_stride=15, lane_offset=13, layout=1, unroll=16, sign_mode=1, pack=1, vector_load=1), 319),
                 (2, 1123, dict(lane_stride=11, lane_offset=2, layout=1, unroll=8, pack=1, vector_load=1), 320))
        for kind, phase, plan, count in cases:
            ptx = self.emit("ptx-sm86", kind, phase, dict(plan, load=2)).stdout
            self.assertEqual(mismatches(ptx, phase or 0, plan["layout"], count), 0, (kind, phase, plan, count))
        # Controls. A sign-bit flip must diverge. Swapping the two halves of the v2 load
        # must diverge at even n and not at odd n, so both staging paths really ran.
        ptx = self.emit("ptx-sm86", 2, 36077, dict(load=2, layout=1, pack=1, vector_load=1, unroll=2)).stdout.decode()
        head, rest = ptx.split("pireus_signs[16] = {", 1)
        words, tail = rest.split("}", 1)
        words = words.split(",")
        words[7] = str(int(words[7]) ^ 2)
        self.assertGreater(mismatches(head + "pireus_signs[16] = {" + ",".join(words) + "}" + tail, 36077, 1, 320), 0)
        swapped = ptx.replace("ld.global.v2.f64 {%fd1,%fd6}", "ld.global.v2.f64 {%fd6,%fd1}")
        self.assertNotEqual(swapped, ptx)
        self.assertGreater(mismatches(swapped, 36077, 1, 320), 0)
        self.assertEqual(mismatches(swapped, 36077, 1, 319), 0)

    def test_b005_outputs_are_frozen_and_block_threads_0_or_16_keeps_them(self):
        frozen = json.loads((HERE / "frozen_inbox_outputs.json").read_text())
        b005 = frozen["outputs_b005"]
        self.assertEqual(len(b005), 16)
        context = HERE / "phase-a" / "context.json"
        variant = self.work / "threads.proposal.json"
        for name, modes in sorted(b005.items()):
            body = json.loads((CONTINUITY / "inbox" / name).read_text())
            self.assertEqual(body["load"], 2, name)
            self.assertNotIn("block_threads", body)
            for mode, digest in modes.items():
                argv = [str(self.admission), str(context), str(CONTINUITY / "inbox" / name)]
                argv += [] if mode == "receipt" else [mode]
                out = subprocess.run(argv, capture_output=True, timeout=30).stdout
                self.assertEqual(hashlib.sha256(receipt_v1(out) if mode == "receipt" else out).hexdigest(),
                                 digest, (name, mode))
                for threads in (0, 16):
                    variant.write_bytes(raw(dict(body, block_threads=threads)))
                    argv = [str(self.admission), str(context), str(variant)] + ([] if mode == "receipt" else [mode])
                    got = subprocess.run(argv, capture_output=True, timeout=30).stdout
                    if mode == "receipt":
                        # The proposal bytes (and so proposal_sha256) differ; everything else is equal.
                        frozen_receipt = json.loads(subprocess.run(
                            [str(self.admission), str(context), str(CONTINUITY / "inbox" / name)],
                            capture_output=True, timeout=30).stdout)
                        got = json.loads(got)
                        self.assertNotIn("block_threads", got)
                        got.pop("proposal_sha256")
                        frozen_receipt.pop("proposal_sha256")
                        self.assertEqual(got, frozen_receipt, (name, threads))
                        continue
                    self.assertEqual(hashlib.sha256(got).hexdigest(), digest, (name, mode, threads))

    def test_block_threads_admission_and_refusals(self):
        for kind, phase in ((1, None), (2, 36007)):
            for load in (0, 1, 2):
                base = json.loads(self.emit(None, kind, phase, dict(load=load, unroll=16)).stdout)
                for threads, step in ((32, 1), (64, 2), (128, 3)):
                    wide = json.loads(self.emit(None, kind, phase, dict(load=load, unroll=16, block_threads=threads)).stdout)
                    self.assertEqual(wide["decision"], "ADMIT", wide)
                    self.assertEqual(wide["block_threads"], threads)
                    self.assertEqual(wide["plan_id"] - step * 1048576, base["plan_id"])
                    self.assertEqual(wide["tensor_sha256"], base["tensor_sha256"])
                self.assertNotIn("block_threads", base)
                for threads in (0, 16):
                    same = json.loads(self.emit(None, kind, phase, dict(load=load, unroll=16, block_threads=threads)).stdout)
                    self.assertEqual(same["plan_id"], base["plan_id"])
                    self.assertNotIn("block_threads", same)
        for plan, mode, reason in ((dict(block_threads=8), "ptx-sm86", "BLOCK_THREADS"),
                                   (dict(block_threads=17), "ptx-sm86", "BLOCK_THREADS"),
                                   (dict(block_threads=48), "ptx-sm75", "BLOCK_THREADS"),
                                   (dict(block_threads=256), "ptx-sm86", "BLOCK_THREADS"),
                                   (dict(block_threads=1024), None, "BLOCK_THREADS"),
                                   (dict(block_threads=32, pack=1), "ptx-sm86", "BLOCK_THREADS_PACK"),
                                   (dict(block_threads=64, pack=1, load=2, layout=1, vector_load=1), None, "BLOCK_THREADS_PACK"),
                                   (dict(block_threads=32, load=2, layout=1, vector_load=1), "ptx-sm86", "VECTOR_LOAD_PACK"),
                                   (dict(block_threads=32), "ptx", "BLOCK_THREADS_TARGET"),
                                   (dict(block_threads=128, load=2), "ptx", "BLOCK_THREADS_TARGET"),
                                   (dict(block_threads=32, sign_mode=1, unroll=8), "ptx-sm86", "SIGN_MODE_UNROLL")):
            run = self.emit(mode, 2, 35, plan)
            self.assertEqual(run.returncode, 1, plan)
            self.assertEqual(json.loads(run.stdout)["reason"], reason, plan)
            self.assertNotIn(b".target", run.stdout)
        # The sm_121 (Spark) mode keeps its bytes when block_threads is 0 or 16.
        for threads in (0, 16):
            self.assertEqual(self.emit("ptx", 2, 35, dict(block_threads=threads)).stdout, self.emit("ptx", 2, 35).stdout)

    def test_wide_ptx_structure(self):
        for threads in (32, 64, 128):
            for load in (0, 1, 2):
                for arch in ("ptx-sm86", "ptx-sm75"):
                    ptx = self.emit(arch, 2, 36007, dict(load=load, layout=1, unroll=4, block_threads=threads)).stdout
                    self.assertIn(b".maxntid " + str(threads).encode() + b",1,1\n", ptx)
                    self.assertNotIn(b".maxntid 16,", ptx)
                    self.assertIn(b"(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u32 pn)\n", ptx)
                    self.assertIn(b"ld.param.u32 %r2, [pn];", ptx)
                    self.assertNotIn(b"%nctaid", ptx)
                    self.assertIn(b"mad.lo.u32 %r24, %r1, " + str(threads // 16).encode() + b", %r23;", ptx)
                    self.assertEqual(ptx.count(b"st.global.f64"), 1)
                    self.assertEqual(ptx.count(b"@%p3 st.global.f64"), 1)
                    self.assertNotIn(b"fma", ptx)
                    self.assertNotIn(b".f32", ptx)
                    self.assertEqual(ptx.count(b"bar.sync 0;"), 1 if load == 2 else 0)
                    if load == 2:
                        self.assertIn(b"pireus_sa[" + str(threads).encode() + b"];", ptx)
                        barrier = ptx.index(b"bar.sync 0;")
                        self.assertLess(ptx.rindex(b"st.shared.f64"), barrier)
                        self.assertGreater(ptx.index(b"ld.shared.f64"), barrier)
                        # No thread leaves between the guard and the barrier.
                        self.assertNotIn(b"bra DONE", ptx[ptx.index(b"min.u32"):barrier])
                    shuffles = ptx.count(b"shfl.sync.idx.b32")
                    self.assertEqual(shuffles, 8 if load == 1 else 0)
                    self.assertEqual(ptx.count(b", 4127, -1;"), shuffles)
                    self.assertNotIn(b"65535", ptx)

    def test_wide_ptx_computes_the_operator_reference_bit_for_bit(self):
        import operator_reference as ref
        import ptx_sim
        rows = ref.load_rows(HERE / "phase-a" / "numeric-fixtures.jsonl")

        def mismatches(ptx, phase, layout, count):
            got = ptx_sim.run_rows(ptx.decode() if isinstance(ptx, bytes) else ptx, rows[:count], layout)
            bad = 0
            for v, row in enumerate(rows[:count]):
                for k, word in enumerate(ref.outputs(phase, row)):
                    other = got[v][k]
                    if word != other and not (ref.is_nan(word & ptx_sim.M64) and ref.is_nan(other & ptx_sim.M64)):
                        bad += 1
            return bad

        # 320 fills every block; 319 leaves a short last block for 32, 64 and 128.
        cases = ((1, None, dict(lane_stride=1, lane_offset=0, layout=0, unroll=1, load=0, block_threads=32), 320),
                 (2, 36007, dict(lane_stride=9, lane_offset=10, layout=0, unroll=1, load=2, block_threads=32), 319),
                 (2, 33825, dict(lane_stride=13, lane_offset=9, layout=1, unroll=16, load=2, block_threads=64), 319),
                 (2, 36077, dict(lane_stride=1, lane_offset=15, layout=0, unroll=16, load=1, sign_mode=1, block_threads=128), 319),
                 (2, 36076, dict(lane_stride=11, lane_offset=3, layout=1, unroll=4, load=1, block_threads=64), 320),
                 (2, 3183, dict(lane_stride=13, lane_offset=4, layout=1, unroll=16, load=2, sign_mode=1, block_threads=128), 320))
        for kind, phase, plan, count in cases:
            ptx = self.emit("ptx-sm86", kind, phase, plan).stdout
            self.assertEqual(mismatches(ptx, phase or 0, plan["layout"], count), 0, (kind, phase, plan, count))
        # Controls. Each must diverge: a flipped sign bit; groups that share one shared-memory
        # slice; a wrong vectors-per-block factor; a shuffle that ignores the 16-lane segment.
        ptx = self.emit("ptx-sm86", 2, 36007, dict(lane_stride=9, lane_offset=10, layout=1, unroll=2, load=2,
                                                    block_threads=64)).stdout.decode()
        head, rest = ptx.split("pireus_signs[16] = {", 1)
        words, tail = rest.split("}", 1)
        words = words.split(",")
        words[5] = str(int(words[5]) ^ 4)
        self.assertGreater(mismatches(head + "pireus_signs[16] = {" + ",".join(words) + "}" + tail, 36007, 1, 320), 0)
        shared = ptx.replace("mul.wide.u32 %rd13, %r23, 128;", "mul.wide.u32 %rd13, %r23, 0;")
        self.assertNotEqual(shared, ptx)
        self.assertGreater(mismatches(shared, 36007, 1, 320), 0)
        mapping = ptx.replace("mad.lo.u32 %r24, %r1, 4, %r23;", "mad.lo.u32 %r24, %r1, 3, %r23;")
        self.assertNotEqual(mapping, ptx)
        self.assertGreater(mismatches(mapping, 36007, 1, 320), 0)
        shuffled = self.emit("ptx-sm86", 2, 36076, dict(lane_stride=11, lane_offset=3, layout=1, unroll=4, load=1,
                                                         block_threads=32)).stdout.decode()
        self.assertEqual(mismatches(shuffled, 36076, 1, 320), 0)
        crossed = shuffled.replace("and.b32 %r8, %r8, 15;", "and.b32 %r8, %r8, 15;\n    xor.b32 %r8, %r8, 1;")
        self.assertNotEqual(crossed, shuffled)
        self.assertGreater(mismatches(crossed, 36076, 1, 320), 0)
        # A shuffle without the 16-lane segment (c = 31) is not the emitted contract: refused.
        with self.assertRaises(AssertionError):
            mismatches(shuffled.replace(", 4127, -1;", ", 31, -1;"), 36076, 1, 320)

    def test_per_architecture_rule_matches_the_spark_bootstrap(self):
        text = "".join(f"{i} {i % 2} {900000 + (i * 17) % 50} 1000000\n" for i in range(30))
        paths = []
        for name in "abcd":
            path = self.work / f"{name}.tsv"
            path.write_text(text)
            paths.append(path)
        spark = subprocess.run([str(self.spark), *map(str, paths)], capture_output=True, text=True, timeout=60)
        arch = subprocess.run([str(self.arch), str(paths[0]), str(paths[1])], capture_output=True, text=True, timeout=60)
        self.assertEqual(spark.returncode, 0)
        self.assertEqual(arch.returncode, 0)
        spark_body = json.loads(spark.stdout)
        arch_body = json.loads(arch.stdout)
        self.assertEqual(arch_body["comparisons"], spark_body["comparisons"][:2])
        self.assertEqual(arch_body["scope"], "one-architecture")
        self.assertFalse(arch_body["cross_architecture"])
        self.assertEqual(subprocess.run([str(self.arch), str(paths[0])], capture_output=True).returncode, 2)
        self.assertEqual(subprocess.run([str(self.arch), *map(str, paths)], capture_output=True).returncode, 2)
        self.assertEqual(subprocess.run([str(self.spark), str(paths[0]), str(paths[1])], capture_output=True).returncode, 2)
        for candidate, gate in ((950000, "PASS"), (950001, "NO_GAIN"), (1000000, "NO_GAIN")):
            path = self.work / f"t{candidate}.tsv"
            path.write_text(series(candidate))
            body = json.loads(subprocess.check_output([str(self.arch), str(path), str(path)], text=True, timeout=60))
            self.assertEqual(body["gain_gate"], gate)
        high = self.work / "high.tsv"
        low = self.work / "low.tsv"
        high.write_text(series(940000))
        low.write_text(series(1000000))
        mixed = json.loads(subprocess.check_output([str(self.arch), str(high), str(low)], text=True, timeout=60))
        self.assertEqual(mixed["gain_gate"], "NO_GAIN")
        zero = self.work / "zero.tsv"
        zero.write_text(series(0))
        self.assertEqual(subprocess.run([str(self.arch), str(zero), str(zero)], capture_output=True).returncode, 2)


class AgentPath(unittest.TestCase):
    """--partition, --runs-root and the non-blocking submit/settle path. Defaults stay as before."""

    def test_defaults_unchanged_and_options_are_allowlisted(self):
        default = launch.build_plan(launch.DEFAULT_SCRATCH, REVIEW_NAME)
        self.assertEqual(default["partition"], "gpu-orangefs")
        self.assertEqual(default["remote_run"], launch.DEFAULT_SCRATCH + "/runs/" + REVIEW_NAME)
        agent = launch.build_plan(launch.DEFAULT_SCRATCH, REVIEW_NAME, partition="pireus-agent",
                                  runs_root="/scratch/pireus/r740/runs-agent")
        self.assertEqual(agent["remote_run"], "/scratch/pireus/r740/runs-agent/" + REVIEW_NAME)
        for step in agent["steps"]:
            argv = step["argv"]
            self.assertEqual(argv[argv.index("-p") + 1], "pireus-agent")
            self.assertIn("/runs-agent/" + REVIEW_NAME, step["script"])
            self.assertIn("#SBATCH -p pireus-agent", step["sbatch_review"])
        for bad in ("all", "gpu-spark", "", "gpu-orangefs,all"):
            with self.assertRaises(ValueError):
                launch.build_plan(launch.DEFAULT_SCRATCH, REVIEW_NAME, partition=bad)
        for bad in ("runs-agent", "/scratch/../etc"):
            with self.assertRaises(ValueError):
                launch.build_plan(launch.DEFAULT_SCRATCH, REVIEW_NAME, runs_root=bad)
        with self.assertRaises(ValueError):
            launch.build_plan(launch.DEFAULT_SCRATCH, REVIEW_NAME, mem="65G", partition="pireus-agent")

    def test_submit_once_settle_then_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            phase_a.bundle(bundle)
            image = declared_image(Path(directory) / "image.sif")
            scratch = Path(directory) / "scratch"
            runs = Path(directory) / "runs-agent"
            root = launch.prepare_run(scratch, REVIEW_NAME, bundle, str(runs))
            self.assertEqual(root, runs / REVIEW_NAME)
            self.assertFalse((scratch / "runs").exists())
            scripts = []

            def sbatch(path):
                scripts.append(Path(path).read_text())
                return str(100 + len(scripts))

            kw = dict(image_lock=image, partition="pireus-agent", runs_root=str(runs))
            with self.assertRaises(RuntimeError) as caught:
                launch.submit_step(scratch, REVIEW_NAME, "timing-sm86", sbatch, **kw)
            self.assertIn("successful probe", str(caught.exception))
            record = launch.submit_step(scratch, REVIEW_NAME, "probe-sm86", sbatch, **kw)
            self.assertEqual(record["status"], "submitted")
            self.assertEqual(record["job"], "101")
            text = scripts[0]
            self.assertIn("#SBATCH -p pireus-agent", text)
            self.assertIn("#SBATCH --gres=gpu:a5000:1", text)
            self.assertIn("#SBATCH --mem=16G", text)
            self.assertIn("#SBATCH --no-requeue", text)
            self.assertIn("#SBATCH --output=" + str(root / "logs" / "probe-sm86-%j.out"), text)
            self.assertIn(image["derived_sif_sha256"], text)
            self.assertNotIn("--kill-on-bad-exit", text)
            self.assertEqual(text.count("--gres="), 1)
            with self.assertRaises(RuntimeError) as caught:
                launch.submit_step(scratch, REVIEW_NAME, "probe-sm75", sbatch, **kw)
            self.assertIn("still open", str(caught.exception))
            running = launch.settle_step(scratch, REVIEW_NAME, "probe-sm86", lambda job: ("RUNNING", "0:0"), str(runs))
            self.assertEqual(running["status"], "submitted")
            with self.assertRaises(RuntimeError):
                launch.submit_step(scratch, REVIEW_NAME, "timing-sm86", sbatch, **kw)
            marker = root / "markers" / "probe-sm86.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}\n")
            done = launch.settle_step(scratch, REVIEW_NAME, "probe-sm86", lambda job: ("COMPLETED", "0:0"), str(runs))
            self.assertEqual(done["status"], "ok")
            with self.assertRaises(RuntimeError):
                launch.settle_step(scratch, REVIEW_NAME, "probe-sm86", lambda job: ("COMPLETED", "0:0"), str(runs))
            with self.assertRaises(RuntimeError) as caught:
                launch.submit_step(scratch, REVIEW_NAME, "probe-sm86", sbatch, **kw)
            self.assertIn("refusing to resubmit", str(caught.exception))
            launch.submit_step(scratch, REVIEW_NAME, "timing-sm86", sbatch, **kw)
            self.assertEqual(len(scripts), 2)
            launch.settle_step(scratch, REVIEW_NAME, "timing-sm86", lambda job: ("COMPLETED", "0:0"), str(runs))
            self.assertEqual(json.loads((root / "stages" / "timing-sm86.json").read_text())["status"], "failed")
            launch.submit_step(scratch, REVIEW_NAME, "probe-sm75", sbatch, **kw)
            failed = launch.settle_step(scratch, REVIEW_NAME, "probe-sm75", lambda job: ("FAILED", "1:0"), str(runs))
            self.assertEqual(failed["status"], "failed")
            with self.assertRaises(RuntimeError):
                launch.submit_step(scratch, REVIEW_NAME, "timing-sm75", sbatch, **kw)
            self.assertEqual(len(scripts), 3)


class OperatorReference(unittest.TestCase):
    def test_phase0_is_the_sounio_fixture_and_a_twist_is_not(self):
        import operator_reference as ref
        fixtures = HERE / "phase-a" / "numeric-fixtures.jsonl"
        rows = ref.load_rows(fixtures)
        self.assertTrue(ref.self_check(rows))
        expected = ref.frame([row["output_bits"] for row in rows])
        self.assertEqual(ref.reference(0, fixtures), expected)
        twisted = ref.reference(1128, fixtures)
        self.assertNotEqual(twisted, expected)
        self.assertEqual(len(twisted), len(expected))
        self.assertEqual(ref.sigma_phase(1128, 3, 5), -ref.cd_sigma(3, 5) if ref.bilinear(1128, 3, 5) else ref.cd_sigma(3, 5))
        for bad in (-1, 65536):
            with self.assertRaises(ValueError):
                ref.reference(bad, fixtures)


class WideLaunch(unittest.TestCase):
    """The r740-only launch for block_threads 32/64/128. gpu_probe.py's call stays the 16 path."""

    class Lib:
        def __init__(self):
            self.calls = []

        def cuLaunchKernel(self, *argv):
            # Read the parameter block now: the driver copies it at launch time too.
            import ctypes
            block = argv[9]
            values = [ctypes.cast(block[i], ctypes.POINTER(ctypes.c_uint64)).contents.value for i in range(3)]
            values.append(ctypes.cast(block[3], ctypes.POINTER(ctypes.c_uint32)).contents.value)
            self.calls.append(argv[:9] + (values, argv[10]))
            return 0

    class FakeDriver:
        def __init__(self):
            self.lib = WideLaunch.Lib()
            self.narrow = []

        def launch(self, function, a, b, c, n):
            self.narrow.append((function, n))

        def check(self, code):
            assert code == 0

    class Tensor:
        def __init__(self, address):
            self.address = address

        def data_ptr(self):
            return self.address

    def header(self, threads):
        params = ".param .u64 pa, .param .u64 pb, .param .u64 pc"
        if threads != 16:
            params += ", .param .u32 pn"
        return (".version 9.0\n.target sm_86\n.address_size 64\n.visible .entry pireus_candidate(" + params
                + ")\n.maxntid " + str(threads) + ",1,1\n{\n}\n").encode()

    def test_geometry_is_read_from_the_ptx_and_must_match_the_manifest(self):
        import r740_wide_launch as wide
        for threads in (16, 32, 64, 128):
            item = {} if threads == 16 else {"block_threads": threads}
            self.assertEqual(wide.geometry(item, self.header(threads)), threads)
        with self.assertRaises(ValueError):
            wide.geometry({"block_threads": 32}, self.header(64))
        with self.assertRaises(ValueError):
            wide.geometry({}, self.header(32))
        for bad in (0, 8, 48, 256, True, "32"):
            with self.assertRaises(ValueError):
                wide.declared_threads({"block_threads": bad})
        wrong = self.header(32).replace(b", .param .u32 pn", b"")
        with self.assertRaises(ValueError):
            wide.ptx_threads(wrong)
        with self.assertRaises(ValueError):
            wide.ptx_threads(self.header(16) + b".maxntid 16,1,1\n")
        # Every vendored Phase A PTX is a 16-thread gpu_probe.py body.
        for path in sorted((HERE / "phase-a" / "ptx").rglob("*.ptx")):
            self.assertEqual(wide.ptx_threads(path.read_bytes()), 16, path)

    def test_grid_and_launch_arguments(self):
        import r740_wide_launch as wide
        self.assertEqual([wide.grid(320, t) for t in (16, 32, 64, 128)], [320, 160, 80, 40])
        self.assertEqual([wide.grid(319, t) for t in (16, 32, 64, 128)], [319, 160, 80, 40])
        self.assertEqual([wide.grid(16384, t) for t in (32, 64, 128)], [8192, 4096, 2048])
        for n in (0, 16777217, True):
            with self.assertRaises(ValueError):
                wide.grid(n, 32)
        a, b, c = self.Tensor(4096), self.Tensor(8192), self.Tensor(12288)
        driver = self.FakeDriver()
        wide.launch(driver, "f16", a, b, c, 320, 16, stream=7)
        self.assertEqual(driver.narrow, [("f16", 320)])
        self.assertEqual(driver.lib.calls, [])
        for threads, blocks in ((32, 160), (64, 80), (128, 40)):
            wide.launch(driver, "fw", a, b, c, 320, threads, stream=7)
            argv = driver.lib.calls[-1]
            self.assertEqual(argv[:8], ("fw", blocks, 1, 1, threads, 1, 1, 0))
            self.assertEqual(argv[8].value, 7)
            self.assertIsNone(argv[10])
            self.assertEqual(argv[9], [4096, 8192, 12288, 320])
        self.assertEqual(driver.narrow, [("f16", 320)])

    def test_observer_launches_only_through_the_wide_module_and_it_is_staged(self):
        observe = (HERE / "r740_observe.py").read_text()
        self.assertNotIn("driver.launch(", observe)
        self.assertEqual(observe.count("launch(driver, function, *tensors, out, "), 3)
        self.assertIn("r740_wide_launch.py", launch.STAGED)
        with tempfile.TemporaryDirectory() as directory:
            bindir = launch.stage_into(directory)
            self.assertEqual((bindir / "r740_wide_launch.py").read_bytes(), (HERE / "r740_wide_launch.py").read_bytes())
            self.assertEqual((bindir / "gpu_probe.py").read_bytes(), (CONTINUITY / "gpu_probe.py").read_bytes())
        source = (HERE / "r740_wide_launch.py").read_text()
        for word in launch.FORBIDDEN:
            self.assertNotIn(word, source)


class SparkUntouched(unittest.TestCase):
    def test_spark_sources_keep_their_prechange_hashes(self):
        for rel, digest in UNCHANGED.items():
            self.assertEqual(sha(ROOT / rel), digest, rel)

    def test_gpu_probe_is_byte_identical_to_bae639fe(self):
        # gpu_probe.py at bae639fe (the b005 head) is d9a01f4c...; the r740 wide launch
        # lives in r740/r740_wide_launch.py and never edits it.
        probe = CONTINUITY / "gpu_probe.py"
        self.assertEqual(sha(probe), "d9a01f4c4c3e9a3b33cc7150fc11e72cd8627aa3b143e11e4942a245c75ec959")
        self.assertEqual(UNCHANGED["tools/pireus/continuity/gpu_probe.py"], sha(probe))
        self.assertIn(b"self.lib.cuLaunchKernel(function,n,1,1,16,1,1,0,", probe.read_bytes())


if __name__ == "__main__":
    unittest.main()
