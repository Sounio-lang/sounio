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

    def test_timing_row_matches_the_spark_grammar(self):
        self.assertEqual(paired_timing.row(1, 940000, 1000000), "1 1 940000 1000000\n")
        spark = (CONTINUITY / "gpu_benchmark.py").read_text()
        self.assertIn('manifest["blocks"]==30', spark.replace(" ", ""))
        self.assertIn("16384", spark)
        self.assertIn('manifest["replays"]==32', spark.replace(" ", ""))
        observe = (HERE / "r740_observe.py").read_text()
        self.assertIn("observation_dir(", observe)
        self.assertIn("device_count() != 1", observe)
        self.assertIn("key_scope", observe)
        self.assertNotIn("check_pair", observe)
        self.assertNotIn("timings-\"+socket.gethostname()", observe)


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


class SparkUntouched(unittest.TestCase):
    def test_spark_sources_keep_their_prechange_hashes(self):
        for rel, digest in UNCHANGED.items():
            self.assertEqual(sha(ROOT / rel), digest, rel)


if __name__ == "__main__":
    unittest.main()
