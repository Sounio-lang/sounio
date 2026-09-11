"""Execute the real gates against synthetic tool outputs; no material acceptance."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[4]

class ExtractionControls(unittest.TestCase):
    def run_gate(self, kind, mutation=""):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = ("pireus_dgx_xor_materializer_gate.sh" if kind == "ptx"
                    else "dgx_ptx_shfl_material_parity_futhark_gate.sh")
            for relative in ["scripts/ci/" + name, "scripts/lib/gate_assert.sh",
                             "scripts/lib/gate_artifact.sh"]:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            def put(path, text, executable=False):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
                if executable:
                    target.chmod(0o755)
            ptx = ".target sm_121\n.visible .entry step(\nshfl.sync.bfly.b32\nst.global.f64\n"
            put("synthetic.ptx", ptx)
            put("tools/pireus/cross_arch_candidates.values.v1",
                "selected=dgx-sm121/xor-shuffle+sign-xor+mul+tree-reduce synthetic\n")
            receipt = ("hardware=2x-NVIDIA-DGX-Spark-GB10-sm121\nresult=PASS\n"
                       "semantic_authority_role=SEMANTIC_AUTHORITY\nproducer_role=MATERIAL_PARITY\n")
            if mutation != "missing-hash":
                receipt += "ptx_sha256=" + hashlib.sha256(ptx.encode()).hexdigest() + "\n"
            if mutation != "missing-job":
                receipt += "slurm_job_id=123\n"
            if mutation == "duplicate-job":
                receipt += "slurm_job_id=124\n"
            put("tools/cluster/evidence/pireus_dgx_typed_xor.receipt.v1", receipt)
            put("tools/pireus/dgx_ptx_shfl_material_parity.fut", "-- synthetic source\n")
            put("bin/souc", '#!/usr/bin/env bash\nwhile [[ $# -gt 0 ]]; do\n'
                'if [[ "$1" == "-o" ]]; then cp synthetic.ptx "$2"; break; fi\nshift\ndone\n'
                'echo "PIREUS_HLIR_TYPED operator_kind=1 bits=4 twist=1 candidate=0 argc=3 callee_len=0"\n', True)
            put("bin/futhark", '#!/usr/bin/env bash\nif [[ "$1" == "-V" ]]; then\n'
                '[[ "$MUTATION" == "empty-version" ]] || echo "Futhark 0.25.27."\nexit 0\nfi\n'
                'printf "#!/usr/bin/env bash\\n" > "$4"\n'
                'if [[ "$MUTATION" != "empty-output" ]]; then echo "echo true" >> "$4"; fi\n'
                'chmod +x "$4"\n', True)
            if mutation == "empty-hash":
                put("bin/sha256sum", "#!/usr/bin/env bash\nexit 0\n", True)
            env = dict(os.environ, PATH=str(root / "bin") + ":" + os.environ["PATH"],
                       MUTATION=mutation)
            return subprocess.run(["bash", str(root / "scripts/ci" / name)],
                                  cwd=root, env=env, capture_output=True, text=True, timeout=10)

    def test_synthetic_complete_records_pass(self):
        for kind in ["ptx", "futhark"]:
            result = self.run_gate(kind)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_and_duplicate_material_identity_refuse(self):
        for mutation in ["missing-hash", "missing-job", "duplicate-job", "empty-hash"]:
            result = self.run_gate("ptx", mutation)
            self.assertNotEqual(result.returncode, 0, mutation)
            self.assertIn("FAIL", result.stderr)

    def test_empty_futhark_extractions_refuse(self):
        for mutation in ["empty-version", "empty-output", "empty-hash"]:
            result = self.run_gate("futhark", mutation)
            self.assertNotEqual(result.returncode, 0, mutation)
            self.assertIn("FAIL", result.stderr)

if __name__ == "__main__":
    unittest.main()
