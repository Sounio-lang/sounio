#!/usr/bin/env python3
"""Tests for the Claude API proposer condition. No network, no API key, no GPU.

Unit tests use a labelled stub admission engine for speed; the integration
test runs the real Sounio admission + atlas executable (compiled with
bin/souc unless --engine/--context-engine are given) on a mock transport.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import claude_proposer as cp  # noqa: E402

SENTINEL = "sk-ant-TEST-SENTINEL-0123456789-must-never-appear"
WORKSPACE = "wrkspc_TEST0000000000000000"  # placeholder, not a real workspace
CONTEXT = b'{"schema":1,"target":701202,"dimension":16,"precision":64,"order":1,"fma":0,"capabilities":55,"lane_width":32,"facts_state":1}\n'
# TEST STUB, not an authority: admits any JSON object with an integer phase,
# reports a fake class per phase. The integration test uses the real engine.
STUB_ADMISSION = r'''#!/usr/bin/env python3
import hashlib, json, sys
raw = open(sys.argv[2], "rb").read()
try:
    p = json.loads(raw); assert isinstance(p, dict) and isinstance(p.get("phase"), int)
except Exception:
    print(json.dumps(dict(schema=1, authority="Sounio", decision="REFUSE", reason="PROPOSAL_FORMAT"))); sys.exit(1)
print(json.dumps(dict(schema=2, authority="Sounio", decision="ADMIT", novelty="N2",
    canonical_table=hashlib.sha256(str(p["phase"] % 7).encode()).hexdigest(),
    scientific_novelty=False, claim_ready=False)))
'''
REAL = {}


def executable(directory, name, text):
    path = Path(directory) / name
    path.write_text(text)
    path.chmod(0o755)
    return path


def all_bytes(root):
    return b"".join(p.read_bytes() for p in Path(root).rglob("*") if p.is_file())


class FakeAnthropicModule(types.ModuleType):
    """Stands in for the SDK. Records the key it was given (so the test can
    prove the key reached only the client) and answers like MockTransport."""
    def __init__(self, fail_with_key=False, constant=False):
        super().__init__("anthropic")
        self.__version__ = "fake-0"
        self.seen_keys = []
        self.bodies = []
        self.headers = []
        module = self

        class Raw:
            def __init__(self, content):
                self.http_response = types.SimpleNamespace(content=content, headers={"request-id": "req_fake"})

        class Messages:
            def __init__(self):
                self.with_raw_response = self

            def create(self, extra_body=None, **body):
                body.update(extra_body or {})
                module.bodies.append(body)
                if fail_with_key:
                    raise ValueError("authentication failed for key " + module.seen_keys[-1])
                if constant:
                    base = json.loads(body["messages"][0]["content"].rsplit("Template: ", 1)[1])
                    text = json.dumps(base | dict(phase=4660))
                    raw = cp.encoded(dict(id="msg_c", type="message", role="assistant", model=body["model"],
                                          content=[dict(type="text", text=text)], stop_reason="end_turn",
                                          usage=dict(input_tokens=600, output_tokens=150)))
                    return Raw(raw)
                raw, _ = cp.MockTransport().send(body)
                return Raw(raw)

        class Anthropic:
            def __init__(self, api_key, max_retries, timeout, default_headers=None):
                module.seen_keys.append(api_key)
                module.headers.append(default_headers)
                self.messages = Messages()

        self.Anthropic = Anthropic


class ProposerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.context = executable(self.dir, "context.sh", "#!/bin/sh\nprintf '%s'\n" % CONTEXT.decode().strip())
        self.engine = executable(self.dir, "stub_admission.py", STUB_ADMISSION)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_WORKSPACE_ID", None)

    def tearDown(self):
        os.environ.pop("ANTHROPIC_WORKSPACE_ID", None)
        sys.modules.pop("anthropic", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self.tmp.cleanup()

    def run_cli(self, *extra, run="run", command="run"):
        argv = [command, "--engine", str(self.engine), "--context-engine", str(self.context),
                "--per-arm", "8", "--rounds", "2", *extra]
        if command != "estimate":
            argv += ["--run", str(self.dir / run)]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cp.main(argv)
            except SystemExit as exit_:
                code = exit_.code
        return code, out.getvalue(), err.getvalue()

    # --- schema ---------------------------------------------------------
    def test_manifest_and_request_schema(self):
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        manifest = json.loads((self.dir / "run/manifest.json").read_text())
        self.assertEqual(manifest["condition_id"], "claude-api-proposer.v2:claude-sonnet-5-5")
        self.assertEqual(manifest["history_mode"], "proposals+classes")
        self.assertEqual(manifest["dedup_policy_sha256"], hashlib.sha256(cp.DEDUP_POLICY.encode()).hexdigest())
        self.assertEqual(manifest["history_head_sha256"], hashlib.sha256(cp.HISTORY_HEAD.encode()).hexdigest())
        self.assertIsNone(manifest["temperature"])
        self.assertIn("NOT the PIREUS_CONTINUITY_PLAN Inkling pilot", manifest["condition_note"])
        self.assertEqual(manifest["arms"], list(cp.ARMS))
        self.assertIs(manifest["scientific_novelty"], False)
        self.assertIs(manifest["claim_ready"], False)
        self.assertEqual(manifest["novelty_oracle"], "Sounio only")
        for key in ("prompt_sha256", "context_sha256", "admission_engine_sha256",
                    "atlas_source_sha256", "proposer_source_sha256", "price_source"):
            self.assertRegex(str(manifest[key]), r"\S")
        models = set()
        for path in (self.dir / "run").rglob("*.request.json"):
            body = json.loads(path.read_text())
            self.assertEqual(set(body), {"model", "max_tokens", "messages", "output_config"})
            self.assertNotIn("temperature", body)   # rejected by claude-sonnet-5-5
            self.assertEqual(len(body["messages"]), 1)
            template = json.loads(body["messages"][0]["content"].rsplit("Template: ", 1)[1])
            context = (self.dir / "run/context.json").read_bytes()
            self.assertEqual(manifest["context_sha256"], hashlib.sha256(context).hexdigest())
            self.assertEqual(template, cp.template(hashlib.sha256(context).hexdigest()))
            models.add(body["model"])
            withheld = "withheld" in body["messages"][0]["content"]
            self.assertEqual(withheld, "claude-no-ontology" in str(path))
        self.assertEqual(models, {"claude-sonnet-5-5"})   # same pinned model, both arms
        self.assertFalse(list((self.dir / "run/round-0/deterministic-search").glob("*.request.json")))
        report = json.loads(out)
        self.assertEqual(len(report["cells"]), 6)
        self.assertIs(report["claim_ready"], False)

    def test_feedback_only_between_rounds_and_only_own_arm(self):
        self.run_cli()
        r0 = (self.dir / "run/round-0/claude-ontology/000.request.json").read_text()
        r1 = json.loads((self.dir / "run/round-1/claude-ontology/000.request.json").read_text())
        self.assertNotIn("Sounio receipts for your earlier", r0)
        text = r1["messages"][0]["content"]
        self.assertIn("Sounio receipts for your earlier", text)
        self.assertEqual(text.count(" | ") + 1, 8)   # exactly the 8 round-0 receipts of this arm

    def test_model_change_is_a_new_condition(self):
        self.run_cli()
        code, _, err = self.run_cli("--model", "claude-opus-5-5")
        self.assertNotEqual(code, 0)
        code, _, _ = self.run_cli("--model", "claude-opus-5-5", run="opus")
        self.assertEqual(code, 0)
        manifest = json.loads((self.dir / "opus/manifest.json").read_text())
        self.assertEqual(manifest["condition_id"], "claude-api-proposer.v2:claude-opus-5-5")
        self.assertEqual(manifest["prices_usd_per_mtok"], [4.0, 20.0])

    def test_unknown_model_needs_declared_price(self):
        with self.assertRaises(ValueError):
            self.run_cli("--model", "claude-unpriced")

    # --- transcripts and batch -------------------------------------------
    def test_transcript_hashes_and_batch_before_admission(self):
        self.run_cli()
        root = self.dir / "run"
        calls = [x for x in map(json.loads, (root / "ledger.jsonl").read_text().splitlines()) if x["event"] == "CALL"]
        self.assertEqual(len(calls), 2 * 2 * 8)
        for row in calls:
            stem = root / row["cell"] / ("%03d" % row["index"])
            self.assertEqual(hashlib.sha256(Path(str(stem) + ".request.json").read_bytes()).hexdigest(), row["request_sha256"])
            self.assertEqual(hashlib.sha256(Path(str(stem) + ".response.json").read_bytes()).hexdigest(), row["response_sha256"])
            self.assertGreater(row["input_tokens"], 0)
            self.assertAlmostEqual(row["cost_usd"], (row["input_tokens"] * 2 + row["output_tokens"] * 10) / 1e6, places=6)
        journal = [json.loads(x) for x in (root / "journal.jsonl").read_text().splitlines()]
        stages = [(x["stage"], x["artifact"]) for x in journal]
        for cell in ("round-0/claude-ontology", "round-1/deterministic-search"):
            batch_at = stages.index(("batch", cell + "/batch.json"))
            first_admit = min(i for i, (s, a) in enumerate(stages) if s == "admit" and a.startswith(cell))
            self.assertLess(batch_at, first_admit)
        batch = json.loads((root / "round-0/claude-ontology/batch.json").read_text())
        self.assertEqual(hashlib.sha256(cp.encoded(batch["proposals"])).hexdigest(), batch["batch_sha256"])

    def test_tampered_proposal_after_batch_is_refused(self):
        run = cp.Run(self.dir / "t")
        run.root.mkdir()
        run.write("context.json", CONTEXT)
        manifest = dict(per_arm=8, rounds=1, model="m", max_tokens=10, effort="", temperature=None, budget_usd=1)
        cell = dict(round=0, arm="deterministic-search")
        cp.generate_cell(run, manifest, None, cell, CONTEXT, None, (2.0, 10.0))
        (run.root / "round-0/deterministic-search/003.proposal.json").write_bytes(b'{"phase":1}')
        with self.assertRaisesRegex(ValueError, "after its batch hash"):
            cp.admit_cell(run, manifest, self.engine, cell)

    def test_proposal_text_is_not_repaired(self):
        self.run_cli()
        root = self.dir / "run"
        fenced = [p for p in root.rglob("*.proposal.json") if p.read_text().startswith("```")]
        self.assertTrue(fenced, "mock emits some fenced replies")
        for path in fenced:
            receipt = json.loads(Path(str(path).replace(".proposal.json", ".receipt.json")).read_text())
            self.assertEqual(receipt["decision"], "REFUSE")

    def test_deterministic_arm_is_fixed_and_without_repeats(self):
        order = cp.deterministic_order()
        self.assertEqual(len(order), 1024)
        self.assertEqual(len(set(order)), 1024)
        self.assertEqual(order, cp.deterministic_order())
        for code in order:
            for i in range(4):
                for j in range(i):
                    self.assertFalse(code >> (4 * i + j) & 1)

    # --- budget -----------------------------------------------------------
    def test_budget_cap_stops_before_overspending(self):
        code, out, err = self.run_cli("--budget-usd", "0.3")
        self.assertEqual(code, 3)
        self.assertIn("BUDGET_STOP", err)
        ledger = [json.loads(x) for x in (self.dir / "run/ledger.jsonl").read_text().splitlines()]
        spent = sum(x["cost_usd"] for x in ledger)
        self.assertLessEqual(spent, 0.3)
        stop = [x for x in ledger if x["event"] == "BUDGET_STOP"]
        self.assertEqual(len(stop), 1)
        self.assertGreater(stop[0]["spent_usd"] + stop[0]["reserve_usd"], 0.3)
        self.assertTrue(json.loads(out)["stopped"])

    def test_worst_case_bounds_the_call(self):
        body = dict(model="claude-sonnet-5-5", max_tokens=8192, messages=[dict(role="user", content="x" * 5000)])
        usage = dict(input_tokens=5000, output_tokens=8192)
        self.assertGreaterEqual(cp.worst_case_cost(body, (2.0, 10.0)), cp.call_cost(usage, (2.0, 10.0)))

    def test_estimate_is_a_dry_run(self):
        code, out, _ = self.run_cli(command="estimate")
        self.assertEqual(code, 0)
        value = json.loads(out)
        self.assertEqual(value["calls"], 2 * 2 * 8)
        self.assertLess(value["expected_usd"], value["worst_case_usd"])
        self.assertFalse((self.dir / "run").exists())

    def test_interrupted_paid_call_is_not_reissued(self):
        self.run_cli("--rounds", "1")
        root = self.dir / "run"
        (root / "round-0/claude-ontology/batch.json").unlink()
        (root / "round-0/claude-ontology/005.response.json").unlink()
        (root / "round-0/claude-ontology/005.proposal.json").unlink()
        code = None
        with self.assertRaisesRegex(RuntimeError, "not re-issuing a paid call"):
            code = cp.main(["run", "--engine", str(self.engine), "--context-engine", str(self.context),
                            "--per-arm", "8", "--rounds", "1", "--run", str(root)])
        self.assertIsNone(code)

    # --- key handling -------------------------------------------------------
    def test_real_mode_requires_confirmation_and_key(self):
        code, _, _ = self.run_cli("--mode", "real")
        self.assertIn("confirm-spend", str(code))
        code, _, _ = self.run_cli("--mode", "real", "--confirm-spend", run="r2")
        self.assertIn("ANTHROPIC_API_KEY is not set", str(code))

    def test_key_reaches_only_the_client(self):
        fake = FakeAnthropicModule()
        sys.modules["anthropic"] = fake
        os.environ["ANTHROPIC_API_KEY"] = SENTINEL
        code, out, err = self.run_cli("--mode", "real", "--confirm-spend")
        self.assertEqual(code, 0)
        self.assertEqual(set(fake.seen_keys), {SENTINEL})
        sent = [cp.encoded(b) for b in fake.bodies]
        stored = sorted((self.dir / "run").rglob("*.request.json"))
        self.assertEqual(sorted(sent), sorted(p.read_bytes() for p in stored))
        self.assertNotIn(SENTINEL.encode(), all_bytes(self.dir / "run"))
        self.assertNotIn(SENTINEL, out + err)
        ledger = [json.loads(x) for x in (self.dir / "run/ledger.jsonl").read_text().splitlines()]
        self.assertEqual({x["request_id"] for x in ledger if x["event"] == "CALL"}, {"req_fake"})
        self.assertEqual({x["transport"] for x in ledger if x["event"] == "CALL"}, {"anthropic-python-sdk"})

    def test_without_workspace_nothing_changes(self):
        fake = FakeAnthropicModule()
        sys.modules["anthropic"] = fake
        os.environ["ANTHROPIC_API_KEY"] = SENTINEL
        code, _, _ = self.run_cli("--mode", "real", "--confirm-spend")
        self.assertEqual(code, 0)
        self.assertEqual(set(map(repr, fake.headers)), {"None"})
        manifest = json.loads((self.dir / "run/manifest.json").read_text())
        self.assertNotIn("anthropic_workspace_id", manifest)
        ledger = (self.dir / "run/ledger.jsonl").read_text()
        self.assertNotIn("anthropic_workspace_id", ledger)

    def check_workspace_run(self, run, *extra):
        fake = FakeAnthropicModule()
        sys.modules["anthropic"] = fake
        os.environ["ANTHROPIC_API_KEY"] = SENTINEL
        code, out, err = self.run_cli("--mode", "real", "--confirm-spend", *extra, run=run)
        self.assertEqual(code, 0)
        self.assertEqual(fake.headers, [{"anthropic-workspace-id": WORKSPACE}])
        root = self.dir / run
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["anthropic_workspace_id"], WORKSPACE)
        calls = [json.loads(x) for x in (root / "ledger.jsonl").read_text().splitlines()]
        self.assertTrue(calls)
        self.assertEqual({x["anthropic_workspace_id"] for x in calls}, {WORKSPACE})
        for path in root.rglob("*.request.json"):
            data = path.read_bytes().lower()
            self.assertNotIn(b"authorization", data)
            self.assertNotIn(b"x-api-key", data)
            self.assertNotIn(b"anthropic-workspace-id", data)
        self.assertNotIn(SENTINEL.encode(), all_bytes(root))
        self.assertNotIn(SENTINEL, out + err)

    def test_workspace_header_from_flag(self):
        self.check_workspace_run("w1", "--workspace-id", WORKSPACE)

    def test_workspace_header_from_environment(self):
        os.environ["ANTHROPIC_WORKSPACE_ID"] = WORKSPACE
        self.check_workspace_run("w2")

    # --- intra-round history and dedup (v2) --------------------------------
    def test_history_enters_prompt_with_sounio_class_tags(self):
        self.run_cli("--rounds", "1")
        root = self.dir / "run"
        for arm in cp.CLAUDE_ARMS:
            for i in range(8):
                text = json.loads((root / ("round-0/%s/%03d.request.json" % (arm, i))).read_text())["messages"][0]["content"]
                if i == 0:
                    self.assertNotIn(cp.HISTORY_HEAD, text)
                    continue
                self.assertIn(cp.HISTORY_HEAD, text)
                self.assertIn(cp.DISTINCT_ASK, text)
                listed = text.split(cp.HISTORY_HEAD, 1)[1].split(". " + cp.DISTINCT_ASK, 1)[0].split(" | ")
                self.assertEqual(len(listed), i)
                for j, line in enumerate(listed):
                    proposal = (root / ("round-0/%s/%03d.proposal.json" % (arm, j))).read_bytes()
                    preview = json.loads((root / ("round-0/%s/%03d.preview.json" % (arm, j))).read_text())
                    try:
                        phase = json.loads(proposal).get("phase")
                    except Exception:
                        phase = "unparseable"
                    tag = ("class-" + hashlib.sha256(preview["canonical_table"].encode()).hexdigest()[:12]
                           if preview.get("canonical_table") else "%s:%s" % (preview["decision"], preview["reason"]))
                    self.assertEqual(line, "phase=%s %s" % (phase, tag))
                    for leak in ("tensor", "fp_parity", "gain", "ptx", "ns", "GiB"):
                        self.assertNotIn(leak, line)
            receipt = (root / ("round-0/%s/006.receipt.json" % arm)).read_bytes()
            self.assertEqual(receipt, (root / ("round-0/%s/006.preview.json" % arm)).read_bytes())
        self.assertFalse(list((root / "round-0/deterministic-search").glob("*.preview.json")))
        journal = [json.loads(x) for x in (root / "journal.jsonl").read_text().splitlines()]
        stages = [(x["stage"], x["artifact"]) for x in journal]
        batch = stages.index(("batch", "round-0/claude-ontology/batch.json"))
        first_admit = min(i for i, (st, a) in enumerate(stages) if st == "admit" and a.startswith("round-0/claude-ontology"))
        self.assertLess(batch, first_admit)

    def test_history_has_the_same_form_in_both_claude_arms(self):
        manifest = dict(model="m", max_tokens=10, effort="", temperature=None)
        history = ["phase=4660 class-0123456789ab", "phase=1 REFUSE:PROPOSAL_FORMAT"]
        texts = {}
        for arm in cp.CLAUDE_ARMS:
            text = cp.request_body(manifest, CONTEXT, arm, 0, 2, [], history)["messages"][0]["content"]
            facts = (cp.ONTOLOGY_LEGEND + " Context: " + CONTEXT.decode().strip()
                     if arm == "claude-ontology" else "Ontology facts: withheld in this condition.")
            self.assertEqual(text.count(facts), 1)
            texts[arm] = text.replace(facts, "<FACTS>")
        self.assertEqual(texts["claude-ontology"], texts["claude-no-ontology"])

    def test_exact_duplicate_is_recorded_and_not_replaced(self):
        fake = FakeAnthropicModule(constant=True)
        sys.modules["anthropic"] = fake
        os.environ["ANTHROPIC_API_KEY"] = SENTINEL
        code, out, _ = self.run_cli("--mode", "real", "--confirm-spend", "--rounds", "1",
                                    "--workspace-id", WORKSPACE)
        self.assertEqual(code, 0)
        root = self.dir / "run"
        ledger = [json.loads(x) for x in (root / "ledger.jsonl").read_text().splitlines()]
        report = {c["cell"]: c for c in json.loads(out)["cells"]}
        for arm in cp.CLAUDE_ARMS:
            cell = "round-0/" + arm
            dups = [x for x in ledger if x["event"] == "DUPLICATE" and x["cell"] == cell]
            self.assertEqual([x["index"] for x in dups], list(range(1, 8)))
            self.assertEqual({x["duplicate_of"] for x in dups}, {0})
            self.assertEqual(len([x for x in ledger if x["event"] == "CALL" and x["cell"] == cell]), 8)
            self.assertEqual(report[cell]["refused"].get("DUPLICATE"), 7)
            self.assertEqual(report[cell]["admitted_after_dedup"], 1)
            self.assertEqual(report[cell]["validated"], 8)      # still admitted by Sounio
        self.assertFalse([x for x in ledger if x["event"] == "DUPLICATE" and "deterministic" in x["cell"]])
        for path in root.rglob("*.request.json"):
            data = path.read_bytes()
            self.assertNotIn(SENTINEL.encode(), data)
            self.assertNotIn(WORKSPACE.encode(), data)
            self.assertNotIn(b"authorization", data.lower())
        self.assertNotIn(SENTINEL.encode(), all_bytes(root))
        # Resume adds no second DUPLICATE row.
        code, _, _ = self.run_cli("--mode", "real", "--confirm-spend", "--rounds", "1", "--workspace-id", WORKSPACE)
        again = [json.loads(x) for x in (root / "ledger.jsonl").read_text().splitlines()]
        self.assertEqual(len(again), len(ledger))

    def test_dedup_key(self):
        a = b'{"phase":5,"kind":2}'
        self.assertEqual(cp.dedup_key(a), cp.dedup_key(b'{ "kind": 2, "phase": 5 }\n'))
        self.assertNotEqual(cp.dedup_key(a), cp.dedup_key(b'{"phase":6,"kind":2}'))
        self.assertEqual(cp.dedup_key(b"```x"), cp.dedup_key(b"```x"))
        self.assertNotEqual(cp.dedup_key(b"```x"), cp.dedup_key(b"```y"))

    def test_history_none_keeps_v1_behaviour(self):
        code, _, _ = self.run_cli("--history", "none", "--rounds", "1")
        self.assertEqual(code, 0)
        root = self.dir / "run"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["condition_id"], "claude-api-proposer.v1:claude-sonnet-5-5")
        for key in ("history_mode", "dedup_policy", "intra_round_preview"):
            self.assertNotIn(key, manifest)
        self.assertFalse(list(root.rglob("*.preview.json")))
        self.assertNotIn("DUPLICATE", (root / "ledger.jsonl").read_text())
        context = (root / "context.json").read_bytes()
        for path in root.rglob("*.request.json"):
            arm, index = path.parent.name, int(path.name[:3])
            expected = cp.request_body(manifest, context, arm, 0, index, [])
            self.assertEqual(path.read_bytes(), cp.encoded(expected))
            self.assertNotIn(cp.HISTORY_HEAD, path.read_text())

    def test_key_in_error_message_is_redacted(self):
        sys.modules["anthropic"] = FakeAnthropicModule(fail_with_key=True)
        os.environ["ANTHROPIC_API_KEY"] = SENTINEL
        with self.assertRaises(RuntimeError) as caught:
            self.run_cli("--mode", "real", "--confirm-spend")
        self.assertNotIn(SENTINEL, str(caught.exception))
        self.assertIn("<api-key-removed>", str(caught.exception))
        self.assertNotIn(SENTINEL.encode(), all_bytes(self.dir / "run"))

    def test_artifact_containing_key_is_never_written(self):
        run = cp.Run(self.dir / "k", SENTINEL)
        run.root.mkdir()
        with self.assertRaises(cp.KeyLeak):
            run.write("x.json", b'{"echo":"' + SENTINEL.encode() + b'"}')
        self.assertFalse((run.root / "x.json").exists())
        self.assertFalse(list(run.root.iterdir()))

    def test_mock_mode_never_reads_the_key(self):
        os.environ["ANTHROPIC_API_KEY"] = SENTINEL
        code, out, err = self.run_cli()
        self.assertEqual(code, 0)
        self.assertNotIn(SENTINEL.encode(), all_bytes(self.dir / "run"))
        source = (HERE / "claude_proposer.py").read_text()
        self.assertEqual(source.count('os.environ.get("ANTHROPIC_API_KEY")'), 1)
        self.assertNotIn("print(secret", source)


class RealEngineIntegration(unittest.TestCase):
    """Mock transport, real Sounio admission + atlas executable."""
    def test_mock_run_with_real_sounio_admission(self):
        if not REAL:
            self.skipTest("no real engines")
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cp.main(["run", "--run", str(Path(tmp) / "run"), "--engine", str(REAL["engine"]),
                                "--context-engine", str(REAL["context"]), "--per-arm", "8", "--rounds", "1"])
            self.assertEqual(code, 0)
            report = json.loads(out.getvalue())
            root = Path(tmp) / "run"
            receipts = [json.loads(p.read_text()) for p in root.rglob("*.receipt.json")]
            self.assertEqual(len(receipts), 24)
            for receipt in receipts:
                self.assertEqual(receipt["authority"], "Sounio")
                if receipt["decision"] == "ADMIT":
                    self.assertEqual(receipt["classifier"], "eisa_h.pireus_operator_atlas.v1")
                    self.assertIn(receipt["novelty"], ("N0", "N1", "N2"))
                    self.assertIs(receipt["scientific_novelty"], False)
                    self.assertIs(receipt["claim_ready"], False)
                else:
                    self.assertTrue(receipt["reason"])
            self.assertEqual(json.loads((root / "context.json").read_text())["facts_state"], 1)
            self.assertEqual(sum(c["validated"] for c in report["cells"]), 24)
            print("CLAUDE_PROPOSER_REAL_ENGINE", json.dumps(dict(
                cells=[(c["cell"], c["admitted"], c["novelty"], c["canonical_classes"]) for c in report["cells"]])))


def build_real(args):
    if args.engine and args.context_engine:
        REAL.update(engine=args.engine.resolve(), context=args.context_engine.resolve())
        return None
    if args.skip_real:
        return None
    work = tempfile.TemporaryDirectory()
    souc = os.environ.get("SOUC_BIN") or str(ROOT / "bin/souc")
    for name, src in (("engine", "tools/pireus/continuity/admission.sio"),
                      ("context", "tools/pireus/continuity/research_context.sio")):
        out = Path(work.name) / (name + ".elf")
        subprocess.run([souc, str(ROOT / src), str(out)], cwd=ROOT, check=True, timeout=900,
                       stdout=subprocess.DEVNULL)
        out.chmod(0o755)
        REAL[name] = out
    return work


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--context-engine", type=Path)
    ap.add_argument("--skip-real", action="store_true")
    known, rest = ap.parse_known_args()
    keep = build_real(known)
    result = unittest.main(argv=[sys.argv[0], "-v", *rest], exit=False).result
    ok = result.wasSuccessful() and (REAL or known.skip_real)
    print("CLAUDE_PROPOSER_TESTS_" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
