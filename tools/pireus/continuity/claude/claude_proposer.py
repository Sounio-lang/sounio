#!/usr/bin/env python3
"""Claude API proposer: a declared NEW proposer condition for Pireus kind=2 operators.

Condition `claude-api-proposer.v1:<model>`. It is not the Inkling pilot of
docs/roadmap/PIREUS_CONTINUITY_PLAN.md: that pilot (cycle.py, pilot.py,
CONDITIONS there) is unchanged and stays an open obligation. This condition
keeps the plan's shape: three arms (deterministic search, Claude without
ontology facts, Claude with the frozen research_context.sio TripleStore/SPARQL
output), 32 proposals per arm per round, 3 rounds, counterexample feedback
only between rounds, batch hash recorded before admission.

Python only transports. Every decision (admission, refusal reason, N0/N1/N2,
canonical class) comes from the native Sounio admission executable built
from admission.sio with eisa_h.pireus_operator_atlas.v1. Claude is never a
novelty oracle: its text is persisted unrepaired and handed to Sounio.

The API key is read from the environment variable ANTHROPIC_API_KEY only, in
--mode real only. It is never printed, logged or written; every artifact is
checked for it before it is written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CONDITION_FAMILY = "claude-api-proposer.v1"
# v2 = v1 + intra-round history of the arm's own proposals + declared dedup.
CONDITION_FAMILY_V2 = "claude-api-proposer.v2"
HISTORY_MODES = ("none", "proposals", "proposals+classes")
DEFAULT_HISTORY = "proposals+classes"
ARMS = ("deterministic-search", "claude-no-ontology", "claude-ontology")
CLAUDE_ARMS = ARMS[1:]
# Pinned Claude API id, verified 2026-11-20 against the Anthropic models
# overview (https://docs.anthropic.com/en/docs/about-claude/models/overview).
# A different model is a different declared condition, never a silent swap.
DEFAULT_MODEL = "claude-sonnet-5-5"
# USD per million tokens (input, output), same source and date. A model not
# listed here needs explicit --price-input/--price-output.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
PRICE_SOURCE = "https://docs.anthropic.com/en/docs/about-claude/models/overview (read 2026-11-20)"
# Caching is not requested. If the API still reports cache tokens they are
# charged conservatively: creation at 2x input, reads at 1x input.
CACHE_WRITE_FACTOR = 2.0
CACHE_READ_FACTOR = 1.0
DEFAULT_BUDGET_USD = 20.0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_EFFORT = "medium"
# Assumptions for the pre-run estimate only; the ledger records real usage.
EXPECTED_OUTPUT_TOKENS = 1500
BYTES_PER_TOKEN_EXPECTED = 3.5

PROMPT = (
    "You propose one untrusted Pireus OPERATOR. Reply with exactly one JSON "
    "object and nothing else: no prose, no code fence. "
    "Keep every key of the template and its value unchanged except `phase` and "
    "the optional lowering keys below. Do not add keys: no expected results, "
    "authority, claims, tensors, hashes or novelty labels; such proposals are refused. "
    "The operator is a product on 16 basis elements graded by XOR: "
    "e_i * e_j = sigma(i,j) e_(i XOR j), sigma(i,j) = cd_sigma(i,j) * (-1)^(i^T B j), "
    "where cd_sigma is the Cayley-Dickson sedenion sign and B is a 4x4 matrix "
    "over F2 that you choose through `phase`, an integer 0..65535 whose bit "
    "(4*i + j) is B[i][j] (i, j index the bits of the basis label). "
    "Different codes can give the same operator up to change of basis "
    "(GL(4,F2)), input swap and sign gauge. A native Sounio program, not you, "
    "rebuilds the 16x16 sign table, admits or refuses the proposal and "
    "classifies it against a frozen atlas; your proposal states no result. "
    "Optional lowering keys: lane_stride odd 1..15; lane_offset 0..15; "
    "load 0 or 1; layout 0 or 1; unroll in [1,2,4,8,16]. ")
ONTOLOGY_LEGEND = (
    "Context keys come from a Sounio TripleStore/SPARQL query over the Pireus "
    "target ontology (declared research facts, not observed hardware): "
    "capabilities is a bitmask of f64 load(1), multiply(2), add(4), indexed "
    "select(8), 16-lane shuffle(16), exact sign multiplication(32); lane_width "
    "is the warp width; facts_state 1 = consistent, 0 = unknown, 2 = contradictory.")
FEEDBACK_HEAD = (
    "Sounio receipts for your earlier proposals in this condition follow "
    "(phase, decision, refusal reason, N-level, canonical class tag). "
    "A class tag already listed is the same operator up to the declared "
    "equivalence. ")


HISTORY_HEAD = (
    "Proposals already made in this round of this condition, in order "
    "(phase; then the class tag or refusal reason from the Sounio admission "
    "of that proposal, when shown): ")
DISTINCT_ASK = (
    "Propose an operator distinct from every proposal listed above; a "
    "proposal whose class tag is already listed is the same operator up to "
    "the declared equivalence and adds nothing. ")
DEDUP_POLICY = (
    "Within one arm and round, a proposal equal to an earlier one (identical "
    "bytes, or the identical JSON object after sorted-key canonicalisation) is "
    "recorded in the ledger as DUPLICATE and counted as refused in the report. "
    "It is still hashed into the batch and admitted by Sounio, so nothing is "
    "hidden. No extra call is made to replace it: the call budget per arm and "
    "round stays per_arm. The same rule applies to every arm.")
PREVIEW_NOTE = (
    "proposals+classes: before each call of a Claude arm, every earlier "
    "proposal of that arm and round is passed through the same Sounio "
    "admission executable; its class tag or refusal reason enters the next "
    "prompt and the receipt is kept as NNN.preview.json. The batch hash is "
    "still written before the official admission. This is intra-round "
    "feedback, which the Inkling plan's pilot does not have; it is part of "
    "this declared condition only, identical in form for both Claude arms.")


class BudgetExhausted(Exception):
    pass


class KeyLeak(Exception):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


class Run:
    """Owns the run directory; every write goes through `write`, which
    refuses to overwrite a different immutable artifact and refuses any byte
    string containing the API key."""

    def __init__(self, root, secret=None):
        self.root = Path(root)
        self._secret = secret.encode() if secret else None

    def guard(self, data):
        if self._secret and self._secret in data:
            raise KeyLeak("refusing to persist an artifact that contains the API key")
        return data

    def write(self, rel, data):
        path = self.root / rel
        self.guard(data)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("immutable artifact differs: " + rel)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial")
        with tmp.open("wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
        return path

    def append(self, rel, value):
        data = self.guard(encoded(value))
        with (self.root / rel).open("ab") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())

    def rows(self, rel):
        path = self.root / rel
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def event(self, stage, rel):
        self.append("journal.jsonl", dict(stage=stage, artifact=rel,
                                          sha256=digest((self.root / rel).read_bytes())))


def prices_for(args):
    if args.price_input is not None and args.price_output is not None:
        return (args.price_input, args.price_output), "declared on the command line"
    if args.model in PRICES:
        return PRICES[args.model], PRICE_SOURCE
    raise ValueError("no pinned price for model %s; pass --price-input and --price-output" % args.model)


def call_cost(usage, prices):
    pin, pout = prices
    tokens_in = (usage.get("input_tokens") or 0) \
        + CACHE_WRITE_FACTOR * (usage.get("cache_creation_input_tokens") or 0) \
        + CACHE_READ_FACTOR * (usage.get("cache_read_input_tokens") or 0)
    return (tokens_in * pin + (usage.get("output_tokens") or 0) * pout) / 1e6


def worst_case_cost(body, prices):
    # Upper bound: no tokenizer emits more tokens than UTF-8 bytes of the
    # request; 64 tokens of framing; the full max_tokens of output.
    tokens_in = len(json.dumps(body).encode()) + 64
    return (tokens_in * prices[0] + body["max_tokens"] * prices[1]) / 1e6


def template(context_sha256):
    return dict(schema=1, target=701202, dimension=16, precision=64, order=1, fma=0,
                kind=2, lane_stride=1, lane_offset=0, load=1, layout=0, unroll=1,
                context=context_sha256, phase=0)


def cells(rounds):
    # Arm order rotates per round, as in the plan's pilot cells.
    return [dict(round=r, arm=ARMS[(r + k) % 3]) for r in range(rounds) for k in range(3)]


def cell_dir(cell):
    return "round-%d/%s" % (cell["round"], cell["arm"])


def deterministic_order():
    # Fixed search order over the 1024 upper-triangular codes (B[i][j], i<=j):
    # sorted by sha256 of the decimal code. No randomness, no model.
    codes = []
    pairs = [(i, j) for i in range(4) for j in range(i, 4)]
    for mask in range(1 << len(pairs)):
        code = 0
        for bit, (i, j) in enumerate(pairs):
            if mask >> bit & 1:
                code |= 1 << (4 * i + j)
        codes.append(code)
    return sorted(codes, key=lambda c: digest(str(c).encode()))


def feedback_lines(run, arm, before_round):
    lines = []
    for r in range(before_round):
        base = "round-%d/%s" % (r, arm)
        for receipt in sorted((run.root / base).glob("*.receipt.json")):
            value = json.loads(receipt.read_text())
            proposal = receipt.with_name(receipt.name.replace(".receipt.json", ".proposal.json"))
            try:
                phase = json.loads(proposal.read_text()).get("phase")
            except Exception:
                phase = None
            table = value.get("canonical_table")
            tag = "class-" + digest(table.encode())[:12] if table else "-"
            lines.append("r%d phase=%s %s %s %s %s" % (
                r, phase, value.get("decision"), value.get("reason", "-"),
                value.get("novelty", "-"), tag))
    return lines


def dedup_key(data):
    try:
        value = json.loads(data)
        if isinstance(value, dict):
            return digest(encoded(value))
    except Exception:
        pass
    return digest(data)


def preview(run, engine, proposal_rel):
    """Sounio admission of an earlier proposal of the round, used only to
    tell the model what it already proposed. Byte-identical to admission."""
    rel = proposal_rel.replace(".proposal.json", ".preview.json")
    if not (run.root / rel).exists():
        result = subprocess.run([str(engine), str(run.root / "context.json"), str(run.root / proposal_rel)],
                                capture_output=True, timeout=120)
        value = json.loads(result.stdout)
        if value.get("authority") != "Sounio" or result.returncode not in (0, 1):
            raise ValueError("invalid preview result for " + proposal_rel)
        run.write(rel, result.stdout)
        run.event("preview", rel)
    return json.loads((run.root / rel).read_text())


def history_lines(run, manifest, engine, base_dir, index):
    mode = manifest.get("history_mode", "none")
    lines = []
    for j in range(index):
        rel = "%s/%03d.proposal.json" % (base_dir, j)
        data = (run.root / rel).read_bytes()
        try:
            phase = json.loads(data).get("phase")
        except Exception:
            phase = "unparseable"
        line = "phase=%s" % (phase,)
        if mode == "proposals+classes":
            value = preview(run, engine, rel)
            table = value.get("canonical_table")
            line += " " + ("class-" + digest(table.encode())[:12] if table
                           else "%s:%s" % (value.get("decision"), value.get("reason")))
        lines.append(line)
    return lines


def record_duplicate(run, base_dir, index):
    rel = "%s/%03d.proposal.json" % (base_dir, index)
    key = dedup_key((run.root / rel).read_bytes())
    for j in range(index):
        if dedup_key((run.root / ("%s/%03d.proposal.json" % (base_dir, j))).read_bytes()) == key:
            if not any(x.get("event") == "DUPLICATE" and x["cell"] == base_dir and x["index"] == index
                       for x in run.rows("ledger.jsonl")):
                run.append("ledger.jsonl", dict(
                    event="DUPLICATE", cell=base_dir, index=index, duplicate_of=j,
                    proposal_sha256=digest((run.root / rel).read_bytes()), dedup_key=key, cost_usd=0.0))
            return j
    return None


def request_body(manifest, context, arm, rnd, index, feedback, history=None):
    base = template(digest(context))
    if arm == "claude-ontology":
        facts = ONTOLOGY_LEGEND + " Context: " + context.decode().strip()
    else:
        facts = "Ontology facts: withheld in this condition."
    text = (PROMPT + facts + " " +
            (FEEDBACK_HEAD + " | ".join(feedback) + " " if feedback else "") +
            (HISTORY_HEAD + " | ".join(history) + ". " + DISTINCT_ASK if history else "") +
            "This is proposal %d of round %d. Template: %s" % (index, rnd, json.dumps(base, sort_keys=True)))
    body = dict(model=manifest["model"], max_tokens=manifest["max_tokens"],
                messages=[dict(role="user", content=text)])
    if manifest["effort"]:
        body["output_config"] = dict(effort=manifest["effort"])
    if manifest["temperature"] is not None:
        body["temperature"] = manifest["temperature"]
    return body


def response_text(raw):
    value = json.loads(raw)
    texts = [b.get("text") for b in value.get("content", []) if b.get("type") == "text"]
    if len(texts) != 1 or not isinstance(texts[0], str):
        return None, value
    return texts[0], value


class MockTransport:
    """Deterministic stand-in shaped like a Messages API response. Usage is
    synthetic (bytes/4 in, fixed out) so budget arithmetic is exercised."""
    name = "mock"
    sdk_version = None
    workspace_id = None

    def __init__(self, output_tokens=900):
        self.output_tokens = output_tokens
        self.calls = 0

    def send(self, body):
        self.calls += 1
        prompt = body["messages"][0]["content"]
        base = json.loads(prompt.rsplit("Template: ", 1)[1])
        h = int(digest(prompt.encode())[:8], 16)
        if h % 11 == 0:
            text = "```json\n" + json.dumps(base | dict(phase=h % 65536)) + "\n```"
        else:
            text = json.dumps(base | dict(phase=h % 65536, lane_stride=1 + 2 * (h % 8)))
        raw = dict(id="msg_mock_%08x" % h, type="message", role="assistant", model=body["model"],
                   content=[dict(type="text", text=text)], stop_reason="end_turn",
                   usage=dict(input_tokens=len(prompt.encode()) // 4,
                              output_tokens=self.output_tokens))
        return encoded(raw), dict(request_id="mock-%d" % self.calls)


class AnthropicTransport:
    """Real Messages API call through the official Python SDK. The raw HTTP
    body is persisted byte for byte; only the request-id header is kept."""
    name = "anthropic-python-sdk"

    def __init__(self, key, workspace_id=None, timeout=600):
        import anthropic  # imported only in --mode real
        self.sdk_version = getattr(anthropic, "__version__", "unknown")
        self.workspace_id = workspace_id
        # A key not scoped to a workspace needs this header on every request.
        # The header is set on the client only; request.json stores the body.
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        self._client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=timeout,
                                           default_headers=headers)

    def send(self, body):
        # output_config goes through extra_body so an SDK release without a
        # typed field still sends the same JSON; the persisted request is the body.
        kwargs = dict(body)
        extra = {k: kwargs.pop(k) for k in ("output_config",) if k in kwargs}
        raw = self._client.messages.with_raw_response.create(**kwargs, extra_body=extra or None)
        http = raw.http_response
        return bytes(http.content), dict(request_id=http.headers.get("request-id"))


def redact(text, secret):
    return text.replace(secret, "<api-key-removed>") if secret else text


def generate_cell(run, manifest, transport, cell, context, secret, prices, engine=None):
    arm, rnd = cell["arm"], cell["round"]
    base_dir = cell_dir(cell)
    if (run.root / base_dir / "batch.json").exists():
        return
    feedback = feedback_lines(run, arm, rnd) if arm in CLAUDE_ARMS else []
    order = deterministic_order() if arm == "deterministic-search" else None
    dedup = manifest.get("history_mode", "none") != "none"
    for index in range(manifest["per_arm"]):
        stem = "%s/%03d" % (base_dir, index)
        proposal_rel = stem + ".proposal.json"
        if (run.root / proposal_rel).exists():
            if dedup:
                record_duplicate(run, base_dir, index)
            continue
        if arm == "deterministic-search":
            code = order[rnd * manifest["per_arm"] + index]
            run.write(proposal_rel, encoded(template(digest(context)) | dict(phase=code)))
            run.event("generate", proposal_rel)
            if dedup:
                record_duplicate(run, base_dir, index)
            continue
        history = history_lines(run, manifest, engine, base_dir, index) if dedup else None
        body = request_body(manifest, context, arm, rnd, index, feedback, history)
        request_rel, response_rel = stem + ".request.json", stem + ".response.json"
        if (run.root / request_rel).exists() and not (run.root / response_rel).exists():
            raise RuntimeError("interrupted call: request retained without response; "
                               "not re-issuing a paid call automatically: " + request_rel)
        if not (run.root / response_rel).exists():
            spent = sum(x["cost_usd"] for x in run.rows("ledger.jsonl"))
            reserve = worst_case_cost(body, prices)
            if spent + reserve > manifest["budget_usd"]:
                run.append("ledger.jsonl", dict(event="BUDGET_STOP", cell=base_dir, index=index,
                                                spent_usd=round(spent, 6),
                                                reserve_usd=round(reserve, 6),
                                                cap_usd=manifest["budget_usd"], cost_usd=0.0))
                raise BudgetExhausted("budget cap US$%.2f: spent %.4f + worst case %.4f"
                                      % (manifest["budget_usd"], spent, reserve))
            request_bytes = encoded(body)
            run.write(request_rel, request_bytes)
            run.event("request", request_rel)
            started = time.time()
            try:
                raw, meta = transport.send(body)
            except Exception as error:  # record the failure, never the key
                run.append("ledger.jsonl", dict(event="CALL_ERROR", cell=base_dir, index=index,
                                                error=type(error).__name__,
                                                message=redact(str(error), secret)[:500],
                                                cost_usd=0.0))
                raise RuntimeError("API call failed (%s): %s" % (
                    type(error).__name__, redact(str(error), secret)[:500])) from None
            run.write(response_rel, raw)
            run.event("response", response_rel)
            text, value = response_text(raw)
            usage = value.get("usage") or {}
            cost = call_cost(usage, prices)
            spent += cost
            run.append("ledger.jsonl", dict(
                event="CALL", cell=base_dir, index=index, transport=transport.name,
                sdk_version=transport.sdk_version,
                **({"anthropic_workspace_id": transport.workspace_id} if transport.workspace_id else {}),
                model_requested=body["model"], model_returned=value.get("model"),
                request_sha256=digest(request_bytes), response_sha256=digest(raw),
                request_id=meta.get("request_id"), stop_reason=value.get("stop_reason"),
                input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
                cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                cost_usd=round(cost, 6), cumulative_usd=round(spent, 6),
                seconds=round(time.time() - started, 3)))
        text, _ = response_text((run.root / response_rel).read_bytes())
        # No repair, fence stripping or JSON normalisation: Sounio judges the text.
        run.write(proposal_rel, (text or "").encode())
        run.event("generate", proposal_rel)
        if dedup:
            record_duplicate(run, base_dir, index)
    entries = []
    for index in range(manifest["per_arm"]):
        rel = "%s/%03d.proposal.json" % (base_dir, index)
        entries.append(dict(index=index, proposal=rel, sha256=digest((run.root / rel).read_bytes())))
    batch = dict(schema=1, cell=base_dir, arm=arm, round=rnd, proposals=entries,
                 batch_sha256=digest(encoded(entries)), recorded_before_admission=True)
    run.write(base_dir + "/batch.json", encoded(batch))
    run.event("batch", base_dir + "/batch.json")


def admit_cell(run, manifest, engine, cell):
    base_dir = cell_dir(cell)
    batch = json.loads((run.root / base_dir / "batch.json").read_text())
    if digest(encoded(batch["proposals"])) != batch["batch_sha256"]:
        raise ValueError("batch record inconsistent: " + base_dir)
    for entry in batch["proposals"]:
        proposal = run.root / entry["proposal"]
        if digest(proposal.read_bytes()) != entry["sha256"]:
            raise ValueError("proposal changed after its batch hash: " + entry["proposal"])
        receipt_rel = entry["proposal"].replace(".proposal.json", ".receipt.json")
        if (run.root / receipt_rel).exists():
            continue
        result = subprocess.run([str(engine), str(run.root / "context.json"), str(proposal)],
                                capture_output=True, timeout=120)
        value = json.loads(result.stdout)
        if value.get("authority") != "Sounio" or result.returncode not in (0, 1) \
                or (result.returncode == 0) != (value.get("decision") == "ADMIT"):
            raise ValueError("invalid admission result for " + entry["proposal"])
        if value.get("decision") == "ADMIT" and (value.get("scientific_novelty") is not False
                                                 or value.get("claim_ready") is not False):
            raise ValueError("admission receipt promotes a claim; refusing to record it")
        run.write(receipt_rel, result.stdout)
        run.event("admit", receipt_rel)


def summarize(run, manifest):
    ledger = run.rows("ledger.jsonl")
    out = dict(schema=1, condition_id=manifest["condition_id"], model=manifest["model"],
               mode=manifest["mode"], cells=[], scientific_novelty=False, claim_ready=False,
               semantic_authority="Sounio admission + eisa_h.pireus_operator_atlas.v1")
    per_arm_classes = {arm: set() for arm in ARMS}
    for cell in cells(manifest["rounds"]):
        base_dir = cell_dir(cell)
        receipts = [json.loads(p.read_text()) for p in sorted((run.root / base_dir).glob("*.receipt.json"))]
        admitted = [r for r in receipts if r.get("decision") == "ADMIT"]
        classes = {r["canonical_table"] for r in admitted if r.get("canonical_table")}
        new = classes - per_arm_classes[cell["arm"]]
        per_arm_classes[cell["arm"]] |= classes
        reasons = {}
        for r in receipts:
            if r.get("decision") != "ADMIT":
                reasons[r.get("reason")] = reasons.get(r.get("reason"), 0) + 1
        calls = [x for x in ledger if x.get("event") == "CALL" and x["cell"] == base_dir]
        duplicates = sorted({x["index"] for x in ledger if x.get("event") == "DUPLICATE" and x["cell"] == base_dir})
        if duplicates:
            reasons["DUPLICATE"] = len(duplicates)
        unique = [json.loads((run.root / ("%s/%03d.receipt.json" % (base_dir, i))).read_text())
                  for i in range(manifest["per_arm"])
                  if i not in duplicates and (run.root / ("%s/%03d.receipt.json" % (base_dir, i))).exists()]
        batch = run.root / base_dir / "batch.json"
        out["cells"].append(dict(
            cell=base_dir, arm=cell["arm"], round=cell["round"],
            batch_sha256=json.loads(batch.read_text())["batch_sha256"] if batch.exists() else None,
            validated=len(receipts), admitted=len(admitted), refused=reasons,
            novelty={lvl: sum(1 for r in admitted if r.get("novelty") == lvl) for lvl in ("N0", "N1", "N2")},
            canonical_classes=len(classes), new_classes_for_arm=len(new),
            duplicates=duplicates,
            admitted_after_dedup=sum(1 for r in unique if r.get("decision") == "ADMIT"),
            calls=len(calls), input_tokens=sum(x["input_tokens"] or 0 for x in calls),
            output_tokens=sum(x["output_tokens"] or 0 for x in calls),
            cost_usd=round(sum(x["cost_usd"] for x in calls), 6)))
    out["classes_per_arm"] = {arm: len(v) for arm, v in per_arm_classes.items()}
    out["models_returned"] = sorted({x.get("model_returned") or "?" for x in ledger if x.get("event") == "CALL"})
    out["cost_usd"] = round(sum(x.get("cost_usd", 0.0) for x in ledger), 6)
    out["budget_usd"] = manifest["budget_usd"]
    out["stopped"] = [x for x in ledger if x.get("event") in ("BUDGET_STOP", "CALL_ERROR")]
    return out


def estimate(manifest, context, prices):
    calls, worst, expected, tokens_in = 0, 0.0, 0.0, 0
    feedback_guess = ["r0 phase=65535 ADMIT - N2 class-0123456789ab"] * manifest["per_arm"]
    history_on = manifest.get("history_mode", "none") != "none"
    for cell in cells(manifest["rounds"]):
        if cell["arm"] not in CLAUDE_ARMS:
            continue
        fb = feedback_guess * cell["round"]
        for index in range(manifest["per_arm"]):
            history = ["phase=65535 class-0123456789ab"] * index if history_on else None
            body = request_body(manifest, context, cell["arm"], cell["round"], index, fb, history)
            size = len(json.dumps(body).encode())
            exp_in = int(size / BYTES_PER_TOKEN_EXPECTED) + 64
            calls += 1
            tokens_in += exp_in
            expected += (exp_in * prices[0] + EXPECTED_OUTPUT_TOKENS * prices[1]) / 1e6
            worst += worst_case_cost(body, prices)
    return dict(calls=calls, expected_input_tokens=tokens_in,
                expected_output_tokens=calls * EXPECTED_OUTPUT_TOKENS,
                expected_usd=round(expected, 2), worst_case_usd=round(worst, 2),
                assumptions=dict(bytes_per_token=BYTES_PER_TOKEN_EXPECTED,
                                 output_tokens_per_call=EXPECTED_OUTPUT_TOKENS,
                                 worst_case="request bytes as tokens + max_tokens output",
                                 feedback="full feedback of every earlier round of the arm",
                                 history="every earlier proposal of the arm and round" if history_on else "none"),
                prices_usd_per_mtok=dict(input=prices[0], output=prices[1]))


def build_manifest(args, context, engine, prices, price_source):
    history = {}
    family = CONDITION_FAMILY
    if args.history != "none":
        family = CONDITION_FAMILY_V2
        history = dict(history_mode=args.history,
                       history_head_sha256=digest(HISTORY_HEAD.encode()),
                       distinct_ask_sha256=digest(DISTINCT_ASK.encode()),
                       dedup_policy=DEDUP_POLICY, dedup_policy_sha256=digest(DEDUP_POLICY.encode()),
                       history_same_form_both_claude_arms=True)
        if args.history == "proposals+classes":
            history["intra_round_preview"] = PREVIEW_NOTE
    return dict(
        schema=1, condition_id=family + ":" + args.model,
        condition_note="new declared proposer condition; NOT the PIREUS_CONTINUITY_PLAN Inkling pilot",
        arms=list(ARMS), rounds=args.rounds, per_arm=args.per_arm, kind=2,
        model=args.model, same_model_for_both_claude_arms=True,
        max_tokens=args.max_tokens, effort=args.effort, temperature=args.temperature,
        sampling_note="temperature omitted unless given: claude-sonnet-5-5 rejects temperature/top_p/top_k",
        api_seed=None, seed_note="the Messages API has no seed; the proposal index and round are in the prompt",
        prompt_sha256=digest(PROMPT.encode()), ontology_legend_sha256=digest(ONTOLOGY_LEGEND.encode()),
        feedback_head_sha256=digest(FEEDBACK_HEAD.encode()),
        deterministic_order_sha256=digest(encoded(deterministic_order())),
        budget_usd=args.budget_usd, prices_usd_per_mtok=list(prices), price_source=price_source,
        context_sha256=digest(context), context_origin="research_context.sio (TripleStore/SPARQL)",
        admission_engine_sha256=digest(engine.read_bytes()),
        admission_source_sha256=digest((HERE.parent / "admission.sio").read_bytes()),
        atlas_source_sha256=digest((HERE.parents[3] / "stdlib/hardware/pireus/operator_atlas.sio").read_bytes()),
        proposer_source_sha256=digest(Path(__file__).read_bytes()),
        mode=args.mode, novelty_oracle="Sounio only", scientific_novelty=False, claim_ready=False,
        # Only present when given, so manifests without a workspace are unchanged.
        **({"anthropic_workspace_id": args.workspace_id} if args.workspace_id else {}),
        # Absent with --history none, so that mode keeps the v1 manifest keys.
        **history)


def make_transport(args, secret):
    if args.mode == "mock":
        return MockTransport()
    if not args.confirm_spend:
        raise SystemExit("--mode real needs --confirm-spend")
    if not secret:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    return AnthropicTransport(secret, args.workspace_id)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["run", "estimate", "report"])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--engine", type=Path, help="native Sounio admission executable")
    ap.add_argument("--context-engine", type=Path, help="research_context.sio executable")
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max", ""])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    ap.add_argument("--price-input", type=float)
    ap.add_argument("--price-output", type=float)
    ap.add_argument("--rounds", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--per-arm", type=int, choices=[8, 32], default=32)
    ap.add_argument("--confirm-spend", action="store_true")
    ap.add_argument("--history", choices=HISTORY_MODES, default=DEFAULT_HISTORY,
                    help="intra-round history of the arm's own proposals (none = v1 behaviour)")
    ap.add_argument("--workspace-id", default=os.environ.get("ANTHROPIC_WORKSPACE_ID") or None,
                    help="sent as the anthropic-workspace-id header (default: env ANTHROPIC_WORKSPACE_ID)")
    args = ap.parse_args(argv)
    secret = os.environ.get("ANTHROPIC_API_KEY") if args.mode == "real" else None
    prices, price_source = prices_for(args)
    if args.command == "report":
        run = Run(args.run)
        print(json.dumps(summarize(run, json.loads((run.root / "manifest.json").read_text())), indent=2))
        return 0
    if not args.context_engine or not args.engine:
        ap.error("--engine and --context-engine are required")
    context = subprocess.run([str(args.context_engine.resolve())], capture_output=True,
                             check=True, timeout=60).stdout
    manifest = build_manifest(args, context, args.engine.resolve(), prices, price_source)
    if args.command == "estimate":  # dry run: no transport, no key, no files
        print(json.dumps(dict(condition_id=manifest["condition_id"], model=args.model,
                              budget_usd=args.budget_usd, **estimate(manifest, context, prices)), indent=2))
        return 0
    if not args.run:
        ap.error("--run is required")
    run = Run(args.run, secret)
    run.root.mkdir(parents=True, exist_ok=True)
    if (run.root / "manifest.json").exists():
        old = json.loads((run.root / "manifest.json").read_text())
        if old != manifest:
            changed = sorted(k for k in manifest if old.get(k) != manifest[k])
            raise SystemExit("frozen run manifest differs (%s); a changed model, prompt, engine or "
                             "budget is a new run" % ",".join(changed))
    run.write("context.json", context)
    run.write("manifest.json", encoded(manifest))
    transport = make_transport(args, secret)
    status = 0
    try:
        for rnd in range(args.rounds):
            round_cells = [c for c in cells(args.rounds) if c["round"] == rnd]
            for cell in round_cells:   # generate the whole round, then admit it
                generate_cell(run, manifest, transport, cell, context, secret, prices,
                              args.engine.resolve())
            for cell in round_cells:
                admit_cell(run, manifest, args.engine.resolve(), cell)
    except BudgetExhausted as stop:
        print("BUDGET_STOP " + str(stop), file=sys.stderr)
        status = 3
    report = summarize(run, manifest)
    run.write("report-%s.json" % digest(encoded(report))[:16], encoded(report))
    print(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
