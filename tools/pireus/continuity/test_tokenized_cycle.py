#!/usr/bin/env python3
"""Transport controls: reject rank disagreement and preserve invalid raw model text."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from cycle import HERE, REVISION, digest, encoded, verify
from tokenized_cycle import pack_encode, pair, pack_decode, finalize, issue_once, pack_offline, accept_offline

class TokenTransportTests(unittest.TestCase):
    def test_interrupted_http_request_is_never_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            request, response = root / "000.token.request.json", root / "000.token.response.json"
            with patch("tokenized_cycle.urllib.request.urlopen", side_effect=TimeoutError("lost reply")) as http:
                with self.assertRaises(TimeoutError):
                    issue_once(root, request, response, "http://127.0.0.1:30000", {"input_ids": [1]})
                with self.assertRaisesRegex(RuntimeError, "never automatically replay"):
                    issue_once(root, request, response, "http://127.0.0.1:30000", {"input_ids": [1]})
                self.assertEqual(http.call_count, 1)
                self.assertTrue(request.exists())
                self.assertFalse(response.exists())

    def test_pair_binding_and_exact_invalid_text(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = dict(budget=8, condition="inkling-ontology", round=0)
            (root / "context.json").write_bytes(b'{"research":"declared"}')
            pack_encode(root, manifest)
            bundle = root / "encode-bundle.json"
            items = [dict(index=i, input_ids=[1, 2, i], stop_token_ids=[3],
                          rendered_sha256="a" * 64) for i in range(8)]
            base = dict(schema=1, stage="TOKENIZER_TRANSPORT", job="123", mode="encode",
                        revision=REVISION, input_sha256=digest(bundle.read_bytes()),
                        helper_sha256=digest((HERE / "runtime/tokenizer_transport.py").read_bytes()),
                        items=items)
            paths = [root / ("input-rank-" + str(i) + ".json") for i in range(2)]
            for i, p in enumerate(paths):
                p.write_bytes(encoded(base | dict(rank=str(i))))
            changed = json.loads(paths[1].read_bytes())
            changed["items"][0]["input_ids"] = [999]
            paths[1].write_bytes(encoded(changed))
            with self.assertRaisesRegex(ValueError, "results differ"):
                pair(root, "encode", paths, manifest)
            paths[1].write_bytes(encoded(base | dict(rank="1")))
            pair(root, "encode", paths, manifest)
            manifest["transport"] = "sglang-offline-token-ids"
            (root / "manifest.json").write_bytes(encoded(manifest))
            pack_offline(root, manifest)
            worker = root / "worker"
            worker.mkdir()
            input_sha = digest((root / "offline-bundle.json").read_bytes())
            results = []
            for i in range(8):
                response = dict(schema=1, transport="sglang-offline-token-ids", index=i,
                                output_ids=[1, i], job="fixture-123", revision=REVISION,
                                input_sha256=input_sha)
                raw = encoded(response)
                for rank in range(2):
                    (worker / ("rank-%d-%03d.json" % (rank, i))).write_bytes(raw)
                results.append(dict(index=i, response_sha256=digest(raw)))
            for rank in range(2):
                completion = dict(rank=str(rank), input_sha256=input_sha, revision=REVISION,
                    helper_sha256=digest((HERE / "runtime/offline_generate.py").read_bytes()),
                    model_loaded=True, job="fixture-123", results=results)
                (worker / ("rank-%d-complete.json" % rank)).write_bytes(encoded(completion))
            bad = worker / "rank-1-000.json"
            good = bad.read_bytes()
            bad.write_bytes(good + b" ")
            with self.assertRaisesRegex(ValueError, "token response disagreement"):
                accept_offline(root, manifest, worker)
            bad.write_bytes(good)
            accept_offline(root, manifest, worker)
            pack_decode(root, manifest)
            text = "  invalid model JSON\n{not repaired}\n"
            decode_items = [dict(index=i, text=text, text_with_special_tokens="<marker>" + text,
                                token_response_sha256=digest((root / ("%03d.token.response.json" % i)).read_bytes()))
                            for i in range(8)]
            base.update(mode="decode", input_sha256=digest((root / "decode-bundle.json").read_bytes()),
                        items=decode_items)
            for i, p in enumerate(paths):
                p.write_bytes(encoded(base | dict(rank=str(i))))
            finalize(root, manifest, paths)
            self.assertTrue(all(p.read_bytes() == text.encode() for p in root.glob("*.proposal.json")))
            before = (root / "journal.jsonl").read_bytes()
            finalize(root, manifest, paths)
            self.assertEqual(before, (root / "journal.jsonl").read_bytes())

if __name__ == "__main__":
    unittest.main()
