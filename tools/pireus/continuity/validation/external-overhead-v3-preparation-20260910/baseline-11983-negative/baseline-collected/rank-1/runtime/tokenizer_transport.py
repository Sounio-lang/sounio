#!/usr/bin/env python3
"""Encode/decode only, in the pinned image under an owned Slurm allocation."""
from collections.abc import Mapping
import argparse
import hashlib
import json
import os
from pathlib import Path

REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"
MODEL = Path("/scratch/pireus/models/Inkling-Small-NVFP4") / REVISION

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    args = ap.parse_args()
    job, rank = os.environ["SLURM_JOB_ID"], os.environ["PIREUS_RANK"]
    if not job.isdecimal() or rank not in ("0", "1"):
        raise ValueError("owned pair required")
    raw = args.input.read_bytes()
    bundle = json.loads(raw)
    if bundle["revision"] != REVISION or bundle["mode"] not in ("encode", "decode"):
        raise ValueError("unrecognized tokenizer bundle")
    files = []
    for entry in json.loads(Path("/scratch/pireus/runtime/inkling-files.json").read_text()):
        name = entry["rfilename"]
        if name.endswith(".safetensors") or name in (".gitattributes", "README.md"):
            continue
        data = (MODEL / name).read_bytes()
        expected = entry.get("lfs", {}).get("sha256")
        actual = digest(data) if expected else hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) != entry["size"] or actual != (expected or entry["blobId"]):
            raise ValueError("tokenizer file identity: " + name)
        files.append(dict(file=name, sha256=digest(data)))
    from transformers import AutoTokenizer
    import transformers, tokenizers
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=True)
    items = []
    for item in bundle["items"]:
        if bundle["mode"] == "encode":
            kwargs = dict(add_generation_prompt=True, reasoning_effort="none")
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, **kwargs)
            ids = tokenizer.apply_chat_template(item["messages"], tokenize=True, **kwargs)
            ids = ids["input_ids"] if isinstance(ids, Mapping) else ids
            if not isinstance(ids, list) or not all(type(x) is int and x >= 0 for x in ids):
                raise ValueError("unexpected tokenizer return type")
            if not ids or len(ids) + item["max_tokens"] > 16384:
                raise ValueError("context budget exceeded")
            stops = tokenizer.eos_token_id
            stops = stops if isinstance(stops, list) else [stops]
            # This model's end-of-turn marker must terminate a token-only response.
            marker = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
            if marker is None or marker == tokenizer.unk_token_id:
                raise ValueError("missing model end-of-turn token")
            stops = sorted(set([x for x in stops if x is not None] + [marker]))
            items.append(dict(index=item["index"], input_ids=ids, stop_token_ids=stops,
                              rendered_sha256=digest(text.encode())))
        else:
            ids = item["output_ids"]
            if not isinstance(ids, list) or not all(type(x) is int and x >= 0 for x in ids):
                raise ValueError("invalid output token IDs")
            items.append(dict(index=item["index"], token_response_sha256=item["token_response_sha256"],
                              text=tokenizer.decode(ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False),
                              text_with_special_tokens=tokenizer.decode(ids, skip_special_tokens=False,
                                                                        clean_up_tokenization_spaces=False)))
    receipt = dict(schema=1, stage="TOKENIZER_TRANSPORT", job=job, rank=rank,
                   mode=bundle["mode"], revision=REVISION, input_sha256=digest(raw),
                   helper_sha256=digest(Path(__file__).read_bytes()),
                   transformers_version=transformers.__version__, tokenizers_version=tokenizers.__version__,
                   tokenizer_files=files, items=items)
    out = Path("/scratch/pireus/receipts") / ("tokenizer-" + job + "-" + rank + ".json")
    serialized = json.dumps(receipt, sort_keys=True) + "\n"
    with out.open("x") as f:
        f.write(serialized)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(stage="TOKENIZER_TRANSPORT_PASS", job=job, rank=rank,
                          mode=bundle["mode"], count=len(items), receipt_sha256=digest(out.read_bytes()))),
          flush=True)

if __name__ == "__main__":
    main()
