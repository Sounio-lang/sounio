#!/usr/bin/env python3
"""Generate one Sounio completion with a local Transformers/PEFT backend.

The script reads one prompt from stdin and writes one completion to stdout. It
is designed to be used by `scripts/dev/sounio_ai_eval.py --generator-command`.
"""

from __future__ import annotations

import argparse
import sys


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_LORA_MODEL = "chiuratto-AIgourakis/sounio-qwen25-coder-1p5b-lora"

SYSTEM_PROMPT = """You are a Sounio programming assistant.
Return one complete Sounio source file only.
Use Sounio syntax: var for mutable bindings, &!T for exclusive references, no Rust macros, no Python imports, and no Markdown fences.
Declare effects explicitly when functions use IO, mutation, division, allocation, panic, approximation, or observation."""

SOURCE_ONLY_PROMPT = """Return raw Sounio source code only.
Do not write an introduction, explanation, Markdown fence, language label, or postscript.
The first generated line must be Sounio code or a Sounio comment."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model id or PEFT adapter id.")
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model to load when --model is a LoRA adapter.",
    )
    parser.add_argument(
        "--adapter",
        help="Explicit PEFT adapter id. If set, --model is treated as the base model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--prompt-style",
        default="chat",
        choices=["chat", "plain", "instruction", "source_only"],
        help="Prompt format to use for generation.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate argument routing without importing Transformers or downloading a model.",
    )
    return parser.parse_args()


def is_probable_adapter(model_id: str, explicit_adapter: str | None, base_model: str) -> bool:
    if explicit_adapter:
        return False
    if model_id == base_model:
        return False
    lowered = model_id.lower()
    return "lora" in lowered or "adapter" in lowered or model_id == DEFAULT_LORA_MODEL


def torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def build_prompt(tokenizer, user_prompt: str, prompt_style: str) -> str:
    if prompt_style == "instruction":
        return user_prompt.strip() + "\n\n"
    if prompt_style == "source_only":
        return (
            f"{SOURCE_ONLY_PROMPT}\n\n"
            f"Task:\n{user_prompt.strip()}\n\n"
            "Begin raw Sounio source now:\n"
        )
    if prompt_style == "plain":
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Task:\n{user_prompt.strip()}\n\n"
            "Sounio source file:\n"
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### System\n{SYSTEM_PROMPT}\n\n### Instruction\n{user_prompt}\n\n### Response\n"


def main() -> None:
    args = parse_args()
    user_prompt = sys.stdin.read().strip()
    if not user_prompt:
        raise SystemExit("no prompt received on stdin")

    adapter_id = args.adapter
    model_id = args.model
    if is_probable_adapter(args.model, args.adapter, args.base_model):
        adapter_id = args.model
        model_id = args.base_model

    if args.dry_run:
        print(
            f"dry-run: base_model={model_id} adapter={adapter_id or '<none>'} "
            f"max_new_tokens={args.max_new_tokens}",
            file=sys.stderr,
        )
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "missing dependency for local generation; install transformers and torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if adapter_id:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise SystemExit("missing dependency for LoRA generation; install peft") from exc
        model = PeftModel.from_pretrained(model, adapter_id)

    prompt_text = build_prompt(tokenizer, user_prompt, args.prompt_style)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_token_count = inputs["input_ids"].shape[-1]
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)

    do_sample = args.temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][input_token_count:]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
