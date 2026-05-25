#!/usr/bin/env python3
"""Generate Sounio model completions in-process for a prompt set.

This is the GPU-friendly companion to `sounio_ai_eval.py`: it keeps each model
resident while generating all requested prompts, then writes raw completions in
the directory layout consumed by `sounio_ai_eval.py --completions-dir`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from local_generate_sounio import DEFAULT_BASE_MODEL, build_prompt, is_probable_adapter, torch_dtype
from sounio_ai_eval import DEFAULT_MODELS, DEFAULT_PROMPTS, ModelSpec, Prompt, load_prompts, parse_model_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="KEY=MODEL_ID",
        help="Model to generate. Repeatable. Defaults to Qwen base and Sounio LoRA ids.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Generate N independent samples per prompt. Files are named prompt.sampleN.sio when N > 1.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--deterministic-first-sample",
        action="store_true",
        help="When generating multiple samples, force sample0 to temperature 0 and sample the remaining candidates.",
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--prompt-style",
        default="chat",
        choices=["chat", "plain", "instruction", "source_only"],
        help="Prompt format to use for generation.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def model_route(spec: ModelSpec, base_model: str) -> tuple[str, str | None]:
    if is_probable_adapter(spec.model_id, None, base_model):
        return base_model, spec.model_id
    return spec.model_id, None


def load_base_model(model_id: str, args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def with_adapter(model, adapter_id: str):
    try:
        from peft import PeftModel
    except ModuleNotFoundError as exc:
        raise SystemExit("missing dependency for LoRA generation; install peft") from exc
    adapted = PeftModel.from_pretrained(model, adapter_id)
    adapted.eval()
    return adapted


def generate_for_model(
    *,
    tokenizer,
    model,
    spec: ModelSpec,
    prompts: list[Prompt],
    args: argparse.Namespace,
) -> None:
    import torch

    model_dir = args.output_dir / spec.key
    model_dir.mkdir(parents=True, exist_ok=True)

    def generation_kwargs_for(temperature: float) -> dict[str, object]:
        do_sample = temperature > 0.0
        kwargs: dict[str, object] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = args.top_p
        return kwargs

    if args.deterministic_first_sample and args.samples_per_prompt > 1:
        jobs_by_temperature = [
            (0.0 if sample_index == 0 else args.temperature, [(prompt, sample_index) for prompt in prompts])
            for sample_index in range(args.samples_per_prompt)
        ]
    else:
        jobs = [
            (prompt, sample_index)
            for prompt in prompts
            for sample_index in range(args.samples_per_prompt)
        ]
        jobs_by_temperature = [(args.temperature, jobs)]

    for temperature, jobs in jobs_by_temperature:
        generation_kwargs = generation_kwargs_for(temperature)
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start : start + args.batch_size]
            texts = [build_prompt(tokenizer, prompt.prompt, args.prompt_style) for prompt, _sample_index in batch]
            encoded = tokenizer(texts, return_tensors="pt", padding=True)
            input_width = encoded["input_ids"].shape[-1]
            if hasattr(model, "device"):
                encoded = encoded.to(model.device)
            batch_start = time.monotonic()
            with torch.no_grad():
                output_ids = model.generate(**encoded, **generation_kwargs)
            elapsed = time.monotonic() - batch_start
            for (prompt, sample_index), row in zip(batch, output_ids, strict=True):
                completion = tokenizer.decode(row[input_width:], skip_special_tokens=True).strip()
                suffix = f".sample{sample_index}" if args.samples_per_prompt > 1 else ""
                out = model_dir / f"{prompt.prompt_id}{suffix}.sio"
                out.write_text(completion + ("\n" if completion else ""), encoding="utf-8")
                print(
                    f"{spec.key:>8} {prompt.prompt_id:<40} sample={sample_index:<2} generated "
                    f"{len(completion)} chars temp={temperature:g} batch_elapsed={elapsed:.1f}s",
                    flush=True,
                )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.samples_per_prompt < 1:
        raise SystemExit("--samples-per-prompt must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts, args.limit)
    specs = parse_model_specs(args.model or DEFAULT_MODELS)
    routes = [(spec, *model_route(spec, args.base_model)) for spec in specs]

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("missing dependency for batch generation; install torch") from exc

    loaded_base_id: str | None = None
    tokenizer = None
    base_model = None
    for spec, base_id, adapter_id in routes:
        if loaded_base_id != base_id:
            tokenizer, base_model = load_base_model(base_id, args)
            loaded_base_id = base_id
        assert tokenizer is not None
        assert base_model is not None
        active_model = with_adapter(base_model, adapter_id) if adapter_id else base_model
        print(
            f"[model] key={spec.key} base={base_id} adapter={adapter_id or '<none>'} "
            f"prompts={len(prompts)} samples_per_prompt={args.samples_per_prompt}",
            file=sys.stderr,
            flush=True,
        )
        generate_for_model(
            tokenizer=tokenizer,
            model=active_model,
            spec=spec,
            prompts=prompts,
            args=args,
        )
        if adapter_id and hasattr(active_model, "disable_adapter"):
            active_model.disable_adapter()


if __name__ == "__main__":
    main()
