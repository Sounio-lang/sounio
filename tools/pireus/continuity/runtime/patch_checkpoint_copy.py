#!/usr/bin/env python3
"""Bound pinned Marlin offline checkpoint copies, preserving TP slicing."""
import ast
import hashlib
SOURCE_SHA256 = "c0d571f5b327e36139479a05b1ac530eeac341fd2fe54584d09b5339b4c530a5"
HELPER = '''
def _pireus_checkpoint_copy(destination, source):
    if (get_moe_runner_backend().is_marlin() and destination.ndim == 3
        and destination.dtype == torch.uint8 and source.device.type == "cpu"
        and destination.device.type == "cuda"):
        assert destination.shape == source.shape
        assert source.dtype == destination.dtype
        assert destination.shape[0] > 0
        # CUDA reads an owned CPU staging buffer, never the checkpoint mmap.
        # Reuse it only after the blocking copy returns.
        staging = torch.empty(destination.shape[1:], dtype=destination.dtype, device="cpu")
        for index in range(destination.shape[0]):
            staging.copy_(source[index])
            destination[index].copy_(staging)
    else:
        destination.copy_(source)
'''
def patched_source(data):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unrecognized pinned FusedMoE source")
    source = data.decode()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoE")
    lines = source.splitlines(keepends=True)
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("_load_w13", "_load_w2")]
    assert len(functions) == 2
    for fn in sorted(functions, key=lambda n:n.lineno, reverse=True):
        body = "".join(lines[fn.lineno-1:fn.end_lineno])
        assert body.count("expert_data.copy_(loaded_weight)") == 1
        body = body.replace("expert_data.copy_(loaded_weight)", "_pireus_checkpoint_copy(expert_data, loaded_weight)")
        if fn.name == "_load_w2":
            old = "(expert_data.dim() != 2 or loaded_weight.dim() != 2)"
            assert body.count(old) == 1
            # The fused caller passes 3D stacks; the ordinary expert caller 2D.
            # Extend only pinned offline packed Marlin stacks, retaining other guards.
            new = """(expert_data.dim() != 2 or loaded_weight.dim() != 2)
            and not (
                get_moe_runner_backend().is_marlin()
                and expert_data.dim() == loaded_weight.dim() == 3
                and expert_data.dtype == loaded_weight.dtype == torch.uint8
                and shard_dim == 2 and not is_bias
            )"""
            body = body.replace(old, new)
        lines[fn.lineno-1:fn.end_lineno] = [body]
    result = "".join(lines) + "\n" + HELPER
    ast.parse(result)
    return result.encode()
