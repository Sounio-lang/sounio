#!/usr/bin/env python3
"""Pinned expert-local Marlin repacking with the same transformed tensor bytes."""
import ast
import hashlib

SOURCE_SHA256 = "dbef823f827745342ae0ed5a34361da123654ede64eccb0b8d08a5dd5abfa864"

WEIGHT_BODY = '''
    def _repack_weight(weight: torch.Tensor, is_w13: bool) -> torch.Tensor:
        if is_w13:
            size_n, size_k = intermediate_size * num_shards, hidden_size
        else:
            size_n, size_k = hidden_size, intermediate_size
        assert weight.shape == (num_experts, size_n, size_k // 2)
        assert weight.dtype == torch.uint8
        assert weight.is_contiguous()
        output = None
        for i in range(num_experts):
            qweight = weight[i].view(torch.int32).T.contiguous()
            packed = gptq_marlin_repack(
                b_q_weight=qweight, perm=perm, size_k=size_k,
                size_n=size_n, num_bits=4,
            )
            assert packed.dtype == torch.int32
            assert packed.numel() * packed.element_size() == weight[i].numel()
            if output is None:
                output = weight.view(torch.int32).view(num_experts, *packed.shape)
                assert output.data_ptr() == weight.data_ptr()
            output[i].copy_(packed)
        assert output is not None
        return output
'''

SCALE_BODY = '''
    def _permute_scales(scales: torch.Tensor, is_w13: bool) -> torch.Tensor:
        if is_w13:
            size_n, size_k = intermediate_size * num_shards, hidden_size
        else:
            size_n, size_k = hidden_size, intermediate_size
        assert scales.dtype == torch.float8_e4m3fn
        assert scales.is_contiguous()
        output = None
        for i in range(num_experts):
            scale = scales[i].to(param_dtype).T.contiguous()
            permuted = marlin_permute_scales(
                s=scale, size_k=size_k, size_n=size_n, group_size=16,
            )
            packed = nvfp4_marlin_process_scales(permuted)
            assert packed.dtype == scales.dtype
            assert packed.numel() == scales[i].numel()
            if output is None:
                output = scales.view(num_experts, *packed.shape)
                assert output.data_ptr() == scales.data_ptr()
            output[i].copy_(packed)
        assert output is not None
        return output
'''

def patched_source(data):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Refusing an unrecognized Marlin source")
    text = data.decode()
    tree = ast.parse(text)
    parent = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "prepare_moe_nvfp4_layer_for_marlin")
    replacements = {"_repack_weight": WEIGHT_BODY, "_permute_scales": SCALE_BODY}
    nodes = [n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name in replacements]
    if len(nodes) != 2:
        raise ValueError("Unexpected nested repack structure")
    lines = text.splitlines(keepends=True)
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        lines[node.lineno-1:node.end_lineno] = [replacements[node.name].strip("\n") + "\n"]
    result = "".join(lines)
    ast.parse(result)
    return result.encode()
