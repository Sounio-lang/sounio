#!/usr/bin/env python3
"""Bound the pinned Marlin offline gate/up rearrangement to one expert."""
import ast
import hashlib

SOURCE_SHA256 = "26b999da4d72fd238c32331782f72b6aa110165adec96fb8885f4252a5a7099c"
HELPER = '''
def _pireus_expert_deinterleave(weight, *, up_first=False):
    if not get_moe_runner_backend().is_marlin():
        return deinterleave_w13(weight, up_first=up_first)
    assert weight.ndim == 3 and weight.shape[0] > 0
    assert weight.is_contiguous()
    assert weight.dtype in (torch.uint8, torch.float8_e4m3fn)
    for index in range(weight.shape[0]):
        transformed = deinterleave_w13(weight[index], up_first=up_first)
        assert transformed.shape == weight[index].shape
        assert transformed.dtype == weight.dtype
        weight[index].copy_(transformed)
    return weight
'''

def patched_source(data):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unrecognized placeholder-patched source")
    source = data.decode()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
               and n.name == "ModelOptNvFp4FusedMoEMethod")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "process_weights_after_loading")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[fn.lineno-1:fn.end_lineno])
    if body.count("= deinterleave_w13(") != 2:
        raise ValueError("Unexpected offline deinterleave calls")
    body = body.replace("= deinterleave_w13(", "= _pireus_expert_deinterleave(")
    lines[fn.lineno-1:fn.end_lineno] = [body]
    result = "".join(lines) + "\n" + HELPER
    ast.parse(result)
    return result.encode()
