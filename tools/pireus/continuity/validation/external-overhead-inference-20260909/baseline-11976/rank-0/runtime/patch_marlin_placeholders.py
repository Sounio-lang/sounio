#!/usr/bin/env python3
"""Custody-bound removal of unused Marlin blockscale placeholders."""
import ast
import hashlib

SOURCE_SHA256 = "0b87d0fb15b1929f490fbc6f224ab5e1e0e11fee848d01e5ee4c511b4c0f4343"

def patched_source(data):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Refusing an unrecognized ModelOpt source")
    source = data.decode()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "ModelOptNvFp4FusedMoEMethod")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "create_weights")
    lines = source.splitlines(keepends=True)
    start, end = method.lineno - 1, method.end_lineno
    body = "".join(lines[start:end])
    old = "if self.enable_flashinfer_trtllm_moe:"
    if body.count(old) != 2:
        raise ValueError("Unexpected blockscale allocation structure")
    body = body.replace(old, "if self.enable_flashinfer_trtllm_moe or get_moe_runner_backend().is_marlin():")
    result = "".join(lines[:start]) + body + "".join(lines[end:])
    ast.parse(result)
    return result.encode()
