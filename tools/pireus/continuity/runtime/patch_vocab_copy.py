#!/usr/bin/env python3
"""Pinned vocabulary row copies through an owned bounded CPU buffer."""
import ast
import hashlib
SOURCE_SHA256 = "f2169ba006a436434cc793a462c3b12a1cd8594266e2e2ce322386a7e18b812b"
HELPER = '''
def _pireus_vocab_copy(destination, source):
    import os
    if (os.environ.get("PIREUS_OFFLINE_MODE") == "generate"
        and destination.ndim == source.ndim == 2
        and destination.dtype == source.dtype == torch.bfloat16
        and destination.device.type == "cuda" and source.device.type == "cpu"):
        assert destination.shape == source.shape
        assert source.shape[1] > 0
        if source.shape[0] == 0:
            return
        row_bytes = source.shape[1] * source.element_size()
        assert row_bytes <= 4*1024**2
        rows = min(source.shape[0], (4*1024**2)//row_bytes)
        staging = torch.empty((rows,source.shape[1]),dtype=source.dtype,device="cpu")
        for start in range(0,source.shape[0],rows):
            stop = min(start+rows,source.shape[0])
            slab = staging[:stop-start]
            slab.copy_(source[start:stop])
            destination[start:stop].copy_(slab)
    else:
        destination.copy_(source)
'''
def patched_source(data):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unrecognized pinned vocabulary source")
    source = data.decode()
    old = "param[: loaded_weight.shape[0]].data.copy_(loaded_weight)"
    assert source.count(old) == 1
    source = source.replace(old, "_pireus_vocab_copy(param[: loaded_weight.shape[0]].data, loaded_weight)")
    result = source + "\n" + HELPER
    ast.parse(result)
    return result.encode()
