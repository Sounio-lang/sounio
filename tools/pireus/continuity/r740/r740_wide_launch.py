"""r740-only launch for block_threads = 32, 64 or 128.

gpu_probe.py (shared with the Sparks, hash-pinned) launches every historical
body as grid = n, block = 16, three pointer parameters; that call is used as is
for 16. A wide body declares `.maxntid T,1,1` and a fourth parameter `pn` (the
vector count), because its grid is ceil(n / (T/16)) and %nctaid.x is no longer
n. The block size is read from the PTX text and must equal the manifest's
declared block_threads, so a manifest cannot launch a body at a size it was not
emitted for. Nothing here touches a Spark launch path, fence or manifest.
"""
import ctypes
import re

NARROW = 16
WIDE = (32, 64, 128)
MAX_VECTORS = 16777216
_MAXNTID = re.compile(rb"^\.maxntid (\d+),1,1$", re.M)
_ENTRY = re.compile(rb"^\.visible \.entry pireus_candidate\((.*)\)$", re.M)
_PARAMS = {
    NARROW: b".param .u64 pa, .param .u64 pb, .param .u64 pc",
    **{t: b".param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u32 pn" for t in WIDE},
}


def declared_threads(item):
    """block_threads from a manifest row; absent means gpu_probe.py's 16."""
    threads = item.get("block_threads", NARROW)
    if isinstance(threads, bool) or threads not in (NARROW,) + WIDE:
        raise ValueError("block_threads")
    return threads


def ptx_threads(ptx_bytes):
    """The block size the PTX was emitted for, and a check of its parameter list."""
    found = _MAXNTID.findall(ptx_bytes)
    entry = _ENTRY.findall(ptx_bytes)
    if len(found) != 1 or len(entry) != 1:
        raise ValueError("ptx geometry")
    threads = int(found[0])
    if threads not in _PARAMS or entry[0] != _PARAMS[threads]:
        raise ValueError("ptx geometry")
    return threads


def geometry(item, ptx_bytes):
    threads = declared_threads(item)
    if ptx_threads(ptx_bytes) != threads:
        raise ValueError("manifest block_threads does not match the PTX .maxntid")
    return threads


def grid(n, threads):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1 or n > MAX_VECTORS:
        raise ValueError("vectors")
    if threads == NARROW:
        return n
    if threads not in WIDE:
        raise ValueError("block_threads")
    per = threads // 16
    return (n + per - 1) // per


def launch(driver, function, a, b, c, n, threads, stream=None):
    """16 is gpu_probe.Driver.launch unchanged; 32/64/128 is the r740 wide launch."""
    if threads == NARROW:
        return driver.launch(function, a, b, c, n)
    blocks = grid(n, threads)
    if stream is None:
        import torch
        stream = torch.cuda.current_stream().cuda_stream
    values = [ctypes.c_uint64(t.data_ptr()) for t in (a, b, c)] + [ctypes.c_uint32(n)]
    parameters = (ctypes.c_void_p * 4)(*[ctypes.addressof(v) for v in values])
    driver.check(driver.lib.cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0,
                                           ctypes.c_void_p(stream), parameters, None))
