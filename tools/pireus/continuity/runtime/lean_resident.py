#!/usr/bin/env python3
"""Keep routed-expert shards in a sparse file and stage one layer at a time.

On a unified-memory host, every anonymous CUDA allocation reduces
MemAvailable. Inkling's routed experts are the bulk of the parameter
storage. They do not all need to be anonymous at once: a forward reads
one decoder layer, then the next. The file pages stay reclaimable. The
CUDA caching allocator is allowed to retain the largest layer that has
already run; it is not allowed to retain every layer.

Inference does not rewrite weights. Releasing a layer points the
parameter back at the file image and does not copy device bytes over it.
"""
import contextlib
import ctypes
import gc
import mmap
import os
from pathlib import Path
import shutil

GIB = 1024**3
# Fail closed if the hook missed the expert stack. This is not the host reserve.
MINIMUM_PARKED_BYTES = 6 * GIB

_BACKINGS = {}
_LOGICAL_RESERVED = 0
_IMAGE_SEQ = 0
_INSTALLED = False


def is_routed_expert_parameter(name):
    """True for routed w13/w2 weights and their scales. Shared experts stay put.

    Callers pass the module path. A bare leaf is accepted only for the routed
    expert module itself; a ``shared_experts`` path is never accepted.
    """
    parts = name.split(".")
    if "shared_experts" in parts:
        return False
    leaf = parts[-1]
    if not (leaf.startswith("w13_weight") or leaf.startswith("w2_weight")):
        return False
    if len(parts) == 1:
        return True
    return "experts" in parts


def layer_key(name):
    parts = name.split(".")
    if "layers" not in parts:
        return name
    index = parts.index("layers")
    if index + 1 >= len(parts) or not parts[index + 1].isdigit():
        return name
    return "llm.layers." + parts[index + 1]


def estimate_anonymous_peak(parameters):
    """Anonymous peak if only the active routed-expert layer is materialized.

    ``parameters`` is a sequence of ``{"name", "bytes"}``. File-backed bytes
    are not anonymous. The CUDA pool retains the largest layer, so the peak
    is the resident remainder plus that layer, not the sum of the layers.
    """
    total = 0
    expert = 0
    layers = {}
    for item in parameters:
        size = item["bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("parameter byte count must be a non-negative int")
        total += size
        if not is_routed_expert_parameter(item["name"]):
            continue
        expert += size
        key = layer_key(item["name"])
        layers[key] = layers.get(key, 0) + size
    max_layer = max(layers.values()) if layers else 0
    after = total - expert + max_layer
    return dict(
        method="file-backed-routed-experts-active-layer-only",
        parameter_bytes=total,
        expert_bytes=expert,
        layer_count=len(layers),
        max_active_layer_bytes=max_layer,
        resident_non_expert_bytes=total - expert,
        anonymous_before_bytes=total,
        anonymous_after_peak_bytes=after,
        saved_bytes=total - after,
    )


class ByteResident:
    """Sparse-file stand-in used to test the park/materialize/release protocol."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.anonymous = 0
        self.peak = 0
        self.slots = {}

    def close(self):
        for slot in self.slots.values():
            slot["file"].close()
        self.slots.clear()

    def park_fresh(self, name, nbytes):
        if nbytes < 0 or name in self.slots:
            raise ValueError("invalid fresh park")
        path = self.directory / (name + ".bin")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        file = os.fdopen(fd, "r+b")
        os.ftruncate(file.fileno(), nbytes)
        copied = 0
        self.slots[name] = dict(path=path, file=file, nbytes=nbytes, hot=None)
        return copied

    def write(self, name, payload):
        slot = self.slots[name]
        if len(payload) != slot["nbytes"]:
            raise ValueError("payload does not match the parked image")
        slot["file"].seek(0)
        slot["file"].write(payload)
        slot["file"].flush()
        os.fdatasync(slot["file"].fileno())

    def materialize(self, name):
        slot = self.slots[name]
        if slot["hot"] is not None:
            raise ValueError("layer is already materialized")
        slot["file"].seek(0)
        payload = slot["file"].read()
        if len(payload) != slot["nbytes"]:
            raise ValueError("short file image")
        os.posix_fadvise(slot["file"].fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        slot["hot"] = payload
        self.anonymous += len(payload)
        self.peak = max(self.peak, self.anonymous)
        return payload

    def release(self, name):
        slot = self.slots[name]
        if slot["hot"] is None:
            raise ValueError("layer is not materialized")
        self.anonymous -= len(slot["hot"])
        slot["hot"] = None


def _torch():
    import torch
    return torch


def _directory():
    path = Path(os.environ.get("PIREUS_LEAN_RESIDENT_DIR", "/scratch/pireus/cache/lean-resident"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _release_allocator():
    torch = _torch()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int
    libc.malloc_trim(0)


def _drop_file_cache(backing, sync):
    if sync:
        os.fdatasync(backing["file"].fileno())
    os.posix_fadvise(backing["file"].fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def _storage_is_mapping(tensor, mapping):
    if tensor.numel() == 0:
        return True
    base = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    return tensor.data_ptr() == base


def _tensor_on_mapping(mapping, dtype, shape, nbytes):
    torch = _torch()
    raw = torch.frombuffer(mapping, dtype=torch.uint8)
    if raw.numel() != nbytes:
        raise RuntimeError("file image does not match the tensor size")
    tensor = raw.view(dtype).reshape(shape)
    if not _storage_is_mapping(tensor, mapping):
        raise RuntimeError("file backing was copied into anonymous memory")
    return tensor


def _open_image(name, dtype, shape, nbytes):
    global _LOGICAL_RESERVED
    if nbytes <= 0:
        raise ValueError("refusing an empty expert image")
    directory = _directory()
    _LOGICAL_RESERVED += nbytes
    if shutil.disk_usage(directory).free < _LOGICAL_RESERVED:
        raise ValueError("insufficient local storage for file-backed routed experts")
    global _IMAGE_SEQ
    _IMAGE_SEQ += 1
    job = os.environ["SLURM_JOB_ID"]
    rank = os.environ["PIREUS_RANK"]
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:80]
    path = directory / ("lean-" + job + "-" + rank + "-" + str(_IMAGE_SEQ) + "-" + safe + ".bin")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    file = os.fdopen(fd, "r+b")
    os.ftruncate(file.fileno(), nbytes)
    mapping = mmap.mmap(file.fileno(), nbytes, access=mmap.ACCESS_WRITE)
    tensor = _tensor_on_mapping(mapping, dtype, shape, nbytes)
    return dict(path=path, file=file, mapping=mapping, tensor=tensor, nbytes=nbytes)


def _close_image(backing):
    global _LOGICAL_RESERVED
    _LOGICAL_RESERVED -= backing["nbytes"]
    backing.pop("tensor", None)
    gc.collect()
    backing["mapping"].close()
    backing["file"].close()
    backing["path"].unlink(missing_ok=True)


def _copy_bytes(destination, source):
    torch = _torch()
    src = source.detach().contiguous().view(torch.uint8).reshape(-1)
    dst = destination.view(torch.uint8).reshape(-1)
    if src.numel() != dst.numel():
        raise RuntimeError("expert image size changed during the copy")
    dst.copy_(src)


def _park(owner, local_name, param):
    torch = _torch()
    if param.device.type == "meta":
        raise RuntimeError("refusing to park a meta parameter")
    if not param.is_contiguous() or param.numel() == 0:
        return
    nbytes = param.numel() * param.element_size()
    before = torch.cuda.memory_allocated() if param.is_cuda else None
    full = local_name if owner is None else getattr(owner, "_pireus_lean_prefix", local_name)
    backing = _open_image(full + "." + local_name, param.dtype, tuple(param.shape), nbytes)
    old = param.data
    param.data = backing["tensor"]
    del old
    _BACKINGS[(id(owner), local_name)] = backing
    files = getattr(owner, "_pireus_lean_files", None)
    if files is None:
        files = []
        owner._pireus_lean_files = files
    files.append(backing)
    if before is not None:
        gc.collect()
        after = torch.cuda.memory_allocated()
        if before - after < nbytes:
            raise RuntimeError("routed expert CUDA storage was retained: " + local_name)


def park_new_layer(layer):
    layer_id = str(getattr(layer, "layer_id", "na"))
    for mod_name, module in layer.named_modules():
        module._pireus_lean_prefix = "layer" + layer_id + "." + mod_name
        for local_name, param in module.named_parameters(recurse=False):
            full = local_name if not mod_name else mod_name + "." + local_name
            if is_routed_expert_parameter(full) and (id(module), local_name) not in _BACKINGS:
                _park(module, local_name, param)
    _release_allocator()


def _repark_one(owner, local_name, param, backing):
    current = param.data
    same = (current.data_ptr() == backing["tensor"].data_ptr()
            and tuple(current.shape) == tuple(backing["tensor"].shape)
            and current.dtype == backing["tensor"].dtype)
    if same:
        _drop_file_cache(backing, sync=True)
        return
    contig = current.detach().contiguous()
    nbytes = contig.numel() * contig.element_size()
    if (backing["nbytes"] != nbytes or tuple(backing["tensor"].shape) != tuple(contig.shape)
            or backing["tensor"].dtype != contig.dtype):
        replacement = _open_image(backing["path"].name + ".repack", contig.dtype,
                                  tuple(contig.shape), nbytes)
        _copy_bytes(replacement["tensor"], contig)
        param.data = replacement["tensor"]
        _BACKINGS[(id(owner), local_name)] = replacement
        files = owner._pireus_lean_files
        if backing in files:
            files.remove(backing)
        files.append(replacement)
        _close_image(backing)
        _drop_file_cache(replacement, sync=True)
        return
    _copy_bytes(backing["tensor"], contig)
    param.data = backing["tensor"]
    _drop_file_cache(backing, sync=True)


def repark_module(module):
    owners = []
    seen = set()
    for _, owner in module.named_modules():
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        owners.append(owner)
    touched = False
    for owner in owners:
        for local_name, param in owner.named_parameters(recurse=False):
            backing = _BACKINGS.get((id(owner), local_name))
            if backing is None:
                continue
            _repark_one(owner, local_name, param, backing)
            touched = True
    if touched:
        _release_allocator()


def _each_parked(layer):
    for module in [layer, *[child for _, child in layer.named_modules()]]:
        for local_name, param in module.named_parameters(recurse=False):
            backing = _BACKINGS.get((id(module), local_name))
            if backing is not None:
                yield module, local_name, param, backing


def materialize_layer(layer):
    for module, local_name, param, backing in _each_parked(layer):
        if param.data.data_ptr() == backing["tensor"].data_ptr():
            cuda = backing["tensor"].detach().to(device="cuda", non_blocking=False)
            if tuple(cuda.shape) != tuple(backing["tensor"].shape) or cuda.dtype != backing["tensor"].dtype:
                raise RuntimeError("materialized expert tensor changed shape")
            param.data = cuda
        # The image was synced when it was parked. This read must not fsync.
        _drop_file_cache(backing, sync=False)


def release_layer(layer):
    for _module, _name, param, backing in _each_parked(layer):
        # The kernel has been queued on this stream. Dropping the CUDA reference
        # returns the block to the caching allocator, which will not reuse it
        # until that stream is done. Do not empty the cache here: the next
        # layer should reuse the largest block instead of allocating another.
        param.data = backing["tensor"]


def status(model):
    parked = 0
    anonymous = 0
    count = 0
    for mod_name, module in model.named_modules():
        for local_name, param in module.named_parameters(recurse=False):
            full = local_name if not mod_name else mod_name + "." + local_name
            if not is_routed_expert_parameter(full):
                continue
            count += 1
            nbytes = param.numel() * param.element_size()
            backing = _BACKINGS.get((id(module), local_name))
            if backing is not None and param.data.data_ptr() == backing["tensor"].data_ptr():
                parked += nbytes
            else:
                anonymous += nbytes
    return dict(
        placement="file-backed-active-layer",
        method="file-backed-routed-experts-active-layer-only",
        routed_expert_parameters=count,
        parked_bytes=parked,
        anonymous_expert_bytes=anonymous,
        logical_reserved_bytes=_LOGICAL_RESERVED,
    )


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import sglang.srt.model_loader.loader as loader
    from sglang.srt.models.inkling import InklingDecoderLayer

    original_init = InklingDecoderLayer.__init__
    original_forward = InklingDecoderLayer.forward

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        park_new_layer(self)

    def forward(self, *args, **kwargs):
        depth = getattr(self, "_pireus_lean_depth", 0)
        self._pireus_lean_depth = depth + 1
        if depth == 0:
            materialize_layer(self)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            self._pireus_lean_depth = depth
            if depth == 0:
                release_layer(self)

    InklingDecoderLayer.__init__ = init
    InklingDecoderLayer.forward = forward
    original_context = loader.device_loading_context

    @contextlib.contextmanager
    def context(module, target_device):
        with original_context(module, target_device):
            yield module
        repark_module(module)

    loader.device_loading_context = context
    loader._pireus_lean = True
    _INSTALLED = True
