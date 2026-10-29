#!/usr/bin/env python3
"""Keep a prefix of routed experts anonymous and stage the rest from a file.

On a unified-memory host, every anonymous CUDA or pinned allocation reduces
MemAvailable. Inkling's routed experts are the bulk of the parameter
storage. The first layers stay in ordinary CUDA memory. The rest stay in a
sparse file and move through a pinned double-buffer of at most two layers:
the copy of layer i+1 is queued on its own stream while layer i runs.
CUDA events order the copy behind the previous user of that slot and the
compute behind its own copy.

posix_fadvise(DONTNEED) is not used. On the pages that stay mapped it does
not drop the page cache, so it does not change the copy.

Inference does not rewrite weights. Releasing a staged layer points the
parameter back at the file image and does not copy device bytes over it.
"""
import contextlib
import ctypes
import gc
import mmap
import os
from pathlib import Path
import shutil
import threading

GIB = 1024**3
# Steady anonymous cap for the resident prefix plus both slabs of the
# double-buffer (pinned and device). Pilot-20261024f reported 69.5e9 of
# headroom above the 33 GiB reserve at the tightest load point. Keeping
# 10 GiB above that reserve leaves 58.76e9. 50e9 sits inside that ceiling.
# Override with PIREUS_LEAN_RESIDENT_BUDGET_BYTES. This is not the guard.
DEFAULT_BUDGET_BYTES = 50_000_000_000
MARGIN_BYTES = 10 * GIB
STAGING_SLOTS = 2
STAGING_COPIES = 2
STAGING_ALIGN = 256

_BACKINGS = {}
_LOGICAL_RESERVED = 0
_IMAGE_SEQ = 0
_INSTALLED = False
_PLAN_READY = False
_PLAN = None
_RESIDENT_IDS = set()
_MMAP_LAYERS = []
_MMAP_INDEX = {}
_STARTED = set()
_STAGE = None
_FIRST_LAYER_ID = None


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


def _layer_sort_key(name):
    parts = name.split(".")
    if "layers" in parts:
        index = parts.index("layers")
        if index + 1 < len(parts) and parts[index + 1].isdigit():
            return (0, int(parts[index + 1]), name)
    return (1, 0, name)


def budget_bytes():
    raw = os.environ.get("PIREUS_LEAN_RESIDENT_BUDGET_BYTES")
    if raw is None or raw == "":
        return DEFAULT_BUDGET_BYTES
    try:
        value = int(raw, 10)
    except ValueError:
        raise ValueError("PIREUS_LEAN_RESIDENT_BUDGET_BYTES must be an integer") from None
    if value < 0:
        raise ValueError("PIREUS_LEAN_RESIDENT_BUDGET_BYTES must be non-negative")
    return value


def headroom_keeps_margin(headroom_bytes, accounted_bytes, margin_bytes=MARGIN_BYTES):
    for value in (headroom_bytes, accounted_bytes, margin_bytes):
        if type(value) is not int or value < 0:
            raise ValueError("headroom accounting requires non-negative ints")
    return headroom_bytes - accounted_bytes >= margin_bytes


def _packed(sizes, align):
    offset = 0
    for size in sizes:
        if type(size) is not int or size < 0:
            raise ValueError("parameter byte count must be a non-negative int")
        offset += size
        if align:
            offset = (offset + align - 1) // align * align
    return offset


def _expert_layers(parameters, align):
    groups = {}
    for item in parameters:
        size = item["bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("parameter byte count must be a non-negative int")
        if size == 0 or not is_routed_expert_parameter(item["name"]):
            continue
        key = layer_key(item["name"])
        groups.setdefault(key, []).append((item["name"], size))
    layers = []
    for key in sorted(groups, key=_layer_sort_key):
        items = sorted(groups[key])
        raw = sum(size for _, size in items)
        layers.append((key, raw, _packed([size for _, size in items], align)))
    return layers


def select_resident_plan(parameters, budget=None, slots=STAGING_SLOTS,
                         copies=STAGING_COPIES, align=STAGING_ALIGN):
    """Longest prefix of routed-expert layers that stays inside ``budget``.

    Layers are ordered by decoder index. The prefix stays anonymous. The
    remainder stays file-backed. ``copies`` slabs (pinned and device) are
    reserved for ``slots`` of those file-backed layers, sized to the largest
    packed remainder. The largest fitting prefix wins.
    """
    if budget is None:
        budget = budget_bytes()
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a non-negative int")
    if type(slots) is not int or not 1 <= slots <= STAGING_SLOTS:
        raise ValueError("staging keeps at most 2 layers")
    if type(copies) is not int or copies < 1:
        raise ValueError("staging copies must be a positive int")
    layers = _expert_layers(parameters, align)
    feasible = []
    for count in range(len(layers) + 1):
        resident = layers[:count]
        mmap = layers[count:]
        resident_bytes = sum(raw for _, raw, _packed_size in resident)
        parked_bytes = sum(raw for _, raw, _packed_size in mmap)
        if mmap:
            used_slots = min(slots, len(mmap))
            capacity = max(packed for _, _raw, packed in mmap)
        else:
            used_slots = 0
            capacity = 0
        staging_bytes = used_slots * capacity * copies
        accounted = resident_bytes + staging_bytes
        if accounted <= budget:
            feasible.append(dict(
                method="resident-prefix-plus-pinned-double-buffer",
                budget_bytes=budget,
                resident_layer_count=count,
                resident_layers=[key for key, _, _ in resident],
                resident_bytes=resident_bytes,
                parked_bytes=parked_bytes,
                mmap_layers=[key for key, _, _ in mmap],
                mmap_layer_count=len(mmap),
                staging_slots=used_slots,
                staging_capacity_bytes=capacity,
                staging_copies=copies,
                staging_bytes=staging_bytes,
                accounted_bytes=accounted,
            ))
    if not feasible:
        raise ValueError("no routed-expert residency fits the anonymous budget")
    return max(feasible, key=lambda plan: plan["resident_layer_count"])


def prefetch_trace(mmap_layers, slots=STAGING_SLOTS):
    """Submission order for one forward of the file-backed layers.

    Slot ``index % slots`` holds layer ``index``. The copy of layer i+1 is
    submitted after that layer's own copy has finished and before compute of
    layer i finishes. A slot is copied again only after the compute that used
    the previous occupant has finished.
    """
    if type(slots) is not int or not 1 <= slots <= STAGING_SLOTS:
        raise ValueError("double-buffer keeps at most 2 layers")
    events = []
    count = len(mmap_layers)
    if count == 0:
        return events
    used = min(slots, count)
    started = set()

    def start(index):
        events.append(("copy_begin", mmap_layers[index], index % used))
        events.append(("copy_end", mmap_layers[index], index % used))
        started.add(index)

    start(0)
    for index, layer in enumerate(mmap_layers):
        if index not in started:
            start(index)
        events.append(("compute_begin", layer, index % used))
        if index + 1 < count and (index + 1) not in started:
            start(index + 1)
        events.append(("compute_end", layer, index % used))
    return events


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


def _sync_image(backing):
    # The image was just written. fdatasync makes that write durable.
    # DONTNEED is omitted: the mapping stays alive, and advising it away
    # does not drop those pages.
    os.fdatasync(backing["file"].fileno())


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
        _sync_image(backing)
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
        _sync_image(replacement)
        return
    _copy_bytes(backing["tensor"], contig)
    param.data = backing["tensor"]
    _sync_image(backing)


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
    seen = set()
    for module in layer.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        for local_name, param in module.named_parameters(recurse=False):
            backing = _BACKINGS.get((id(module), local_name))
            if backing is not None:
                yield module, local_name, param, backing


def materialize_layer(layer):
    # Load-time path, before the residency plan exists. Generate does not use it.
    for _module, _local_name, param, backing in _each_parked(layer):
        if param.data.data_ptr() == backing["tensor"].data_ptr():
            cuda = backing["tensor"].detach().to(device="cuda", non_blocking=False)
            if tuple(cuda.shape) != tuple(backing["tensor"].shape) or cuda.dtype != backing["tensor"].dtype:
                raise RuntimeError("materialized expert tensor changed shape")
            param.data = cuda


def release_layer(layer):
    for _module, _name, param, backing in _each_parked(layer):
        # The kernel has been queued on this stream. Dropping the CUDA reference
        # returns the block to the caching allocator, which will not reuse it
        # until that stream is done. Do not empty the cache here: the next
        # layer should reuse the largest block instead of allocating another.
        param.data = backing["tensor"]


def _decoder_layers(model):
    found = []
    for module in model.modules():
        layer_id = getattr(module, "layer_id", None)
        if type(layer_id) is not int:
            continue
        if getattr(module, "_pireus_lean_prefix", None) != "layer" + str(layer_id) + ".":
            continue
        found.append(module)
    found.sort(key=lambda module: module.layer_id)
    return found


def _layer_items(layer):
    items = []
    seen = set()
    for module in layer.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        parts = getattr(module, "_pireus_lean_prefix", "").split(".")
        if "shared_experts" in parts or "experts" not in parts:
            continue
        for local_name, param in module.named_parameters(recurse=False):
            if not is_routed_expert_parameter(local_name) or param.numel() == 0:
                continue
            items.append((module, local_name, param, param.numel() * param.element_size()))
    items.sort(key=lambda item: item[1])
    return items


def _retain_parameter(module, local_name, param):
    torch = _torch()
    backing = _BACKINGS.get((id(module), local_name))
    if backing is None:
        if param.device.type != "cuda":
            param.data = param.detach().to(device="cuda", non_blocking=False)
        return
    cuda = backing["tensor"].detach().to(device="cuda", non_blocking=False)
    if tuple(cuda.shape) != tuple(backing["tensor"].shape) or cuda.dtype != backing["tensor"].dtype:
        raise RuntimeError("retained expert changed shape")
    param.data = cuda
    _BACKINGS.pop((id(module), local_name), None)
    files = getattr(module, "_pireus_lean_files", None)
    if files is not None and backing in files:
        files.remove(backing)
    _close_image(backing)


def _allocate_stage(capacity, slots):
    torch = _torch()
    if slots == 0 or capacity == 0:
        return None
    buffers = []
    for _ in range(slots):
        buffers.append(dict(
            pinned=torch.empty(capacity, dtype=torch.uint8, pin_memory=True),
            device=torch.empty(capacity, dtype=torch.uint8, device="cuda"),
            copy_done=torch.cuda.Event(),
            compute_done=torch.cuda.Event(),
            thread=None,
            pieces=None,
            error=None,
            used=False,
        ))
    return dict(capacity=capacity, slot_count=slots, slots=buffers, stream=torch.cuda.Stream())


def _start_copy(layer, slot):
    stage = _STAGE
    buf = stage["slots"][slot]
    if buf["thread"] is not None and buf["thread"].is_alive():
        raise RuntimeError("staging slot is still copying")
    pieces = []
    offset = 0
    parked = sorted(_each_parked(layer), key=lambda item: item[1])
    for _module, _local_name, param, backing in parked:
        src = backing["tensor"]
        if not src.is_contiguous():
            raise RuntimeError("mmap expert is not contiguous")
        nbytes = src.numel() * src.element_size()
        if offset + nbytes > stage["capacity"]:
            raise RuntimeError("staged layer exceeds the pinned buffer")
        pieces.append(dict(param=param, src=src, offset=offset, nbytes=nbytes,
                           shape=tuple(src.shape), dtype=src.dtype, backing=backing))
        offset += nbytes
        offset = (offset + STAGING_ALIGN - 1) // STAGING_ALIGN * STAGING_ALIGN
    if offset > stage["capacity"]:
        raise RuntimeError("aligned layer exceeds the pinned buffer")
    buf["pieces"] = pieces
    buf["error"] = None

    def work(buf=buf, pieces=pieces):
        try:
            torch = _torch()
            stream = stage["stream"]
            with torch.cuda.stream(stream):
                if buf["used"]:
                    stream.wait_event(buf["compute_done"])
                for piece in pieces:
                    raw = piece["src"].detach().view(torch.uint8).reshape(-1)
                    pinned = buf["pinned"].narrow(0, piece["offset"], piece["nbytes"])
                    pinned.copy_(raw)
                    device = buf["device"].narrow(0, piece["offset"], piece["nbytes"])
                    device.copy_(pinned, non_blocking=True)
                buf["copy_done"].record(stream)
            buf["used"] = True
        except BaseException as exc:
            buf["error"] = exc

    buf["thread"] = threading.Thread(target=work, name="pireus-lean-copy", daemon=True)
    buf["thread"].start()


def _wait_slot(slot):
    buf = _STAGE["slots"][slot]
    thread = buf["thread"]
    if thread is not None:
        thread.join()
    if buf["error"] is not None:
        raise RuntimeError("expert staging copy failed") from buf["error"]
    _torch().cuda.current_stream().wait_event(buf["copy_done"])


def _bind(slot):
    buf = _STAGE["slots"][slot]
    for piece in buf["pieces"]:
        view = buf["device"].narrow(0, piece["offset"], piece["nbytes"])
        view = view.view(piece["dtype"]).reshape(piece["shape"])
        piece["param"].data = view


def _unbind(layer):
    for _module, _name, param, backing in _each_parked(layer):
        param.data = backing["tensor"]


def _ensure_started(index):
    if index in _STARTED:
        return
    slots = _STAGE["slot_count"]
    _start_copy(_MMAP_LAYERS[index], index % slots)
    _STARTED.add(index)


def _prime():
    if _MMAP_LAYERS:
        _ensure_started(0)


def _staged_forward(layer, original, args, kwargs):
    index = _MMAP_INDEX[id(layer)]
    slot = index % _STAGE["slot_count"]
    _ensure_started(index)
    _wait_slot(slot)
    _bind(slot)
    if index + 1 < len(_MMAP_LAYERS):
        _ensure_started(index + 1)
    try:
        return original(layer, *args, **kwargs)
    finally:
        _STAGE["slots"][slot]["compute_done"].record(_torch().cuda.current_stream())
        _unbind(layer)


def commit_residency(model, budget=None):
    """Keep the budget's prefix on CUDA and arm the double-buffer for the rest.

    Parked bytes must then equal the plan. That equality is the strict check:
    a layer the budget left on the file cannot remain anonymous, and a layer
    the budget kept cannot remain on the file.
    """
    global _PLAN_READY, _PLAN, _RESIDENT_IDS, _MMAP_LAYERS, _MMAP_INDEX, _STAGE, _STARTED
    global _FIRST_LAYER_ID
    if _PLAN_READY:
        return dict(status(model), **_PLAN)
    layers = _decoder_layers(model)
    if layers:
        _FIRST_LAYER_ID = id(layers[0])
    parameters = []
    by_key = {}
    for layer in layers:
        key = "llm.layers." + str(layer.layer_id)
        items = _layer_items(layer)
        if not items:
            continue
        by_key[key] = (layer, items)
        for _module, local_name, _param, nbytes in items:
            parameters.append(dict(name=key + ".mlp.experts." + local_name, bytes=nbytes))
    plan = select_resident_plan(parameters, budget)
    for key in plan["resident_layers"]:
        layer, items = by_key[key]
        for module, local_name, param, _nbytes in items:
            _retain_parameter(module, local_name, param)
        _RESIDENT_IDS.add(id(layer))
        _release_allocator()
    _MMAP_LAYERS = []
    _MMAP_INDEX = {}
    for key in plan["mmap_layers"]:
        layer, _items = by_key[key]
        for module, local_name, param, _nbytes in _items:
            backing = _BACKINGS.get((id(module), local_name))
            if backing is None or param.data.data_ptr() != backing["tensor"].data_ptr():
                raise RuntimeError("file-backed expert is not on its image: " + key)
        _MMAP_INDEX[id(layer)] = len(_MMAP_LAYERS)
        _MMAP_LAYERS.append(layer)
    _release_allocator()
    _STAGE = _allocate_stage(plan["staging_capacity_bytes"], plan["staging_slots"])
    _STARTED = set()
    _PLAN = plan
    _PLAN_READY = True
    actual = status(model)
    if actual["parked_bytes"] != plan["parked_bytes"]:
        raise RuntimeError("parked expert bytes differ from the budget plan")
    if actual["anonymous_expert_bytes"] != plan["resident_bytes"]:
        raise RuntimeError("resident expert bytes differ from the budget plan")
    if plan["accounted_bytes"] > plan["budget_bytes"]:
        raise RuntimeError("routed-expert plan exceeds the anonymous budget")
    if actual["parked_bytes"] + actual["anonymous_expert_bytes"] != plan["resident_bytes"] + plan["parked_bytes"]:
        raise RuntimeError("routed-expert bytes are neither resident nor parked")
    merged = dict(actual)
    merged.update(plan)
    merged["placement"] = "resident-prefix-plus-file"
    return merged


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
        global _STARTED
        depth = getattr(self, "_pireus_lean_depth", 0)
        self._pireus_lean_depth = depth + 1
        try:
            if depth != 0:
                return original_forward(self, *args, **kwargs)
            if _PLAN_READY and id(self) == _FIRST_LAYER_ID:
                # A new token. The previous token recorded compute_done on every
                # slot it used, so the next copy waits for that kernel. The first
                # file-backed layer starts copying while this layer runs.
                _STARTED = set()
                _prime()
            if not _PLAN_READY:
                materialize_layer(self)
                try:
                    return original_forward(self, *args, **kwargs)
                finally:
                    release_layer(self)
            if id(self) in _RESIDENT_IDS:
                _prime()
                return original_forward(self, *args, **kwargs)
            if id(self) not in _MMAP_INDEX:
                return original_forward(self, *args, **kwargs)
            return _staged_forward(self, original_forward, args, kwargs)
        finally:
            self._pireus_lean_depth = depth

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
