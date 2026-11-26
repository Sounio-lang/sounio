"""Observation keys for one GPU. A hostname is not an identity when a node has two."""
import json
import math
from pathlib import Path
from paired_timing import VECTORS

ADMITTED_SM = (75, 86, 121)
# cudaDeviceProp.uuid is a 16-byte CUuuid. PyTorch 2.15 binds it as _CUuuid:
# `.bytes` is those raw bytes (a bytes-like or a list of ints) and `__str__`
# is uuid_to_string from torch/csrc/utils.cpp, the dashed hex in print order
# with no GPU- prefix. nvidia-smi -L is that same hex with the prefix.


def _grouped(hexchars):
    return "GPU-" + "-".join((
        hexchars[0:8], hexchars[8:12], hexchars[12:16], hexchars[16:20], hexchars[20:32],
    ))


def _canonical_text(value):
    text = value.strip()
    if not text:
        raise ValueError("GPU UUID is missing")
    body = text.lower()[4:] if text.lower().startswith("gpu-") else text.lower()
    if any(ch not in "0123456789abcdef-" for ch in body):
        raise ValueError("GPU UUID is malformed")
    hexchars = "".join(ch for ch in body if ch in "0123456789abcdef")
    if len(hexchars) != 32:
        raise ValueError("GPU UUID is malformed")
    if "-" in body and [len(group) for group in body.split("-")] != [8, 4, 4, 4, 12]:
        raise ValueError("GPU UUID is malformed")
    return _grouped(hexchars)


def _raw_uuid(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        blob = getattr(value, "bytes", None)
        if blob is None:
            return None
        if isinstance(blob, (bytes, bytearray, memoryview)):
            raw = bytes(blob)
        elif isinstance(blob, (list, tuple)):
            if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > 255 for item in blob):
                raise ValueError("GPU UUID is malformed")
            raw = bytes(blob)
        else:
            raise ValueError("GPU UUID is malformed")
    if len(raw) != 16:
        raise ValueError("GPU UUID is malformed")
    return raw


def canonical_gpu_uuid(value):
    """nvidia-smi -L form. str, 16 raw bytes, and a torch _CUuuid are accepted."""
    if value is None or value == "" or value == b"":
        raise ValueError("GPU UUID is missing")
    if isinstance(value, str):
        return _canonical_text(value)
    rendered = ""
    try:
        rendered = str(value).strip()
    except Exception:
        rendered = ""
    if rendered and not rendered.startswith("<"):
        try:
            text = rendered if rendered.lower().startswith("gpu-") else "GPU-" + rendered
            return _canonical_text(text)
        except ValueError:
            pass
    raw = _raw_uuid(value)
    if raw is None:
        raise ValueError("GPU UUID is malformed")
    return _grouped(raw.hex())


def normalize_uuid(uuid):
    return "".join(ch for ch in canonical_gpu_uuid(uuid) if ch in "0123456789abcdef")


def cuda_int(value, label):
    """A CUDA int, including a numpy integer. bool is not a capability or a size."""
    if isinstance(value, bool) or value is None:
        raise ValueError(label + " is missing")
    if isinstance(value, int):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception as exc:
            raise ValueError(label + " is malformed") from exc
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(label + " is malformed")
        return value
    raise ValueError(label + " is malformed")


def cuda_text(value, label):
    """A CUDA char buffer: str, or bytes with a trailing NUL."""
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).split(b"\0", 1)[0].decode("utf-8", "replace")
    if not isinstance(value, str):
        raise ValueError(label + " is missing")
    value = value.split("\0", 1)[0].strip()
    if not value:
        raise ValueError(label + " is missing")
    return value


def device_key(uuid, major, minor):
    if isinstance(major, bool) or isinstance(minor, bool) or not isinstance(major, int) or not isinstance(minor, int):
        raise ValueError("compute capability required")
    if major < 0 or minor < 0 or minor > 9 or major > 99:
        raise ValueError("compute capability required")
    sm = major * 10 + minor
    if sm not in ADMITTED_SM:
        raise ValueError("compute capability is outside the admitted pilot")
    return "gpu-" + normalize_uuid(uuid) + "-sm" + str(sm)


def observation_dir(phase, uuid, major, minor):
    if phase == "probe":
        prefix = "observed-"
    elif phase == "timing":
        prefix = "timings-"
    else:
        raise ValueError("phase")
    return prefix + device_key(uuid, major, minor)


def marker_name(phase, sm):
    if phase not in ("probe", "timing") or sm not in (75, 86):
        raise ValueError("marker")
    return "markers/" + phase + "-sm" + str(sm) + ".json"


def identity_record(uuid, major, minor, name):
    uuid = canonical_gpu_uuid(uuid)
    sm = major * 10 + minor
    if sm not in ADMITTED_SM:
        raise ValueError("compute capability is outside the admitted pilot")
    return {
        "gpu_uuid": uuid,
        "gpu_uuid_hex": normalize_uuid(uuid),
        "compute_capability": str(major) + "." + str(minor),
        "sm": sm,
        "gpu": name,
        "key": device_key(uuid, major, minor),
        "key_scope": "per-gpu",
    }


def device_identity(props, name, expect_sm):
    """Identity from a torch device-properties object. Fails closed on a bad field."""
    uuid = canonical_gpu_uuid(getattr(props, "uuid", None))
    major = cuda_int(getattr(props, "major", None), "compute capability major")
    minor = cuda_int(getattr(props, "minor", None), "compute capability minor")
    if name is None:
        name = getattr(props, "name", None)
    identity = identity_record(uuid, major, minor, cuda_text(name, "GPU name"))
    if identity["sm"] != expect_sm:
        raise RuntimeError("visible GPU SM does not match the gres this step requested")
    memory = getattr(props, "total_memory", None)
    if memory is not None:
        memory = cuda_int(memory, "GPU total memory")
        if memory < 0:
            raise ValueError("GPU total memory is malformed")
        identity["total_memory_bytes"] = memory
    return identity


def version_text(value):
    """A torch version field. Unknown objects become text instead of breaking JSON."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise ValueError("version is malformed")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).split(b"\0", 1)[0].decode("utf-8", "replace")
    item = getattr(value, "item", None)
    if callable(item):
        try:
            reduced = item()
        except Exception:
            reduced = value
        if reduced is not value:
            return version_text(reduced)
    return str(value)


def _json_ready(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("observation field is not JSON")
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).split(b"\0", 1)[0].decode("utf-8", "replace")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("observation field is not JSON")
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            reduced = item()
        except Exception as exc:
            raise ValueError("observation field is not JSON") from exc
        if reduced is value:
            raise ValueError("observation field is not JSON")
        return _json_ready(reduced)
    raise ValueError("observation field is not JSON")


def observation_document(identity, **fields):
    """Merge an identity into a record. A foreign key_scope is an error, not an overwrite."""
    if not isinstance(identity, dict):
        raise RuntimeError("observation identity is missing")
    scope = identity.get("key_scope")
    if scope != "per-gpu":
        raise RuntimeError("observation key_scope must be per-gpu, got " + repr(scope))
    clashes = [key for key in fields if key in identity and identity[key] != fields[key]]
    if clashes:
        raise RuntimeError("observation field clashes with identity: " + ", ".join(sorted(clashes)))
    merged = dict(identity)
    merged.update(fields)
    return _json_ready(merged)


def document_text(identity, **fields):
    return json.dumps(observation_document(identity, **fields), indent=2, allow_nan=False) + "\n"


def log_line(**fields):
    return json.dumps(_json_ready(fields), allow_nan=False, separators=(", ", ": "))


def probe_record_fields(job, node, cuda):
    return {"job": job, "node": node, "cuda": version_text(cuda), "observation_only": True}


def timing_record_fields(job, node, cuda, torch_version):
    return {
        "job": job,
        "node": node,
        "cuda": version_text(cuda),
        "torch": version_text(torch_version),
        "timing": "CUDA events around 32 captured kernel launches",
        "vectors": VECTORS,
        "layout_conversion_included": False,
        "observation_only": True,
    }


def _write_document(path, identity, **fields):
    document = observation_document(identity, **fields)
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def write_inventory(directory, identity, **fields):
    return _write_document(Path(directory) / "inventory.json", identity, **fields)


def write_marker(root, phase, identity, job):
    document = observation_document(identity, phase=phase, job=job)
    path = Path(root) / marker_name(phase, document["sm"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return path
