"""Observation keys for one GPU. A hostname is not an identity when a node has two."""

ADMITTED_SM = (75, 86, 121)


def normalize_uuid(uuid):
    if not isinstance(uuid, str) or not uuid:
        raise ValueError("gpu uuid required")
    hexchars = "".join(ch for ch in uuid.lower() if ch in "0123456789abcdef")
    if len(hexchars) != 32:
        raise ValueError("gpu uuid must contain 32 hex digits")
    return hexchars


def device_key(uuid, major, minor):
    if not isinstance(major, int) or not isinstance(minor, int):
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
