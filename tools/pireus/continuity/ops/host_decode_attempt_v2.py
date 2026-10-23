"""Prospective source binding; reuse the pinned v1 execution implementation."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
IMPLEMENTATION_SHA='1f78b7bfab99af604a1da3137c5d320b3f700661605117984aeec72e12106a65'
PROTOCOL_SHA='7eda91b30c938a4721ced565388806fdd7c5e8e0844f53f39d1e84b6c64cb267'
def context():
    path=HERE/'ops/host_decode_attempt.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=IMPLEMENTATION_SHA:
        raise ValueError('parent execution implementation changed')
    spec=importlib.util.spec_from_file_location('pireus_host_decode_source_v2_context',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL=HERE/'validation/external-overhead-v3-preparation-20260910/host-decode-probe/source-v2/diagnostic-protocol.json'
    module.PROTOCOL_SHA=PROTOCOL_SHA
    module.HELPERS=module.HELPERS+('ops/host_decode_attempt_v2.py',)
    module.protocol()
    return module
