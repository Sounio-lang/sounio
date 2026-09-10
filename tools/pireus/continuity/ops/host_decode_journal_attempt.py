"""Prospective journal protocol with composed source, runtime and journal custody."""
import hashlib
import importlib.util
import json
from pathlib import Path
import host_decode_journal_custody as journals

HERE=Path(__file__).resolve().parents[1]
IMPLEMENTATION_SHA="1f78b7bfab99af604a1da3137c5d320b3f700661605117984aeec72e12106a65"
PROTOCOL_SHA="aae02919fa352416a285904ab8d3701c9d726bad8a7cf952b3dad6216d7f5686"
PROTOCOL=HERE/"validation/external-overhead-v3-preparation-20260910/host-decode-probe/journal-v1/protocol/diagnostic-protocol.json"
def context():
    path=HERE/"ops/host_decode_attempt.py"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=IMPLEMENTATION_SHA:
        raise ValueError("parent execution implementation changed")
    spec=importlib.util.spec_from_file_location("pireus_host_journal_context",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.PROTOCOL=PROTOCOL;module.PROTOCOL_SHA=PROTOCOL_SHA
    module.HELPERS=module.HELPERS+(
        "ops/host_decode_journal_attempt.py","ops/host_decode_journal_custody.py",
        "ops/host_decode_journal.py",
        "validation/external-overhead-v3-preparation-20260910/host-decode-probe/runtime-binding-audit/audit.py")
    p=module.protocol()
    module.require(p["runtime_sha256"]["offline_generate.py"]==journals.RUNTIME_SHA,"journal runtime pin")
    return module

def collect(frozen,stage,output,job,run=None):
    attempt=context()
    attempt.verify(frozen)
    output.mkdir(exist_ok=False)
    args={} if run is None else {"run":run}
    base=attempt.collect(frozen,stage,output/"base","baseline",job,**args)
    base_pin=journals.sha((output/"base/collection.json").read_bytes())
    journals.collect(output/"base",base_pin,output/"journals",**args)
    journal_pin=journals.sha((output/"journals/collection.json").read_bytes())
    result=dict(schema="pireus-host-journal-composed-custody-v1",job=job,
                source_commit=base["source_commit"],freeze_sha256=base["freeze_sha256"],
                base_collection_sha256=base_pin,journal_collection_sha256=journal_pin,
                diagnostic_qualified=False)
    (output/"collection.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

def inspect(frozen,root,pin):
    attempt=context();spec=attempt.verify(frozen)
    c=json.loads(journals.pinned(root,"collection.json",pin))
    journals.require(c["schema"]=="pireus-host-journal-composed-custody-v1"
        and c["source_commit"]==spec["source_commit"]
        and c["freeze_sha256"]==journals.sha((frozen/"execution-freeze.json").read_bytes()),"composed source/freeze")
    base=attempt.inspect_collection(frozen,root/"base",c["base_collection_sha256"])
    result=journals.inspect(root/"base",c["base_collection_sha256"],root/"journals",c["journal_collection_sha256"])
    journals.require(c["job"]==base["job"]==result["job"],"composed job")
    # Both checks execute, including complete source check receipts, worker/boot,
    # runtime/input bytes and lifecycle-bound raw journal metrics.
    return dict(schema="pireus-host-journal-composed-inspection-v1",job=c["job"],
        collection_sha256=pin,source_commit=spec["source_commit"],
        source_and_base_custody_verified=True,base_inspection=base,journal_inspection=result,
        complete_probe_windows=result["complete_probe_windows"],
        loaded_model_qualified=False,timing_eligible=False,pilot_acceptance=False,
        causal_memory_explanation_established=False)
