#!/usr/bin/env python3
"""Exercise actual Sounio admission executable with raw adversarial JSON bytes."""
import argparse, hashlib, json, pathlib, subprocess, tempfile
def raw(value): return (json.dumps(value,separators=(",",":"))+"\n").encode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("engine");args=ap.parse_args()
 context=dict(schema=1,target=701202,dimension=16,precision=64,order=1,fma=0,capabilities=55,lane_width=32,facts_state=1)
 proposal=dict(schema=1,target=701202,dimension=16,precision=64,order=1,fma=0,kind=1,lane_stride=1,lane_offset=0,load=1,layout=0,unroll=1)
 with tempfile.TemporaryDirectory() as directory:
  root=pathlib.Path(directory)
  def check(label,reason=None,changes=None,context_changes=None,transform=None,mode=None):
   c=context| (context_changes or {});cb=raw(c)
   p=proposal|{"context":hashlib.sha256(cb).hexdigest()}| (changes or {})
   pb=raw(p)
   if transform:pb=transform(pb)
   (root/"context.json").write_bytes(cb);(root/"proposal.json").write_bytes(pb)
   argv=[args.engine,str(root/"context.json"),str(root/"proposal.json")]+([mode] if mode else [])
   run=subprocess.run(argv,capture_output=True,text=True,timeout=30)
   try: receipt=json.loads(run.stdout)
   except Exception:raise AssertionError((label,run.returncode,run.stdout,run.stderr))
   if reason:
    assert run.returncode==1 and receipt["decision"]=="REFUSE" and receipt["reason"]==reason,(label,receipt)
   else:
    assert run.returncode==0 and receipt["decision"]=="ADMIT",(label,receipt)
    assert receipt["proposal_sha256"]==hashlib.sha256(pb).hexdigest(),receipt
    assert receipt["context_sha256"]==hashlib.sha256(cb).hexdigest(),receipt
    assert receipt["tensor_components"]==4096 and not receipt["claim_ready"]
   print(label,json.dumps(receipt),flush=True)
   return receipt
  first=check("positive")
  second=check("permuted-layout-schedule",changes=dict(lane_stride=15,lane_offset=13,load=0,layout=1,unroll=16))
  assert first["plan_id"]!=second["plan_id"] and first["tensor_sha256"]==second["tensor_sha256"]
  for label,reason,changes in [
   ("bad-snapshot","SNAPSHOT",dict(context="0"*64)),
   ("bad-target","TARGET",dict(target=999)),
   ("precision","PRECISION",dict(precision=32)),
   ("order","ORDER",dict(order=2)),
   ("fma","FMA",dict(fma=1)),
   ("lane-alias","LANE_MAP",dict(lane_stride=2)),
   ("lane-range","LANE_MAP",dict(lane_offset=16)),
   ("unroll","UNROLL",dict(unroll=3)),
   ("unknown-kind","KIND",dict(kind=3)),
   ("kind2-without-phase","KIND_MASK",dict(kind=2)),
   ("kind1-with-phase","KIND_MASK",dict(phase=0)),
   ("phase-out-of-range","PHASE_RANGE",dict(kind=2,phase=65536)),
   ("result-injection","PROPOSAL_FORMAT",dict(expected=1)),
   ("authority-injection","PROPOSAL_FORMAT",dict(claim_ready=1))]:
    check(label,reason,changes)
  check("unknown-facts","UNKNOWN_OR_CONTRADICTORY_FACTS",context_changes=dict(facts_state=0))
  check("contradictory-facts","UNKNOWN_OR_CONTRADICTORY_FACTS",context_changes=dict(facts_state=2))
  check("missing-shuffle","CAPABILITY",context_changes=dict(capabilities=39))
  for label,transform in [
   ("duplicate",lambda b:b.replace(b'"schema":1',b'"schema":1,"schema":1')),
   ("trailing",lambda b:b+b"{}"),
   ("fraction",lambda b:b.replace(b'"unroll":1',b'"unroll":1.0')),
   ("boolean",lambda b:b.replace(b'"unroll":1',b'"unroll":true')),
   ("leading-zero",lambda b:b.replace(b'"unroll":1',b'"unroll":01')),
   ("trailing-comma",lambda b:b.rstrip()[:-1]+b",}"),
   ("nul",lambda b:b+b"\0"),
   ("negative",lambda b:b.replace(b'"unroll":1',b'"unroll":-1'))]:
    check(label,"PROPOSAL_FORMAT",transform=transform)
  # kind=2 proposes an operator, not a schedule. Phase 0 is the Cayley-Dickson
  # constant itself, so its tensor must be byte-identical to the kind=1 tensor:
  # the twist is an extension of the admitted algebra, not a replacement for it.
  base=check("operator-phase-0",changes=dict(kind=2,phase=0))
  assert base["kind"]=="operator" and base["phase_code"]==0,base
  assert base["tensor_sha256"]==first["tensor_sha256"],(base,first)
  # Well-definedness is all this engine establishes. Novelty belongs to the
  # GL(4,2) atlas and must stay undischarged in the receipt.
  assert base["novelty"]=="NOT_CLASSIFIED",base
  assert base["classifier"]=="eisa_h.pireus_operator_atlas.v1",base
  # Distinct phases name distinct operators: the phase code determines the
  # tensor, so a collision would mean the receipt does not identify what was
  # admitted. These eight codes land in six distinct GL(4,2) affine classes,
  # so the set is not a family of gauge-equivalent restatements of one twist.
  seen={base["tensor_sha256"]:0}
  for phase in (1128,2,8,74,198,1129,4,16):
   receipt=check("operator-phase-%d"%phase,changes=dict(kind=2,phase=phase))
   assert receipt["phase_code"]==phase,receipt
   assert receipt["tensor_sha256"] not in seen,(phase,seen[receipt["tensor_sha256"]],receipt)
   seen[receipt["tensor_sha256"]]=phase
  assert len(seen)==9,seen
  # emit_lowering_ptx encodes the Cayley-Dickson signs and takes no phase.
  # Materialising a twisted operator through it would emit the wrong kernel
  # silently, under a green receipt, so the engine must refuse outright.
  check("materialize-kind2","MATERIALIZE_KIND",changes=dict(kind=2,phase=1128),mode="ptx")
 print("PIREUS_EXTERNAL_ADMISSION_GATE_PASS",flush=True)
if __name__=="__main__":main()
