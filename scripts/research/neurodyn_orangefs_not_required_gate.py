#!/usr/bin/env python3
"""
neurodyn_orangefs_not_required_gate — gate P3 NeuroDyn (Python, generalizado).

Prova (não afirma) que o OrangeFS NÃO é requisito para a persistência científica
do P3, para DUAS classes de artefato:
  - pilot            (ds006731: p3_fmriprep_pilot_receipt.json, pilot_*/roi/*.1D)
  - balanced_manifest(ds002748: mdd_neurodyn_prepare_summary.json, mdd_neurodyn_manifest.tsv, rois/*.1D)
  - ossm_smoke       (ds006731: pacote O-SSM smoke com checkpoint/replay/controles)

Armadilha central que o gate mata: um artifact root pode começar com "/tmp/..."
mesmo estando em Ceph RBD/CephFS persistente. O gate resolve o BACKING REAL
(findmnt), nunca confia no nome do path.

4 CHECKS (todos precisam passar):
  1. artifact_landing_path      — artefatos da classe presentes, FORA do worker scratch
  2. stable_backing_filesystem  — backing = /dev/rbd* ou ceph/ceph-fuse; nunca overlay/tmpfs
  3. bundle_inclusion_hash      — evidence bundle presente; sha256 de cada artefato bate; bundle_sha256 raiz bate
  4. worker_loss_survivability  — nada aponta pro worker scratch; artefatos são arquivos reais legíveis

Uso:
  neurodyn_orangefs_not_required_gate.py <ARTIFACT_ROOT> [--worker-scratch PATH] [--emit-md] [--out-dir DIR]

Saída: JSON (schema brain_ossm.neurodyn_orangefs_not_required_gate.v1) no stdout;
       MD opcional em <out-dir> ou <ARTIFACT_ROOT>. Exit 0 = orangefs_not_required, 1 = still_required.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, datetime, glob

BUNDLE_NAME = "neurodyn_evidence_bundle_manifest.json"
GATE_SCHEMA = "brain_ossm.neurodyn_orangefs_not_required_gate.v1"
DEFAULT_WORKER_SCRATCH = "/tmp/neurodyn-ds006731-pilot"
DECISION_PASS = "orangefs_not_required_for_p3_roi_manifest"
DECISION_FAIL = "orangefs_still_required_for_p3"
CLAIM = ("Storage/reproducibility gate only. Atesta persistência durável+íntegra "
         "de ROI/manifest/receipts; não autoriza afirmação científica, clínica ou de biomarcador.")


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def within(child: str, parent: str) -> bool:
    child, parent = os.path.realpath(child), os.path.realpath(parent)
    return child == parent or child.startswith(parent.rstrip("/") + "/")


def detect_class(root: str) -> str | None:
    if os.path.isfile(os.path.join(root, "mdd_neurodyn_prepare_summary.json")):
        return "balanced_manifest"
    if os.path.isfile(os.path.join(root, "p3_fmriprep_pilot_receipt.json")):
        return "pilot"
    if os.path.isfile(os.path.join(root, "neurodyn_bucket8_smoke_summary.json")):
        return "ossm_smoke"
    return None


# ---- CHECK 1 --------------------------------------------------------------
def check_artifact_landing(root: str, cls: str, worker: str) -> tuple[bool, str]:
    if not os.path.isdir(root):
        return False, f"artifact_root não existe: {root}"
    if within(root, worker):
        return False, f"artifact_root está DENTRO do worker scratch ({worker}) — persistência ilusória"
    missing: list[str] = []
    if cls == "balanced_manifest":
        req = ["mdd_neurodyn_prepare_summary.json", "mdd_neurodyn_manifest.tsv",
               "mdd_neurodyn_temporal_shuffle_manifest.tsv",
               "mdd_neurodyn_temporal_reverse_manifest.tsv",
               "mdd_neurodyn_label_within_site_shuffle_manifest.tsv", "mdd_phenotypic.csv"]
        for f in req:
            if not os.path.isfile(os.path.join(root, f)):
                missing.append(f)
        if not glob.glob(os.path.join(root, "rois", "*.1D")):
            missing.append("rois/*.1D")
    elif cls == "pilot":
        req = ["p3_fmriprep_pilot_receipt.json", "p3_storage_preprocessing_gate.json"]
        for f in req:
            if not os.path.isfile(os.path.join(root, f)):
                missing.append(f)
        subs = glob.glob(os.path.join(root, "pilot_*"))
        if not subs:
            missing.append("pilot_*")
        for s in subs:
            if not glob.glob(os.path.join(s, "roi", "*.1D")):
                missing.append(f"{os.path.basename(s)}/roi/*.1D")
            if not os.path.isfile(os.path.join(s, "roi_phenotypic.csv")):
                missing.append(f"{os.path.basename(s)}/roi_phenotypic.csv")
    elif cls == "ossm_smoke":
        req = [
            "neurodyn_bucket8_smoke_summary.json",
            "neurodyn_bucket8_smoke_summary.tsv",
            "checkpoint_python_fresh_replay.json",
            "single_site_cv_test_results/sounio_structured/overall_metrics.json",
            "temporal_shuffle_results/sounio_structured/overall_metrics.json",
            "temporal_reverse_results/sounio_structured/overall_metrics.json",
            "label_within_site_shuffle_results/sounio_structured/overall_metrics.json",
        ]
        for f in req:
            if not os.path.isfile(os.path.join(root, f)):
                missing.append(f)
        if not glob.glob(os.path.join(root, "single_site_cv_test_results", "checkpoints", "*", "model.ckpt")):
            missing.append("single_site_cv_test_results/checkpoints/*/model.ckpt")
    else:
        return False, "classe de artefato não reconhecida (nem pilot, balanced_manifest, nem ossm_smoke)"
    if missing:
        return False, f"faltando: {missing}"
    return True, f"artefatos da classe '{cls}' presentes, fora do worker scratch"


# ---- CHECK 2 --------------------------------------------------------------
def check_stable_backing(root: str) -> tuple[bool, dict]:
    def fm(field: str) -> str:
        try:
            return subprocess.run(["findmnt", "-n", "-o", field, "--target", root],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    src, fstype, mnt = fm("SOURCE"), fm("FSTYPE"), fm("TARGET")
    info = {"device": src, "fstype": fstype, "mountpoint": mnt}
    if fstype in ("overlay", "tmpfs", "ramfs"):
        return False, {**info, "why": f"fstype '{fstype}' é EFÊMERO — não sobrevive ao pod/host"}
    if src.startswith("/dev/rbd"):
        return True, {**info, "why": f"Ceph RBD ({src}) — persistente/replicado"}
    if fstype in ("ceph", "fuse.ceph-fuse"):
        return True, {**info, "why": f"CephFS ({fstype}) — persistente/replicado"}
    return False, {**info, "why": f"backing '{src}'/'{fstype}' não reconhecido como persistente"}


# ---- CHECK 3 --------------------------------------------------------------
def check_bundle_hash(root: str) -> tuple[bool, str]:
    bpath = os.path.join(root, BUNDLE_NAME)
    if not os.path.isfile(bpath):
        return False, f"evidence bundle ausente ({BUNDLE_NAME}) — rode neurodyn_evidence_bundle.py build"
    try:
        bundle = json.load(open(bpath))
    except Exception as e:
        return False, f"bundle ilegível: {e}"
    arts = bundle.get("artifacts", [])
    if not arts:
        return False, "bundle sem artefatos"
    mism = []
    for a in arts:
        abs_ = os.path.join(root, a["path"])
        if not os.path.isfile(abs_):
            mism.append(f"{a['path']}(ausente)"); continue
        if sha256_file(abs_) != a["sha256"]:
            mism.append(f"{a['path']}(sha256)")
    # raiz determinística
    lines = "".join(f"{a['sha256']}  {a['path']}\n"
                    for a in sorted(arts, key=lambda x: x["path"]))
    root_ok = hashlib.sha256(lines.encode()).hexdigest() == bundle.get("bundle_sha256")
    if mism:
        return False, f"hash divergente: {mism}"
    if not root_ok:
        return False, "bundle_sha256 raiz não bate (bundle adulterado/inconsistente)"
    return True, f"{len(arts)} artefatos hash-verificados; bundle_sha256 raiz OK ({bundle.get('bundle_sha256','')[:16]}…)"


# ---- CHECK 4 --------------------------------------------------------------
def check_worker_survivability(root: str, worker: str) -> tuple[bool, str]:
    escapes = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.islink(p):
                tgt = os.path.realpath(p)
                if within(tgt, worker) or tgt.startswith("/tmp/neurodyn-"):
                    escapes.append(f"{os.path.relpath(p, root)} -> {tgt}")
    if escapes:
        return False, f"symlink(s) dependem do worker scratch: {escapes}"
    # artefatos-chave são arquivos regulares legíveis
    keys = glob.glob(os.path.join(root, "rois", "*.1D")) + \
           glob.glob(os.path.join(root, "pilot_*", "roi", "*.1D")) + \
           glob.glob(os.path.join(root, "single_site_cv_test_results", "checkpoints", "*", "model.ckpt")) + \
           [os.path.join(root, BUNDLE_NAME)]
    for k in keys:
        if os.path.exists(k) and not (os.path.isfile(k) and os.access(k, os.R_OK)):
            return False, f"não legível como arquivo real: {os.path.relpath(k, root)}"
    return True, "sem dependência de symlink no worker scratch; artefatos são arquivos reais legíveis"


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate P3: orangefs_not_required_for_p3_roi_manifest")
    ap.add_argument("artifact_root")
    ap.add_argument("--worker-scratch", default=DEFAULT_WORKER_SCRATCH)
    ap.add_argument("--emit-md", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    root = os.path.abspath(a.artifact_root)
    cls = detect_class(root)

    c1p, c1d = check_artifact_landing(root, cls, a.worker_scratch)
    c2p, c2i = check_stable_backing(root)
    c3p, c3d = check_bundle_hash(root)
    c4p, c4d = check_worker_survivability(root, a.worker_scratch)
    all_pass = all([c1p, c2p, c3p, c4p])
    decision = DECISION_PASS if all_pass else DECISION_FAIL

    result = {
        "schema": GATE_SCHEMA, "claim_boundary": CLAIM, "created_at_utc": now_utc(),
        "artifact_root": root, "artifact_class": cls, "worker_scratch": a.worker_scratch,
        "backing": {k: c2i.get(k) for k in ("device", "fstype", "mountpoint")},
        "checks": {
            "artifact_landing_path": {"pass": c1p, "detail": c1d},
            "stable_backing_filesystem": {"pass": c2p, "detail": c2i.get("why")},
            "bundle_inclusion_hash": {"pass": c3p, "detail": c3d},
            "worker_loss_survivability": {"pass": c4p, "detail": c4d},
        },
        "all_pass": all_pass, "decision": decision,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if a.emit_md:
        od = a.out_dir or root
        mark = lambda b: "✅" if b else "❌"
        md = [f"# Gate: {DECISION_PASS}", "",
              f"- **decisão:** `{decision}`  (`all_pass={all_pass}`)",
              f"- **artifact_root:** `{root}`  (classe: `{cls}`)",
              f"- **backing:** `{result['backing']['device']}` / `{result['backing']['fstype']}`",
              f"- **claim boundary:** {CLAIM}", "", "| check | pass | detalhe |", "|---|---|---|"]
        for k, v in result["checks"].items():
            md.append(f"| {k} | {mark(v['pass'])} | {v['detail']} |")
        try:
            open(os.path.join(od, "neurodyn_orangefs_not_required_gate.md"), "w").write("\n".join(md) + "\n")
        except Exception as e:
            sys.stderr.write(f"[md] falha ao escrever: {e}\n")

    sys.stderr.write(f"[gate] {decision}  (all_pass={all_pass}, classe={cls})\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
