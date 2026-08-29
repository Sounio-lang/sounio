#!/usr/bin/env python3
"""
neurodyn_evidence_bundle — camada de integridade (hash) para o manifest P3 NeuroDyn.

Motivação: os artefatos NeuroDyn (mdd_neurodyn_manifest.tsv, null-models
temporal_shuffle/temporal_reverse/label_within_site_shuffle, phenotypic, ROIs,
prepare_summary.json) hoje são referenciados só por PATH — sem integridade de
conteúdo. Sem hash, não há atestado cirúrgico de reprodutibilidade, e o gate
orangefs_not_required_for_p3_roi_manifest não consegue provar o check 3.

Este script emite/verifica um evidence bundle com sha256 por artefato e uma raiz
determinística (bundle_sha256), casando com o check 3 do gate.

Uso:
  build:   neurodyn_evidence_bundle.py build  <ARTIFACT_ROOT> [--summary NAME] [--dry]
  verify:  neurodyn_evidence_bundle.py verify <ARTIFACT_ROOT> [--bundle NAME]

  <ARTIFACT_ROOT> = dir que contém o prepare_summary.json (ex.:
    .../ds002748/bounded_sub01_05_ctrl52_56/manifest_balanced_fmriprep)
    ou um pacote de smoke O-SSM com neurodyn_bucket8_smoke_summary.json.

Saídas (no ARTIFACT_ROOT, ou stdout com --dry):
  neurodyn_evidence_bundle_manifest.json   (schema brain_ossm.neurodyn_evidence_bundle.v1)
  neurodyn_evidence_bundle_verify.json     (schema ...verify.v1, no verify)

Exit: 0 ok, 1 falha (build: artefato faltando; verify: hash divergente).
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, datetime

BUNDLE_NAME = "neurodyn_evidence_bundle_manifest.json"
VERIFY_NAME = "neurodyn_evidence_bundle_verify.json"
SUMMARY_DEFAULT = "mdd_neurodyn_prepare_summary.json"
CLAIM = ("Content-integrity manifest only. Fixa hashes de artefatos P3; "
         "não autoriza nenhuma afirmação científica, clínica ou de biomarcador.")


def sha256_file(path: str, buf: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha256(); n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b); n += len(b)
    return h.hexdigest(), n


def resolve(root: str, ref: str) -> str | None:
    """Resolve um artefato referenciado (path absoluto do summary OU relativo)
    contra o ARTIFACT_ROOT local, por basename — portável entre hosts."""
    for cand in (ref, os.path.join(root, ref), os.path.join(root, os.path.basename(ref))):
        if cand and os.path.isfile(cand):
            return cand
    return None


def collect_smoke(root: str) -> tuple[list[dict], list[str]]:
    required = [
        "neurodyn_bucket8_smoke_summary.json",
        "neurodyn_bucket8_smoke_summary.tsv",
        "checkpoint_python_fresh_replay.json",
        "single_site_cv_test_results/checkpoints/neurodyn-ds006731-bucket8-original-precompiled-20260705T173500Z/model.ckpt",
    ]
    missing = [name for name in required if not os.path.isfile(os.path.join(root, name))]
    if missing:
        return [], missing
    artifacts = []
    skip = {
        "SHA256SUMS",
        BUNDLE_NAME,
        VERIFY_NAME,
        "neurodyn_orangefs_not_required_gate.json",
        "neurodyn_orangefs_not_required_gate.md",
    }
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename in skip:
                continue
            abs_path = os.path.join(dirpath, filename)
            rel = os.path.relpath(abs_path, root)
            digest, size = sha256_file(abs_path)
            role = "ossm_smoke"
            if rel.endswith("model.ckpt"):
                role = "checkpoint"
            elif rel == "checkpoint_python_fresh_replay.json":
                role = "checkpoint_replay"
            elif rel.startswith("single_site_cv_test_results/"):
                role = "original_run_output"
            elif rel.startswith("temporal_shuffle_results/"):
                role = "null_output.temporal_shuffle"
            elif rel.startswith("temporal_reverse_results/"):
                role = "null_output.temporal_reverse"
            elif rel.startswith("label_within_site_shuffle_results/"):
                role = "null_output.label_within_site_shuffle"
            artifacts.append({
                "role": role,
                "path": rel,
                "sha256": digest,
                "bytes": size,
            })
    return artifacts, []


def collect(root: str, summary_name: str) -> tuple[list[dict], list[str]]:
    """Coleta (role, path_rel, abs) de todo artefato científico a hashear."""
    if os.path.isfile(os.path.join(root, "neurodyn_bucket8_smoke_summary.json")):
        return collect_smoke(root)

    spath = os.path.join(root, summary_name)
    missing: list[str] = []
    if not os.path.isfile(spath):
        return [], [summary_name]
    with open(spath) as f:
        summ = json.load(f)

    wanted: list[tuple[str, str]] = [("summary", spath)]
    man = summ.get("manifests", {})
    for role_key in ("original", "temporal_shuffle", "temporal_reverse",
                     "label_within_site_shuffle"):
        ref = man.get(role_key)
        if ref:
            role = "roi_manifest" if role_key == "original" else f"null_manifest.{role_key}"
            a = resolve(root, ref)
            wanted.append((role, a)) if a else missing.append(os.path.basename(ref))
    for key, role in (("phenotypic_output", "phenotypic"),):
        ref = summ.get(key)
        if ref:
            a = resolve(root, ref)
            wanted.append((role, a)) if a else missing.append(os.path.basename(ref))
    # ROIs no roi_cache_dir
    roidir_ref = summ.get("roi_cache_dir")
    roidir = None
    if roidir_ref:
        for cand in (roidir_ref, os.path.join(root, os.path.basename(roidir_ref))):
            if os.path.isdir(cand):
                roidir = cand; break
    if roidir:
        for fn in sorted(os.listdir(roidir)):
            if fn.endswith(".1D"):
                wanted.append(("roi", os.path.join(roidir, fn)))
    else:
        missing.append("roi_cache_dir")
    # inventory opcional
    inv = os.path.join(root, "mdd_neurodyn_inventory.tsv")
    if os.path.isfile(inv):
        wanted.append(("inventory", inv))

    artifacts = []
    for role, abs_ in wanted:
        digest, size = sha256_file(abs_)
        artifacts.append({
            "role": role,
            "path": os.path.relpath(abs_, root),
            "sha256": digest,
            "bytes": size,
        })
    return artifacts, missing


def bundle_root(artifacts: list[dict]) -> str:
    """Raiz determinística: sha256 sobre linhas 'sha256␠path\\n' ordenadas por path."""
    lines = "".join(f"{a['sha256']}  {a['path']}\n"
                    for a in sorted(artifacts, key=lambda x: x["path"]))
    return hashlib.sha256(lines.encode()).hexdigest()


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_build(args) -> int:
    root = os.path.abspath(args.artifact_root)
    artifacts, missing = collect(root, args.summary)
    if missing:
        sys.stderr.write(f"[build] FALHA — artefatos faltando: {missing}\n")
        return 1
    dataset_id = "unknown"
    try:
        with open(os.path.join(root, args.summary)) as f:
            dataset_id = json.load(f).get("dataset_id", "unknown")
    except Exception:
        pass
    bundle = {
        "schema": "brain_ossm.neurodyn_evidence_bundle.v1",
        "claim_boundary": CLAIM,
        "created_at_utc": now_utc(),
        "dataset_id": dataset_id,
        "source_summary": args.summary,
        "artifact_count": len(artifacts),
        "total_bytes": sum(a["bytes"] for a in artifacts),
        "artifacts": sorted(artifacts, key=lambda x: x["path"]),
        "bundle_sha256": bundle_root(artifacts),
    }
    out = json.dumps(bundle, indent=2, ensure_ascii=False)
    if args.dry:
        print(out)
    else:
        with open(os.path.join(root, BUNDLE_NAME), "w") as f:
            f.write(out + "\n")
        sys.stderr.write(f"[build] OK — {len(artifacts)} artefatos, "
                         f"bundle_sha256={bundle['bundle_sha256'][:16]}… → {BUNDLE_NAME}\n")
    return 0


def cmd_verify(args) -> int:
    root = os.path.abspath(args.artifact_root)
    bpath = os.path.join(root, args.bundle)
    if not os.path.isfile(bpath):
        sys.stderr.write(f"[verify] FALHA — bundle ausente: {bpath}\n"); return 1
    with open(bpath) as f:
        bundle = json.load(f)
    mism = []
    for a in bundle.get("artifacts", []):
        abs_ = os.path.join(root, a["path"])
        if not os.path.isfile(abs_):
            mism.append({"path": a["path"], "reason": "arquivo ausente"}); continue
        got, size = sha256_file(abs_)
        if got != a["sha256"]:
            mism.append({"path": a["path"], "reason": "sha256 divergente",
                         "expected": a["sha256"], "got": got})
    recomputed = bundle_root([{"path": a["path"], "sha256": a["sha256"]}
                              for a in bundle.get("artifacts", [])])
    root_ok = recomputed == bundle.get("bundle_sha256")
    all_ok = not mism and root_ok
    result = {
        "schema": "brain_ossm.neurodyn_evidence_bundle_verify.v1",
        "created_at_utc": now_utc(),
        "bundle": args.bundle,
        "bundle_sha256_expected": bundle.get("bundle_sha256"),
        "bundle_sha256_recomputed": recomputed,
        "bundle_root_match": root_ok,
        "artifact_count": len(bundle.get("artifacts", [])),
        "all_match": all_ok,
        "mismatches": mism,
    }
    out = json.dumps(result, indent=2, ensure_ascii=False)
    if args.dry:
        print(out)
    else:
        with open(os.path.join(root, VERIFY_NAME), "w") as f:
            f.write(out + "\n")
    sys.stderr.write(f"[verify] {'OK ✓' if all_ok else 'FALHA ✗'} — "
                     f"{len(mism)} divergência(s), root_match={root_ok}\n")
    return 0 if all_ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="NeuroDyn P3 evidence bundle (hash layer)")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("artifact_root")
    b.add_argument("--summary", default=SUMMARY_DEFAULT); b.add_argument("--dry", action="store_true")
    b.set_defaults(fn=cmd_build)
    v = sub.add_parser("verify"); v.add_argument("artifact_root")
    v.add_argument("--bundle", default=BUNDLE_NAME); v.add_argument("--dry", action="store_true")
    v.set_defaults(fn=cmd_verify)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
