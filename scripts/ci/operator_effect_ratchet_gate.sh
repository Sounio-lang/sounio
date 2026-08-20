#!/usr/bin/env bash
# operator_effect_ratchet_gate.sh — freeze `/` without Div and `%` without Mod.
#
# The compiler refuses those operators without the matching effect (E233).
# About eight thousand existing functions would need a declaration. This
# gate records the current residue and only allows it to shrink. It does
# not rewrite those functions.
#
# Usage:
#   bash scripts/ci/operator_effect_ratchet_gate.sh
#   bash scripts/ci/operator_effect_ratchet_gate.sh --write-frozen
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FROZEN="${OPERATOR_EFFECT_RATCHET_FROZEN:-$ROOT/scripts/ci/operator_effect_ratchet.frozen}"
WRITE=0
[[ "${1:-}" == "--write-frozen" ]] && WRITE=1

python3 - "$ROOT" "$FROZEN" "$WRITE" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
frozen_path = pathlib.Path(sys.argv[2])
write = sys.argv[3] == "1"

def mask(text):
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+2] == "//":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if text[i:i+2] == "/*":
            i += 2
            out.extend("  ")
            while i + 1 < n and text[i:i+2] != "*/":
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            if i + 1 < n:
                out.extend("  ")
                i += 2
            continue
        if text[i] == '"':
            out.append(" ")
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    out.extend("  ")
                    i += 2
                    continue
                if text[i] == '"':
                    out.append(" ")
                    i += 1
                    break
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)

FN = re.compile(r"\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*[<(]")
WITH = re.compile(r"\bwith\b([^{;/]+)")

def count(root: pathlib.Path):
    div_need = mod_need = 0
    for p in root.rglob("*.sio"):
        rel = str(p.relative_to(root))
        if rel.startswith(("archive/", "bootstrap/", "self-hosted/bootstrap/")):
            continue
        if rel.endswith(".sio.old"):
            continue
        masked = mask(p.read_text(encoding="utf-8", errors="replace"))
        parts = list(FN.finditer(masked))
        for i, m in enumerate(parts):
            chunk = masked[m.start() : parts[i + 1].start() if i + 1 < len(parts) else len(masked)]
            brace = chunk.find("{")
            sig = chunk[:brace] if brace >= 0 else chunk
            body = chunk[brace:] if brace >= 0 else ""
            decl = " ".join(WITH.findall(sig))
            has_div = re.search(r"\bDiv\b", decl) is not None
            has_mod = re.search(r"\bMod\b", decl) is not None
            if re.search(r"[A-Za-z0-9_)\]}\s]/[A-Za-z0-9_(\[\s]", body) and not has_div:
                div_need += 1
            if "%" in body and not has_mod:
                mod_need += 1
    return div_need, mod_need

div_need, mod_need = count(root)
print(f"OPERATOR_EFFECT_RATCHET div_without_Div={div_need} rem_without_Mod={mod_need}")

if write or not frozen_path.is_file():
    frozen_path.write_text(
        "# Frozen residue of operator-without-effect sites. Shrink-only.\n"
        f"div_without_Div={div_need}\n"
        f"rem_without_Mod={mod_need}\n",
        encoding="utf-8",
    )
    print(f"OPERATOR_EFFECT_RATCHET_WROTE {frozen_path}")
    raise SystemExit(0)

frozen = {}
for line in frozen_path.read_text(encoding="utf-8").splitlines():
    if line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    frozen[k.strip()] = int(v.strip())

fails = []
for key, now in (("div_without_Div", div_need), ("rem_without_Mod", mod_need)):
    was = frozen.get(key)
    if was is None:
        fails.append(f"missing_frozen_key={key}")
        continue
    if now > was:
        fails.append(f"{key}_rose:{was}->{now}")
    elif now < was:
        print(f"OPERATOR_EFFECT_RATCHET_SHRINK {key} {was}->{now} — lower the frozen file")

if fails:
    print("OPERATOR_EFFECT_RATCHET_FAIL")
    for f in fails:
        print(f"  {f}")
    raise SystemExit(1)
print("OPERATOR_EFFECT_RATCHET_OK")
PY
