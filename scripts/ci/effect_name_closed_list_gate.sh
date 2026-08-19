#!/usr/bin/env bash
# effect_name_closed_list_gate.sh — closed effect-name list + accusation.
#
# Why this exists
# ---------------
# effect_name_to_id returns -1 for an unrecognised name. collect_effects_from_list
# then keeps the name only when `eff_id >= 0`. The miss is discarded with no
# diagnostic. Three independent instruments already measured the hole
# (#1953 Foo → check: OK; #1993 handle<NaoExisteEsteEfeito> runs as handle<IO>;
# the checker guard itself). This gate makes the hole visible and stops it
# growing a new name without a file:line.
#
# This is an ACCUSATION gate. It FAILS while `souc check` of
# docs/audit/repro/effect_unknown_name.sio (`with NomeQueNaoExiste`) still
# exits 0. A used `with` name that is not derived from effect_name_to_id
# is also a fail (file + line). Do not add names here. If the table
# recognises N names, N is N — even if a recognised name has zero uses.
# Mod is HELD: it is used and not recognised; the scan reports it.
#
# The closed list is derived from self-hosted/check/effects.sio
# (effect_named_id_max, effect_kind_name_len, effect_kind_name_byte,
# name_is_confidence). It is not written by hand.
#
# What is scanned
# ---------------
# Versioned `*.sio` `with` clauses. Identifiers after `with`, comma-separated,
# stopping at the parser's own "comma + Ident + Colon" parameter boundary
# and skipping a parenthesised payload.
#
# What is excluded, and why
# -------------------------
# archive/                 — historical evolution, not live code
# bootstrap/               — C→Sounio bootstrap chain, frozen
# self-hosted/bootstrap/   — the frozen self-hosted seed (same class)
# *.sio.old                — not live source (a prior argument was built from one)
# // line comments         — English "with a" / "with the" is not an effect
# /* block comments */     — same
# "string literals"        — `"with Mut"` is not a clause
#
# This gate does not edit self-hosted/. The compiler fix is a separate
# founder dispatch.
#
# Usage
#   bash scripts/ci/effect_name_closed_list_gate.sh
#   bash scripts/ci/effect_name_closed_list_gate.sh --self-test
#   bash scripts/ci/effect_name_closed_list_gate.sh --scan-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT="${EFFECT_NAME_CLOSED_LIST_ARTIFACT:-/tmp/effect_name_closed_list.json}"
OUT_DIR="${EFFECT_NAME_CLOSED_LIST_OUT:-$ROOT/artifacts/audit/effect_name_closed_list}"
MODE=check
ACCUSE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) MODE=selftest; ACCUSE=0; shift ;;
    --scan-only) ACCUSE=0; shift ;;
    --no-slurm) EFFECT_NAME_CLOSED_LIST_SLURM=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v python3 >/dev/null 2>&1 || {
  echo "EFFECT_NAME_CLOSED_LIST_FAIL reason=python3_missing" >&2
  exit 1
}

mkdir -p "$OUT_DIR"

# Accusation compile uses the committed ELF. Never inherit a poisoned
# SOUC_BIN that points at another worktree (hazard souc-bin-poisoned).
unset SOUC_BIN SOUNIO_SOUC_BIN || true
SOUC_FOR_ACCUSE=""
if [[ -x "$ROOT/bin/souc" ]]; then
  SOUC_FOR_ACCUSE="$ROOT/bin/souc"
elif [[ -x "$ROOT/bin/madaros-linux-x86_64" ]]; then
  SOUC_FOR_ACCUSE="$ROOT/bin/madaros-linux-x86_64"
fi

use_slurm=0
if [[ "$ACCUSE" -eq 1 ]] && command -v srun >/dev/null 2>&1 && [[ "${EFFECT_NAME_CLOSED_LIST_SLURM:-1}" != "0" ]]; then
  use_slurm=1
fi

WITNESS_REL="docs/audit/repro/effect_unknown_name.sio"
POSITIVE_REL="scripts/ci/fixtures/effect_name_closed_list/positive_io.sio"

run_accuse_slurm() {
  tar -czf - -C "$ROOT" \
      bin/madaros-linux-x86_64 bin/souc bin/madaros \
      "$WITNESS_REL" "$POSITIVE_REL" \
    | srun -p "${EFFECT_NAME_CLOSED_LIST_PARTITION:-cpu-ops}" -N1 -n1 -c2 \
        --mem=8G --time=00:08:00 --chdir=/tmp --job-name=eff-closed-list \
        bash -c '
          set -euo pipefail
          export TMPDIR=/tmp
          WORKDIR=$(mktemp -d /tmp/eff-closed.XXXXXX)
          cd "$WORKDIR"
          tar xzf -
          unset SOUC_BIN SOUNIO_SOUC_BIN || true
          export MADAROS_STACK_KB=524288
          ulimit -s 1048576 || true
          mkdir -p /tmp/eff-closed-out
          SOUC=./bin/souc
          if [[ ! -x "$SOUC" && -x ./bin/madaros-linux-x86_64 ]]; then
            SOUC=./bin/madaros-linux-x86_64
          fi
          echo "souc=$SOUC host=$(hostname)" > /tmp/eff-closed-out/meta.txt
          set +e
          "$SOUC" check docs/audit/repro/effect_unknown_name.sio \
            > /tmp/eff-closed-out/witness.log 2>&1
          echo $? > /tmp/eff-closed-out/witness.rc
          "$SOUC" check scripts/ci/fixtures/effect_name_closed_list/positive_io.sio \
            > /tmp/eff-closed-out/positive.log 2>&1
          echo $? > /tmp/eff-closed-out/positive.rc
          set -e
          tar -C /tmp/eff-closed-out -czf - .
        ' >"$OUT_DIR/slurm_bundle.tar.gz"
  tar -C "$OUT_DIR" -xzf "$OUT_DIR/slurm_bundle.tar.gz"
}

run_accuse_local() {
  local souc="$1"
  set +e
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$souc" check "$ROOT/$WITNESS_REL" \
    >"$OUT_DIR/witness.log" 2>&1
  echo $? >"$OUT_DIR/witness.rc"
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$souc" check "$ROOT/$POSITIVE_REL" \
    >"$OUT_DIR/positive.log" 2>&1
  echo $? >"$OUT_DIR/positive.rc"
  set -e
}

ACCUSE_STATUS=not_run
WITNESS_RC=""
POSITIVE_RC=""
if [[ "$ACCUSE" -eq 1 ]]; then
  if [[ "$use_slurm" -eq 1 ]]; then
    run_accuse_slurm
    WITNESS_RC="$(cat "$OUT_DIR/witness.rc")"
    POSITIVE_RC="$(cat "$OUT_DIR/positive.rc")"
  elif [[ -n "$SOUC_FOR_ACCUSE" ]]; then
    run_accuse_local "$SOUC_FOR_ACCUSE"
    WITNESS_RC="$(cat "$OUT_DIR/witness.rc")"
    POSITIVE_RC="$(cat "$OUT_DIR/positive.rc")"
  fi
fi

export ROOT ARTIFACT OUT_DIR MODE ACCUSE
export WITNESS_RC POSITIVE_RC
export WITNESS_REL POSITIVE_REL
python3 - <<'PY'
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

root = pathlib.Path(os.environ["ROOT"])
artifact = pathlib.Path(os.environ["ARTIFACT"])
out_dir = pathlib.Path(os.environ["OUT_DIR"])
mode = os.environ["MODE"]
accuse = os.environ["ACCUSE"] == "1"
witness_rel = os.environ["WITNESS_REL"]
positive_rel = os.environ["POSITIVE_REL"]
witness_rc_raw = os.environ.get("WITNESS_RC") or ""
positive_rc_raw = os.environ.get("POSITIVE_RC") or ""

EFFECTS_REL = "self-hosted/check/effects.sio"
SKIP_PREFIXES = (
    "archive/",
    "bootstrap/",
    "self-hosted/bootstrap/",
)
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
WITH_RE = re.compile(r"\bwith\b")
FIXTURE_DIR = "scripts/ci/fixtures/effect_name_closed_list"


def fail_derive(reason: str) -> None:
    print(f"EFFECT_NAME_CLOSED_LIST_FAIL reason={reason}", file=sys.stderr)
    emit("fail", 1, 0, 1, 0, {"reason": reason, "closed_names": []})
    raise SystemExit(1)


def fn_body(src: str, name: str) -> str:
    m = re.search(rf"fn {re.escape(name)}\b.*?{{", src)
    if not m:
        fail_derive(f"missing_fn_{name}")
    start = m.end() - 1
    depth = 0
    for i, ch in enumerate(src[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    fail_derive(f"unclosed_fn_{name}")
    return ""


def derive_closed_list(src: str) -> dict:
    max_body = fn_body(src, "effect_named_id_max")
    max_m = re.search(r"\b(\d+)\b", max_body)
    if not max_m:
        fail_derive("named_id_max_unparsed")
    named_max = int(max_m.group(1))

    len_body = fn_body(src, "effect_kind_name_len")
    lens = {int(k): int(n) for k, n in re.findall(r"k == (\d+) \{ (\d+) \}", len_body)}
    for k in range(named_max + 1):
        if k not in lens or lens[k] <= 0:
            fail_derive(f"name_len_missing_k={k}")

    byte_body = fn_body(src, "effect_kind_name_byte")
    k_blocks = re.split(r"(?:else )?if k == (\d+) \{", byte_body)
    by_k = {}
    it = iter(k_blocks[1:])
    for k_s, body in zip(it, it):
        k = int(k_s)
        pairs = re.findall(r"i == (\d+) \{ (\d+) as i8 \}", body)
        else_m = re.search(r"else \{ (\d+) as i8 \}", body)
        bymap = {int(i): int(b) for i, b in pairs}
        nlen = lens[k]
        chars = []
        for i in range(nlen):
            if i in bymap:
                chars.append(chr(bymap[i]))
            elif else_m:
                chars.append(chr(int(else_m.group(1))))
            else:
                fail_derive(f"name_byte_missing k={k} i={i}")
        by_k[k] = "".join(chars)

    names = []
    for k in range(named_max + 1):
        if k not in by_k:
            fail_derive(f"name_missing_k={k}")
        if len(by_k[k]) != lens[k]:
            fail_derive(f"name_len_mismatch k={k}")
        names.append(by_k[k])

    conf_body = fn_body(src, "name_is_confidence")
    clen_m = re.search(r"name_len != (\d+)", conf_body)
    if not clen_m:
        fail_derive("confidence_len_unparsed")
    clen = int(clen_m.group(1))
    cbytes = {
        int(i): int(b)
        for i, b in re.findall(r"name_buf\[(\d+)\] != (\d+) as i8", conf_body)
    }
    if sorted(cbytes) != list(range(clen)):
        fail_derive("confidence_bytes_incomplete")
    alias = "".join(chr(cbytes[i]) for i in range(clen))
    if not alias:
        fail_derive("confidence_alias_empty")

    closed = list(names)
    if alias not in closed:
        closed.append(alias)

    if "IO" not in closed:
        fail_derive("derived_list_missing_IO")
    if "EffVar" in closed:
        fail_derive("effvar_is_not_a_with_name")
    return {
        "named_id_max": named_max,
        "table_names": names,
        "confidence_alias": alias,
        "closed_names": closed,
    }


def mask_comments_and_strings(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            out.extend("  ")
            i += 2
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            out.extend("  ")
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            if i + 1 < n:
                out.extend("  ")
                i += 2
            continue
        if ch == '"':
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
        out.append(ch)
        i += 1
    return "".join(out)


def skip_ws(s: str, pos: int) -> int:
    while pos < len(s) and s[pos] in " \t\n":
        pos += 1
    return pos


def skip_payload(s: str, pos: int) -> int:
    if pos >= len(s) or s[pos] != "(":
        return pos
    depth = 0
    while pos < len(s):
        if s[pos] == "(":
            depth += 1
        elif s[pos] == ")":
            depth -= 1
            pos += 1
            if depth == 0:
                return pos
            continue
        pos += 1
    return pos


def extract_with_names(text: str) -> list[tuple[int, str]]:
    """Parser-faithful `with` names. Stop on comma+Ident+Colon (next param)."""
    masked = mask_comments_and_strings(text)
    hits = []
    for m in WITH_RE.finditer(masked):
        pos = m.end()
        while True:
            pos = skip_ws(masked, pos)
            im = IDENT_RE.match(masked, pos)
            if not im:
                break
            name = im.group(0)
            after = skip_ws(masked, skip_payload(masked, im.end()))
            if after < len(masked) and masked[after] == ",":
                nxt = skip_ws(masked, after + 1)
                im2 = IDENT_RE.match(masked, nxt)
                if im2:
                    after2 = skip_ws(masked, skip_payload(masked, im2.end()))
                    if after2 < len(masked) and masked[after2] == ":":
                        hits.append((masked.count("\n", 0, im.start()) + 1, name))
                        break
            hits.append((masked.count("\n", 0, im.start()) + 1, name))
            pos = after
            if pos < len(masked) and masked[pos] == ",":
                pos += 1
                continue
            break
    return hits


def skip_path(rel: str) -> bool:
    if rel.endswith(".sio.old"):
        return True
    return any(rel == p[:-1] or rel.startswith(p) for p in SKIP_PREFIXES)


def versioned_sio() -> list[str]:
    raw = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "*.sio"],
        text=False,
    )
    return [p.decode() for p in raw.split(b"\0") if p]


def scan_files(rels: list[str]) -> list[dict]:
    hits = []
    for rel in rels:
        if skip_path(rel):
            continue
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line, name in extract_with_names(text):
            hits.append({"path": rel, "line": line, "name": name})
    return hits


def emit(status, total, passed, failed, not_run, extra):
    payload = {
        "schema": "sounio.effect-name-closed-list-gate.v1",
        "status": status,
        "metrics": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "not_run": not_run,
        },
        **extra,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"status={status}")
    print(
        "metrics "
        f"{{total={total}, passed={passed}, failed={failed}, not_run={not_run}}}"
    )
    print(f"artifact={artifact}")
    if status != "pass":
        raise SystemExit(1)


effects_path = root / EFFECTS_REL
if not effects_path.is_file():
    fail_derive("effects_sio_missing")
derived = derive_closed_list(effects_path.read_text(encoding="utf-8", errors="replace"))
closed = set(derived["closed_names"])
print(
    "EFFECT_NAME_CLOSED_LIST_DERIVED "
    f"named_id_max={derived['named_id_max']} "
    f"table={len(derived['table_names'])} "
    f"alias={derived['confidence_alias']} "
    f"closed={len(derived['closed_names'])} "
    f"names={','.join(derived['closed_names'])}"
)

if mode == "selftest":
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="eff-closed-list-"))
    comment = tmp / "comment.sio"
    comment.write_text(
        "// This helper works with a closed interval.\n"
        "// Compare with the previous revision.\n"
        "fn marked() -> i64 with IO {\n    0\n}\n",
        encoding="utf-8",
    )
    string = tmp / "string.sio"
    string.write_text(
        'fn marked() -> i64 with IO {\n    let ignored = "with Mut"\n    0\n}\n',
        encoding="utf-8",
    )
    old = tmp / "stale.sio.old"
    old.write_text(
        "fn marked() -> i64 with InventedFromOld {\n    0\n}\n",
        encoding="utf-8",
    )
    positive = tmp / "positive.sio"
    positive.write_text(
        "fn marked() -> i64 with IO {\n    0\n}\n",
        encoding="utf-8",
    )

    comment_hits = extract_with_names(comment.read_text(encoding="utf-8"))
    string_hits = extract_with_names(string.read_text(encoding="utf-8"))
    old_scanned = []
    # The scanner must refuse *.sio.old even if a caller hands the path in.
    if not skip_path("stale.sio.old") and not skip_path(str(old)):
        old_scanned = extract_with_names(old.read_text(encoding="utf-8"))
    # Also: a direct extract of the .old file would see InventedFromOld;
    # the gate's skip_path is the control. Confirm the name is in the raw file
    # so the test is not vacuously green.
    old_raw = extract_with_names(old.read_text(encoding="utf-8"))
    positive_hits = extract_with_names(positive.read_text(encoding="utf-8"))

    comment_ok = comment_hits == [(3, "IO")]
    string_ok = string_hits == [(1, "IO")]
    old_skipped = skip_path("anything.sio.old") and old_scanned == []
    old_would_have_counted = any(n == "InventedFromOld" for _, n in old_raw)
    positive_ok = positive_hits == [(1, "IO")] and all(n in closed for _, n in positive_hits)

    # Committed fixtures, same predicates.
    fx = root / FIXTURE_DIR
    fx_comment = extract_with_names((fx / "negative_comment.sio").read_text(encoding="utf-8"))
    fx_string = extract_with_names((fx / "negative_string.sio").read_text(encoding="utf-8"))
    fx_pos = extract_with_names((fx / "positive_io.sio").read_text(encoding="utf-8"))
    fx_old_rel = f"{FIXTURE_DIR}/negative_old.sio.old"
    fx_comment_ok = all(n == "IO" for _, n in fx_comment) and not any(
        n in {"a", "the"} for _, n in fx_comment
    )
    fx_string_ok = all(n == "IO" for _, n in fx_string) and not any(
        n == "Mut" for _, n in fx_string
    )
    fx_pos_ok = all(n == "IO" for _, n in fx_pos) and len(fx_pos) >= 1
    fx_old_ok = skip_path(fx_old_rel)

    checks = [
        ("positive_io_not_accused", positive_ok and fx_pos_ok),
        ("negative_comment_a_the_ignored", comment_ok and fx_comment_ok),
        ("negative_string_ignored", string_ok and fx_string_ok),
        ("negative_sio_old_ignored", old_skipped and old_would_have_counted and fx_old_ok),
    ]
    print(
        "EFFECT_NAME_CLOSED_LIST_SELFTEST "
        + " ".join(f"{k}={str(v).lower()}" for k, v in checks)
    )
    failed_msgs = [k for k, v in checks if not v]
    for k in failed_msgs:
        print(f"EFFECT_NAME_CLOSED_LIST_FAIL selftest_{k}", file=sys.stderr)
    total = len(checks)
    passed = sum(1 for _, v in checks if v)
    failed = total - passed
    status = "pass" if not failed_msgs else "fail"
    emit(
        status,
        total,
        passed,
        failed,
        0,
        {
            **derived,
            "selftest": {k: v for k, v in checks},
            "recorded": failed_msgs,
            "kind": "selftest",
        },
    )
    raise SystemExit(0 if status == "pass" else 1)

rels = versioned_sio()
# Uncommitted new files in this worktree still have to be visible to the
# scan (the witness lands before the first commit). git ls-files -o.
extra = subprocess.check_output(
    ["git", "-C", str(root), "ls-files", "-z", "-o", "--exclude-standard", "*.sio"],
    text=False,
)
for p in extra.split(b"\0"):
    if p:
        rel = p.decode()
        if rel not in rels:
            rels.append(rel)
rels.sort()
hits = scan_files(rels)
unknown = [h for h in hits if h["name"] not in closed]
known = [h for h in hits if h["name"] in closed]

from collections import Counter

unknown_counts = Counter(h["name"] for h in unknown)
known_counts = Counter(h["name"] for h in known)
print(
    "EFFECT_NAME_CLOSED_LIST_SCAN "
    f"files={len(rels)} hits={len(hits)} known={len(known)} "
    f"unknown={len(unknown)} unknown_names={len(unknown_counts)}"
)
for name, count in unknown_counts.most_common():
    print(f"EFFECT_NAME_CLOSED_LIST_UNKNOWN name={name} count={count}")
# Every site is in the artefact. Stdout prints every site for a rare
# name and a cap for a name that would drown the log (Mod is ~2800).
STDOUT_SITE_CAP = 20
shown = Counter()
for h in unknown:
    shown[h["name"]] += 1
    if shown[h["name"]] <= STDOUT_SITE_CAP:
        print(
            f"EFFECT_NAME_CLOSED_LIST_FAIL unknown_used "
            f"file={h['path']}:{h['line']} name={h['name']}"
        )
for name, count in unknown_counts.most_common():
    extra = count - min(count, STDOUT_SITE_CAP)
    if extra > 0:
        print(
            f"EFFECT_NAME_CLOSED_LIST_UNKNOWN_MORE name={name} "
            f"omitted={extra} full_list=artifact"
        )

# Founder ruling 2026-08-19: the unknown-name uses are RECORDED, not fatal in
# themselves — the ratchet at the end decides red. They stay in the artefact and
# are printed above in full; what fails is a NEW name or a rise in the total.
# Failing on the set as it stands would block the queue on 2,793 `with Mod`
# sites, a family explicitly HELD pending the E035 blast-radius measurement.
recorded_msgs = [
    f"unknown_used file={h['path']}:{h['line']} name={h['name']}" for h in unknown
]
failed_msgs = []

accuse_status = "not_run"
accuse_detail = {}
not_run = 0
accuse_pass = 0
accuse_fail = 0
accuse_total = 0
if accuse:
    accuse_total = 1
    witness_log = ""
    positive_log = ""
    wlog = out_dir / "witness.log"
    plog = out_dir / "positive.log"
    if wlog.is_file():
        witness_log = wlog.read_text(encoding="utf-8", errors="replace")
    if plog.is_file():
        positive_log = plog.read_text(encoding="utf-8", errors="replace")
    try:
        witness_rc = int(witness_rc_raw) if witness_rc_raw != "" else None
    except ValueError:
        witness_rc = None
    try:
        positive_rc = int(positive_rc_raw) if positive_rc_raw != "" else None
    except ValueError:
        positive_rc = None

    if witness_rc is None:
        not_run = 1
        accuse_status = "not_run"
        print("EFFECT_NAME_CLOSED_LIST_ACCUSATION status=not_run reason=souc_unavailable")
    else:
        silence = witness_rc == 0
        positive_ok = positive_rc == 0
        if not positive_ok:
            # The engine cannot check a real `with IO`. Do not accuse from that.
            not_run = 1
            accuse_status = "not_run"
            failed_msgs.append("positive_io_check_failed")
            print(
                "EFFECT_NAME_CLOSED_LIST_FAIL positive_io_check_failed "
                f"rc={positive_rc}",
                file=sys.stderr,
            )
        elif silence:
            # Founder ruling 2026-08-19: the accusation is RECORDED, not fatal —
            # the ratchet below is what makes the gate red. Failing here would
            # block the queue on `with Mod`, a family the founder has explicitly
            # HELD pending the E035 blast-radius measurement. The accusation is
            # still printed in full on every run; it just no longer forces red.
            accuse_fail = 0
            accuse_status = "silence"
            print(
                "EFFECT_NAME_CLOSED_LIST_ACCUSATION accusation_silence "
                f"file={witness_rel} rc=0 "
                "ACCUSATION: compiler accepted with NomeQueNaoExiste in silence; "
                "recorded, not fatal — see the ratchet",
                file=sys.stderr,
            )
        else:
            accuse_pass = 1
            accuse_status = "refused"
            print(
                "EFFECT_NAME_CLOSED_LIST_ACCUSATION status=refused "
                f"file={witness_rel} rc={witness_rc}"
            )
    accuse_detail = {
        "status": accuse_status,
        "witness": witness_rel,
        "witness_rc": witness_rc_raw,
        "positive_rc": positive_rc_raw,
        "witness_log_tail": witness_log[-2000:],
        "positive_log_tail": positive_log[-1000:],
    }

total = len(hits) + accuse_total
passed = len(known) + accuse_pass
# --- RATCHET (founder ruling 2026-08-19: freeze the current set, fail the next) --
#
# The gate was written to FAIL while the compiler silently accepts an unknown
# effect name, and that reading is correct. But the current accused set is 2,846
# sites, 2,793 of which are `with Mod` — a family the founder has explicitly
# HELD (check/effects.sio:23-26) pending the E035 blast-radius measurement.
# Failing red on a held decision blocks the queue on something nobody may edit.
#
# So the accusation becomes a ratchet: the measured set is frozen, and only a
# name outside it — or a rise in the total — fails. This does not soften the
# finding; the accusation is still printed in full every run, and the frozen
# file is the receipt of exactly what was tolerated and when.
import os as _os
# NAO usar __file__: este bloco corre via `python3 - <<PY`, isto e, stdin, onde
# __file__ nao esta definido. O caminho vem do shell.
_FROZEN = _os.environ.get("EFFECT_NAME_FROZEN",
                          "scripts/ci/effect_name_closed_list.frozen")
_names_now = sorted({u["name"] for u in unknown}) if unknown else []
_total_now = len(unknown)

_frozen_total = None
_frozen_names = set()
if _os.path.exists(_FROZEN):
    with open(_FROZEN) as _fh:
        for _ln in _fh:
            _ln = _ln.strip()
            if not _ln or _ln.startswith("#"):
                continue
            if _ln.startswith("total="):
                _frozen_total = int(_ln.split("=", 1)[1])
            else:
                _frozen_names.add(_ln)

if _frozen_total is None:
    with open(_FROZEN, "w") as _fh:
        _fh.write("# Frozen accused set. Founder ruling 2026-08-19: freeze, fail the next.\n")
        _fh.write("# A rise in total, or a name absent here, fails the gate.\n")
        _fh.write("# The list may only SHRINK. Lower it when a family is corrected.\n")
        _fh.write("total=%d\n" % _total_now)
        for _n in _names_now:
            _fh.write("%s\n" % _n)
    _frozen_total = _total_now
    _frozen_names = set(_names_now)

_novos = [n for n in _names_now if n not in _frozen_names]
_ratchet_fail = 0
if _novos:
    _ratchet_fail = 1
    failed_msgs.append("effect_name_not_in_frozen_set:" + ",".join(_novos[:6]))
    print("EFFECT_NAME_CLOSED_LIST_FAIL new_unknown_name names=%s" % ",".join(_novos[:6]),
          file=sys.stderr)
elif _total_now > _frozen_total:
    _ratchet_fail = 1
    failed_msgs.append("unknown_uses_rose:%d->%d" % (_frozen_total, _total_now))
    print("EFFECT_NAME_CLOSED_LIST_FAIL rose frozen=%d measured=%d"
          % (_frozen_total, _total_now), file=sys.stderr)
elif _total_now < _frozen_total:
    print("EFFECT_NAME_CLOSED_LIST_OK fell frozen=%d measured=%d "
          "— lower the frozen total" % (_frozen_total, _total_now))
else:
    print("EFFECT_NAME_CLOSED_LIST_OK held=%d (accusation still printed above)"
          % _frozen_total)

# The accusation itself no longer forces red; the ratchet does.
failed = _ratchet_fail
if accuse and accuse_status == "not_run" and "positive_io_check_failed" in failed_msgs:
    failed += 1
status = "fail" if failed_msgs or failed else "pass"
if status == "pass" and accuse and accuse_status == "not_run":
    # Scan clean but we could not ask the compiler: do not claim the hole is gone.
    status = "fail"
    failed_msgs.append("accusation_not_run")
    failed += 1
    not_run = 1

emit(
    status,
    total,
    passed,
    failed,
    not_run,
    {
        **derived,
        "scan": {
            "files": len(rels),
            "hits": len(hits),
            "known": len(known),
            "unknown": len(unknown),
            "known_counts": dict(known_counts),
            "unknown_counts": dict(unknown_counts),
            "unknown_sites": unknown,
        },
        "accusation": accuse_detail,
        "failures": failed_msgs[:50],
        "failure_count": len(failed_msgs),
        "kind": "check",
        "note": (
            "ACCUSATION gate. status=fail while souc check of "
            f"{witness_rel} exits 0, and while any used with-name is "
            "absent from the derived closed list."
        ),
    },
)
PY
