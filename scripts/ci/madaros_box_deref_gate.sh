#!/usr/bin/env bash
# Guards that Box::new actually allocates and that a read through the Box returns the stored
# value. Both halves matter and each has a distinct failure mode that was live in this tree:
#
#   1. Box::new compiled to a call to a bodyless stub returning 0 (the call-site Box detection
#      tested the MANGLED callee name "Box_new" against a 3-byte "Box"), so lower_box_new was
#      dead code. The 0 then sat below the handle/raw-pointer threshold, resolved via handle
#      table slot 0 — never allocated — to object_base 0, and any read faulted at 0x20.
#   2. The pointer-class lookup that decides whether to emit the deref was by FIELD NAME across
#      all structs, so two structs sharing a field name with differing pointer-ness silently
#      skipped the deref and returned a garbage value with exit 0 — a wrong value rather than a
#      crash, which no exit-code-only check would catch.
#
# So this gate asserts VALUES, not just exit status, and it reads a field at a NON-ZERO offset.
# At offset 0 the Box deref and the field read emit the identical field_get(x, 0), which makes
# the two hypotheses indistinguishable — four fix attempts were invalidated by testing only
# offset 0.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[madaros-box-deref] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[madaros-box-deref] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

OUT_DIR="${SOUNIO_MADAROS_BOX_DEREF_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-box-deref.XXXXXX)}"
mkdir -p "$OUT_DIR"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# The fix this gate guards lives in self-hosted/ir/lower.sio, i.e. in MADAROS SOURCE. Resolving
# the compiler the usual way (scripts/lib/resolve_souc.sh) would test the CHECKED-IN Madaros ELF,
# which predates the fix and is not rebuilt by a checkout — the gate would then report a failure
# that says nothing about the working tree, and a later regression in lower.sio would not move it
# at all. So the subject must be a Madaros built from the current source.
#
#   SOUNIO_MADAROS_BOX_DEREF_GATE_BIN=<elf>  use an already-built one — this is what CI does, via
#                                            madaros_current_source_f64_lowering_gate.sh, so one
#                                            ~10-minute bundle build is shared across gates
#   unset                                    build one here, through the global build lock
SOUC_BIN="${SOUNIO_MADAROS_BOX_DEREF_GATE_BIN:-}"
if [[ -z "$SOUC_BIN" ]]; then
  SOUC_BIN="$OUT_DIR/madaros-from-source.elf"
  printf '[madaros-box-deref] no SOUNIO_MADAROS_BOX_DEREF_GATE_BIN; building Madaros from source\n'
  if ! "$ROOT_DIR/scripts/dev/souc-build-lock.sh" \
        bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$SOUC_BIN" \
        >"$OUT_DIR/build.log" 2>&1; then
    echo "[madaros-box-deref] FAIL: could not build Madaros from source" >&2
    tail -n 30 "$OUT_DIR/build.log" >&2 || true
    exit 1
  fi
fi
if [[ ! -x "$SOUC_BIN" ]]; then
  chmod +x "$SOUC_BIN" 2>/dev/null || true
fi
if [[ ! -x "$SOUC_BIN" ]]; then
  echo "[madaros-box-deref] FAIL: compiler not executable: $SOUC_BIN" >&2
  exit 1
fi

printf '[madaros-box-deref] souc=%s\n' "$SOUC_BIN"
printf '[madaros-box-deref] out=%s\n' "$OUT_DIR"

fail=0

run_case() {
  local label="$1" src="$2" expected="$3"
  local bin="$OUT_DIR/$label" log="$OUT_DIR/$label.log" out="$OUT_DIR/$label.stdout"
  if ! "$SOUC_BIN" "$src" -o "$bin" >"$log" 2>&1; then
    echo "[madaros-box-deref] FAIL($label): compile failed" >&2
    tail -n 20 "$log" >&2 || true
    fail=1
    return
  fi
  chmod +x "$bin"
  local rc=0
  "$bin" >"$out" 2>&1 || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "[madaros-box-deref] FAIL($label): exit $rc (139 = the pre-fix SIGSEGV at 0x20)" >&2
    tail -n 10 "$out" >&2 || true
    fail=1
    return
  fi
  if ! grep -qE "$expected" "$out"; then
    echo "[madaros-box-deref] FAIL($label): expected /$expected/ in output" >&2
    echo "  a wrong VALUE with exit 0 is the silent-miscompile mode this gate exists for" >&2
    sed 's/^/    /' "$out" >&2 || true
    fail=1
    return
  fi
  printf '[madaros-box-deref] PASS(%s)\n' "$label"
}

# Case 1: offset 0 AND offset 1 through a Box. Offset 1 is the load-bearing one.
cat >"$OUT_DIR/box_offsets.sio" <<'SIO'
struct Inner {
    a: i64,
    b: i64,
}

fn empty_inner() -> Inner {
    Inner { a: 7, b: 8 }
}

fn main() with IO, Mut, Panic, Div {
    let bx = Box::new(empty_inner())
    print_int((*bx).a)
    print_int((*bx).b)
    println(" BOXOFFSETS")
}
SIO
run_case box_offsets "$OUT_DIR/box_offsets.sio" '^78 BOXOFFSETS'

# Case 2: two structs sharing the field name `inner`, one inline and one Box. This is the shape
# that a by-name pointer-class lookup gets wrong, returning garbage with exit 0.
cat >"$OUT_DIR/box_homonym.sio" <<'SIO'
struct Inner {
    a: i64,
    b: i64,
}

fn empty_inner() -> Inner {
    Inner { a: 7, b: 8 }
}

struct HolderInline {
    inner: Inner,
    n: i64,
}

struct HolderBox {
    inner: Box<Inner>,
    n: i64,
}

fn main() with IO, Mut, Panic, Div {
    var hi: HolderInline = HolderInline { inner: empty_inner(), n: 1 }
    var hb: HolderBox = HolderBox { inner: Box::new(empty_inner()), n: 2 }
    print_int(hi.inner.a)
    print_int(hb.inner.a)
    print_int(hb.inner.b)
    println(" BOXHOMONYM")
}
SIO
run_case box_homonym "$OUT_DIR/box_homonym.sio" '^778 BOXHOMONYM'

# Case 3: the in-tree witness, a //@ run-pass that could never pass before this was fixed and
# was wired to no gate.
run_case box_boundary_witness \
  "$ROOT_DIR/tests/native-v2/box_return_deref_boundary_witness.sio" \
  'box return deref boundary stable'

if [[ "$fail" -ne 0 ]]; then
  echo "[madaros-box-deref] GATE FAILED" >&2
  exit 1
fi

echo "[madaros-box-deref] PASS: Box::new allocates and reads through the Box return stored values"
