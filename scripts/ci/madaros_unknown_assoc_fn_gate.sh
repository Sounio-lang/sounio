#!/usr/bin/env bash
# W044 — associated-function call on a type that does not exist.
#
# Madaros rejects an undefined plain call (`nao_existe(3)` -> error[E137]) but accepted ANY
# path-form `Tipo::metodo(...)` whose target resolved to nothing, silently, evaluating it to 0.
# That is the mechanism that turned a missing `Box::new` lowering case into a call to a bodyless
# stub returning 0 rather than a diagnostic: the checker never demanded that a call target exist,
# so `Coisa::inexistente(9)` compiled clean and `Rc::new(7)` + `*v` SIGSEGV'd.
#
# W044 was HALF-landed before this gate: self-hosted/check/check.sio's print_warning_message
# already carried the `code == 44` text, but nothing anywhere emitted code 44 — the diagnostic
# was documented and dead. This adds the detection and pins it.
#
# The gate pins BOTH directions, because the risk here is not a missing warning — it is a warning
# that fires on legitimate code. The suppression rules under test:
#   - first segment names a known struct or enum  -> silent (preserves the documented
#     multi-module residual for imported associated forms)
#   - a free fn matches the last segment          -> silent (module-qualified calls such as
#     gpu::launch, wmma::mma_sync)
#   - Box::new                                    -> handled earlier, never reaches the check
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[madaros-unknown-assoc-fn] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[madaros-unknown-assoc-fn] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

OUT_DIR="${SOUNIO_MADAROS_UNKNOWN_ASSOC_FN_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-unknown-assoc.XXXXXX)}"
mkdir -p "$OUT_DIR"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# Subject must be built from current source: the check lives in self-hosted/check/check.sio, so a
# run against the checked-in ELF would say nothing about the working tree. In CI this gate does
# NOT build: it is wired into the madaros-current-source-deref-f64 job, which builds Madaros once
# in madaros_current_source_f64_lowering_gate.sh (KEEP=1) and hands the ELF to every later step
# via *_GATE_BIN. The build below is the local/standalone path only.
SOUC_BIN="${SOUNIO_MADAROS_UNKNOWN_ASSOC_FN_GATE_BIN:-}"
if [[ -z "$SOUC_BIN" ]]; then
  SOUC_BIN="$OUT_DIR/madaros-from-source.elf"
  printf '[madaros-unknown-assoc-fn] no *_GATE_BIN; building Madaros from source\n'
  # NOT wrapped in scripts/dev/souc-build-lock.sh: build_modular_madaros.sh takes that lock
  # itself (lines 101 and 115) and flock(1) is not reentrant across the extra process, so
  # wrapping it deadlocks — the outer holds the lock while the inner waits for it forever.
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$SOUC_BIN" \
        >"$OUT_DIR/build.log" 2>&1; then
    echo "[madaros-unknown-assoc-fn] FAIL: could not build Madaros from source" >&2
    tail -n 30 "$OUT_DIR/build.log" >&2 || true
    exit 1
  fi
fi
chmod +x "$SOUC_BIN" 2>/dev/null || true
[[ -x "$SOUC_BIN" ]] || { echo "[madaros-unknown-assoc-fn] FAIL: not executable: $SOUC_BIN" >&2; exit 1; }

# Madaros needs a large stack to RUN AT ALL: the bundle build reports 150 "stack frame too large"
# warnings with frames up to ~31 MB, so compiling even a three-line program overflows the 16384 KiB
# soft limit GitHub runners default to. Measured: at 16384 every case here fails with
# "compile failed"; raised, all pass. This is the same requirement, and the same mechanism, as
# madaros_imported_call_arity_13_gate.sh — which is why that gate had to raise it first, and why
# these three had never actually run in CI until it stopped failing ahead of them.
stack_kb="${SOUNIO_MADAROS_UNKNOWN_ASSOC_FN_STACK_KB:-524288}"
[[ "$stack_kb" =~ ^[1-9][0-9]*$ && ${#stack_kb} -le 9 ]] || { echo "[madaros-unknown-assoc-fn] FAIL: invalid stack size: $stack_kb" >&2; exit 1; }
stack_before="$(ulimit -S -s 2>/dev/null)" || { echo "[madaros-unknown-assoc-fn] FAIL: soft stack limit unavailable" >&2; exit 1; }
if [[ "$stack_before" != "unlimited" ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || { echo "[madaros-unknown-assoc-fn] FAIL: could not raise soft stack limit to ${stack_kb} KiB" >&2; exit 1; }
fi
stack_after="$(ulimit -S -s 2>/dev/null)"
if [[ "$stack_after" != "unlimited" ]] && ((stack_after < stack_kb)); then
  echo "[madaros-unknown-assoc-fn] FAIL: soft stack limit remained below ${stack_kb} KiB: $stack_after" >&2
  exit 1
fi
printf '[madaros-unknown-assoc-fn] stack_kb before=%s after=%s requested=%s\n' "$stack_before" "$stack_after" "$stack_kb"

printf '[madaros-unknown-assoc-fn] souc=%s\n' "$SOUC_BIN"

fail=0

# want=warn expects at least one W044; want=quiet expects none.
#
# ANTI-VACUITY. A `want=quiet` case passes on w044=0, and a compiler that crashed, failed to
# build the program, or bailed early ALSO yields w044=0. Three of the five cases here are quiet,
# so without a liveness assertion this gate would report PASS against a broken subject. Every
# case therefore additionally requires that `--check` exited 0 and emitted no `error[`. The two
# `warn` cases carry that requirement too: W044 is a WARNING, so a well-formed subject must
# accept both programs -- if `Coisa::inexistente(9)` ever starts erroring, the gate should tell
# us that the diagnostic changed class rather than quietly still counting a match.
run_case() {
  local label="$1" want="$2" src="$3"
  local log="$OUT_DIR/$label.log"
  local rc=0
  "$SOUC_BIN" --check "$src" >"$log" 2>&1 || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "[madaros-unknown-assoc-fn] FAIL($label): --check exited $rc; the case never typechecked, so its W044 count is meaningless" >&2
    tail -n 15 "$log" >&2 || true
    fail=1
    return
  fi
  if grep -q 'error\[' "$log"; then
    echo "[madaros-unknown-assoc-fn] FAIL($label): --check reported an error; the case never typechecked, so its W044 count is meaningless" >&2
    grep -n 'error\[' "$log" | head -n 5 >&2 || true
    fail=1
    return
  fi
  local n
  n="$(grep -c 'W044' "$log" || true)"
  if [[ "$want" == "warn" && "$n" -eq 0 ]]; then
    echo "[madaros-unknown-assoc-fn] FAIL($label): expected W044, got none" >&2
    tail -n 8 "$log" >&2 || true
    fail=1
    return
  fi
  if [[ "$want" == "quiet" && "$n" -ne 0 ]]; then
    echo "[madaros-unknown-assoc-fn] FAIL($label): W044 fired $n time(s) on legitimate code" >&2
    grep -n 'W044' "$log" | head -n 5 >&2 || true
    fail=1
    return
  fi
  printf '[madaros-unknown-assoc-fn] PASS(%s) want=%s w044=%s\n' "$label" "$want" "$n"
}

# Positive: an invented receiver. Nothing named Coisa exists anywhere.
cat >"$OUT_DIR/invented.sio" <<'SIO'
fn main() with IO, Mut, Panic, Div {
    let z = Coisa::inexistente(9)
    println(" INVENTED")
}
SIO
run_case invented warn "$OUT_DIR/invented.sio"

# Positive: the shape that is live in stdlib/epistemic — Vec::new() into a struct field, while no
# Vec type exists in the tree. This already evaluates to 0 today; the warning makes it visible.
cat >"$OUT_DIR/vec_field.sio" <<'SIO'
struct Prov {
    source: string,
    steps: i64,
}

fn main() with IO, Mut, Panic, Div {
    let p = Prov { source: "constant", steps: Vec::new() }
    print_int(p.steps)
    println(" VECFIELD")
}
SIO
run_case vec_field warn "$OUT_DIR/vec_field.sio"

# Negative: Box::new is a real builtin and returns before the check. A warning here would fire on
# 817 in-tree call sites.
cat >"$OUT_DIR/box_quiet.sio" <<'SIO'
struct Inner {
    a: i64,
    b: i64,
}

fn empty_inner() -> Inner {
    Inner { a: 7, b: 8 }
}

fn main() with IO, Mut, Panic, Div {
    let bx = Box::new(empty_inner())
    print_int((*bx).b)
    println(" BOXQUIET")
}
SIO
run_case box_quiet quiet "$OUT_DIR/box_quiet.sio"

# Negative: an associated function on a type that DOES exist, called through the path form.
cat >"$OUT_DIR/known_type.sio" <<'SIO'
struct Ponto {
    x: i64,
}

impl Ponto {
    fn origem() -> Ponto {
        Ponto { x: 0 }
    }
}

fn main() with IO, Mut, Panic, Div {
    let p = Ponto::origem()
    print_int(p.x)
    println(" KNOWNTYPE")
}
SIO
run_case known_type quiet "$OUT_DIR/known_type.sio"

# Negative: an ordinary multi-module program with an imported free function.
mkdir -p "$OUT_DIR/mm"
cat >"$OUT_DIR/mm/helper.sio" <<'SIO'
pub fn twice(x: i64) -> i64 {
    x * 2
}
SIO
cat >"$OUT_DIR/mm/mmmain.sio" <<'SIO'
use helper::{twice}

fn main() with IO, Mut, Panic, Div {
    print_int(twice(21))
    print_int(str_len("abcd"))
    print_f64(2.5)
    println(" MULTIMOD")
}
SIO
run_case multimodule quiet "$OUT_DIR/mm/mmmain.sio"

if [[ "$fail" -ne 0 ]]; then
  echo "[madaros-unknown-assoc-fn] GATE FAILED" >&2
  exit 1
fi

echo "[madaros-unknown-assoc-fn] PASS: unknown receivers warn; Box::new, known types and imported free fns stay quiet"
