#!/usr/bin/env bash
# refresh_lean_seed.sh — executable recipe to resync bin/souc-lean-single-x86_64
#
# WHY THIS EXISTS
# ---------------
# scripts/ci/canonical_compiler_gate.sh FAILS when the committed lean_single ELF
# is not the byte-identical self-reproducing fixed point of
# self-hosted/compiler/lean_single.sio. Three historical commits prove the
# repair works:
#
#   8ef762a99d / a30726e1c9  (#1606)  bootstrap → gen1 → gen2==gen3; ship gen2
#   4581f72345               (#1750)  seed → s1 → s2 (s1==s2); ship s2
#   973b022b1a               (#1768)  seed → g1 → g2 → g3 (g1≠g2, g2==g3); ship g2
#
# Until this script existed, the procedure lived only in those commit messages.
# That folklore cost: open PRs stall on a 2.4 s CI check because nobody outside
# the authors of those commits can satisfy the gate. See
# docs/ops/LEAN_SINGLE_SEED_REFRESH.md and
# docs/audit/CANONICAL_COMPILER_GATE_STRUCTURAL_COST_2026-08-18.md.
#
# WHAT THIS DOES NOT DO BY DEFAULT
# --------------------------------
# Default mode is --print (recipe + cost number). It never rebuilds unless you
# pass --execute AND set SOUNIO_SEED_REFRESH_EXECUTE=1. Rebuild is a founder
# decision: it consumes cluster CPU. Agents must not run --execute unprompted.
#
# USAGE
# -----
#   bash scripts/dev/refresh_lean_seed.sh              # print recipe (default)
#   bash scripts/dev/refresh_lean_seed.sh --print
#   bash scripts/dev/refresh_lean_seed.sh --check      # run verify only (cheap)
#   bash scripts/dev/refresh_lean_seed.sh --stage      # copy inputs to OrangeFS
#   # founder only, after staging:
#   SOUNIO_SEED_REFRESH_EXECUTE=1 bash scripts/dev/refresh_lean_seed.sh --execute --via-slurm
#   SOUNIO_SEED_REFRESH_EXECUTE=1 bash scripts/dev/refresh_lean_seed.sh --execute --local-locked
#
# DO NOT use sbatch. Use scripts/dev/slurm_srun_minimal.sh (srun). See
# docs/ops/SLURM_LAUNCH_REPAIR_2026-08-17.md — sbatch is held for this submitter;
# cluster hardware is up; held jobs are corpses, not capacity.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SRC="self-hosted/compiler/lean_single.sio"
SEED="bin/souc-lean-single-x86_64"
# Alternate bootstrap used by Makefile / verify_lean_seed derive recipe.
# Generation 1 from THIS binary is NOT the fixed point — ship gen2+ only.
BOOTSTRAP_LINUX="bin/souc-linux-x86_64"
GATE="scripts/ci/canonical_compiler_gate.sh"
VERIFY="scripts/ci/verify_lean_seed.sh"
SRUN="scripts/dev/slurm_srun_minimal.sh"
LOCK="scripts/dev/souc-build-lock.sh"

# OrangeFS is visible on compute nodes; /workspace is not.
ORANGEFS_ROOT="${SOUNIO_SEED_ORANGEFS_ROOT:-/orangefs/training/sounio/seed-refresh}"
STAGE_DIR="${SOUNIO_SEED_STAGE_DIR:-}"

MODE="print"
VIA="none"          # none | slurm | local-locked
MAX_GENS="${SOUNIO_SEED_MAX_GENS:-6}"
STACK_KB="${SOUNIO_SEED_STACK_KB:-1048576}"
SLURM_PARTITION="${SOUNIO_SEED_SLURM_PARTITION:-cpu-ops}"
SLURM_TIME="${SOUNIO_SEED_SLURM_TIME:-00:45:00}"

# Cost of unwritten folklore (measured 2026-08-18; re-measure with --cost).
# Sole open PR whose Contracts hard-fail is only the canonical md5 step: #1750.
FOLKLORE_COST_OPEN_PRS_MD5_ONLY=1

usage() {
  sed -n '2,45p' "$0" | sed 's/^# \?//'
  cat <<EOF

Modes:
  --print            Print the recipe and exit 0 (default). No rebuild.
  --check            Run canonical_compiler_gate + verify_lean_seed on this tree.
  --stage            Stage SRC+SEED (+ optional linux bootstrap) onto OrangeFS.
  --execute          Run the derive chain. Requires SOUNIO_SEED_REFRESH_EXECUTE=1.
  --cost             Print the open-PR cost number of unwritten folklore and exit.

Execute placement (with --execute):
  --via-slurm        Stage if needed, then srun via $SRUN (SUPPORTED path).
  --local-locked     Run under $LOCK on this host (pod — last resort).

Environment:
  SOUNIO_SEED_REFRESH_EXECUTE=1   required with --execute
  SOUNIO_SEED_STAGE_DIR=PATH      reuse an existing stage directory
  SOUNIO_SEED_ORANGEFS_ROOT=PATH  default $ORANGEFS_ROOT
  SOUNIO_SEED_MAX_GENS=N          default $MAX_GENS
  SOUNIO_SEED_SLURM_PARTITION=…   default $SLURM_PARTITION
  SOUNIO_SEED_SLURM_TIME=…        default $SLURM_TIME
  SOUNIO_SEED_DDC=1               after install, also run verify DDC leg

Never:
  sbatch                          broken for openvscode-server (held corpses)
  cp … bin/souc                   bin/souc is the Madaros WRAPPER, not the seed
  commit generation 1             only genN where genN == genN+1 is shippable
EOF
}

die() { echo "[refresh-lean-seed] FAIL: $*" >&2; exit 1; }
note() { echo "[refresh-lean-seed] $*"; }

md5_of() { md5sum "$1" | awk '{print $1}'; }

require_inputs() {
  [[ -f "$SRC" ]]  || die "missing $SRC"
  [[ -x "$SEED" ]] || die "missing or non-executable $SEED"
  [[ -x "$GATE" ]] || die "missing $GATE"
  [[ -x "$VERIFY" || -f "$VERIFY" ]] || die "missing $VERIFY"
}

# ── recipe text (the folklore, written) ─────────────────────────────────────
print_recipe() {
  cat <<'EOF'
# =============================================================================
# LEAN_SINGLE SEED REFRESH — executable recipe
# Reconstituted from commits 8ef762a99d / a30726e1c9, 4581f72345, 973b022b1a
# Full prose: docs/ops/LEAN_SINGLE_SEED_REFRESH.md
# Driver:     bash scripts/dev/refresh_lean_seed.sh
# =============================================================================

## 0. Quantified cost (re-measured 2026-08-18)
EOF
  echo "#   Open PRs blocked ONLY by canonical md5 divergence: ${FOLKLORE_COST_OPEN_PRS_MD5_ONLY}"
  cat <<'EOF'
#     specimen: #1750 (Contracts sole hard-fail step = Canonical; all else green)
#   Open PRs blocked by md5 + something else:              0
#   Open PRs that touch lean_single.sio AND pass Canonical: 1 live (#1729, comment-only)
#   Open PRs that touch lean_single.sio total:              5
#   Full table: docs/ops/LEAN_SINGLE_SEED_REFRESH.md §0

## 1. When you need this
#   You edited self-hosted/compiler/lean_single.sio (codegen, not just comments)
#   and Contracts fails:
#     [canonical-compiler] committed md5    = <old>
#     [canonical-compiler] self-compile md5 = <new>
#   Check ~2.4 s. Repair = multi-generation self-compile on srun.

## 2. HARD STOP — M1 fixed point is mandatory (not optional)
#   Let SEED = bin/souc-lean-single-x86_64   (ONLY install path; NEVER bin/souc)
#   Let SRC  = self-hosted/compiler/lean_single.sio
#
#   WRONG:  md5(new_elf) != md5(old_seed)                 ← merely different
#   RIGHT:  exists k: md5(g_k) == md5(g_{k+1})            ← M1 SETTLE  **REQUIRED**
#      AND  md5(C compiling SRC) == md5(C)                ← M2 SELF-REPRO **REQUIRED**
#      AND  two compiles of C agree                       ← M3 DETERMINISM **REQUIRED**
#
#   Without M1 you may ship gen1 (#1606). That is worse than no refresh.
#   Driver will NOT install unless out/SETTLED.md5 exists (M1 recorded).
#   Hand-derive without M1 = off-recipe. STOP. Do not cp. Do not commit.
#   DDC is optional only AFTER M1–M3 are green.

## 3. Placement (founder decision — consumes cluster)
#   ONLY supported launch:
#     bash scripts/dev/slurm_srun_minimal.sh --partition=cpu-ops --time=00:45:00 -- '<cmd>'
#   Positive control: cpuops-t560-proxmox, 32 cores, rc=0.
#   Cluster UP. Held sbatch jobs = corpses. Do NOT invoke sbatch.
#   /workspace invisible on compute → stage to /orangefs first.
#   Pod last resort: scripts/dev/souc-build-lock.sh (eviction history).

## 4. Stage (login/pod — cheap)
#   bash scripts/dev/refresh_lean_seed.sh --stage

## 5. Derive until M1 holds — no install before this
#   Prefer g0 = current committed seed:
#     ulimit -s 1048576
#     g0→g1→g2→… until md5(g_k)==md5(g_{k+1})     # M1 — FAIL CLOSED if missing
#     then: g_k compiling SRC == g_k               # pre-install M2
#     write out/SETTLED.md5 = that md5
#   4581: settled g1==g2.  973b: g1≠g2, settled g2==g3.
#   Alternate souc-linux start: NEVER ship gen1; require gen2==gen3 (a30726).
#
#   SOUNIO_SEED_REFRESH_EXECUTE=1 \
#     bash scripts/dev/refresh_lean_seed.sh --execute --via-slurm

## 6. Install ONLY if SETTLED.md5 present
#     test -f out/SETTLED.md5 || exit 1
#     cp -f out/gK.elf bin/souc-lean-single-x86_64 && chmod +x bin/souc-lean-single-x86_64

## 7. Post-install M2+M3 (still required; DDC optional after)
#     bash scripts/ci/canonical_compiler_gate.sh     # M2
#     bash scripts/ci/verify_lean_seed.sh            # M2+M3
#   Commit message MUST contain:
#     settle: gK==g{K+1} md5=<H>     # M1 — no line, no merge
#     canonical: committed==self-compile==<H>
#     placement: srun/cpu-ops
#   NOT enough: "md5 changed" / "newer binary" / "make build ran"

## 8. Commit + post-merge
#     git add bin/souc-lean-single-x86_64 && git commit …
#   After merging main into a lean_single-touching branch: re-run from §1.
#   (#1750 lost a correct seed when main overwrote the ELF.)

## 9. Wall-clock estimate (no timer in the three commits; extrapolated)
#   GHA 1× self-compile ~2.4 s (#1750). 2–4 gens → seconds of pure compile.
#   Plan: 5–15 min seed-only on idle srun; hard budget 45 min.
#   #1606 "~2 hours" was full corpus parity — NOT the seed derive.
#   Details: docs/ops/LEAN_SINGLE_SEED_REFRESH.md §4

## 10. Three commits (do not re-discover)
#   4581   two-pass settle when g1 already emits new codegen
#   973b   bootstrap-surface codegen needs g0…g3; g1≠g2 expected
#   a30726 shipping gen1 from souc-linux is WRONG
EOF
}

print_cost() {
  cat <<EOF
folklore_cost_open_prs_blocked_only_by_canonical_md5=${FOLKLORE_COST_OPEN_PRS_MD5_ONLY}
# census 2026-08-18 (open=51; touch lean_single.sio = 5):
#   A  md5-only block                              = 1  (#1750)
#   B  md5 + other                                 = 0
#   C  touch lean_single + live Canonical PASS     = 1  (#1729 comment-only; no seed ship)
#   X  touch lean_single; no live Canonical step   = 3  (#1034 pre-gate+ships seed,
#                                                       #1527 pre-gate, #1758 docs-registry)
# #1729 is NOT a gate hole and NOT a rebuild author — comments keep the FP.
# #1034 is the in-era rebuild author (ships ELF) but pre-gate, so not live C.
# specimen #1750 also merge CONFLICTING/DIRTY — resync necessary, not sufficient.
# docs/ops/LEAN_SINGLE_SEED_REFRESH.md §0
EOF
}

# ── cheap verify ────────────────────────────────────────────────────────────
run_check() {
  require_inputs
  note "running $GATE"
  bash "$GATE"
  note "running $VERIFY"
  bash "$VERIFY"
  if [[ "${SOUNIO_SEED_DDC:-0}" == "1" ]]; then
    note "running $VERIFY with DDC"
    SOUNIO_SEED_DDC=1 bash "$VERIFY"
  fi
  note "CHECK PASS"
}

# ── stage to OrangeFS ───────────────────────────────────────────────────────
run_stage() {
  require_inputs
  if [[ -z "$STAGE_DIR" ]]; then
    local ts
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    STAGE_DIR="${ORANGEFS_ROOT}/${ts}"
  fi
  if [[ ! -d "$(dirname "$STAGE_DIR")" ]]; then
    die "OrangeFS parent missing: $(dirname "$STAGE_DIR")
  Compute staging requires /orangefs. If you are not on a host that mounts it,
  stop and run from the workspace login that sees OrangeFS, or use --local-locked
  (founder-only, pod risk)."
  fi
  mkdir -p "$STAGE_DIR"/{bin,self-hosted/compiler,out,scripts/ci}
  cp -f "$SRC" "$STAGE_DIR/$SRC"
  cp -f "$SEED" "$STAGE_DIR/$SEED"
  chmod +x "$STAGE_DIR/$SEED"
  if [[ -x "$BOOTSTRAP_LINUX" ]]; then
    cp -f "$BOOTSTRAP_LINUX" "$STAGE_DIR/$BOOTSTRAP_LINUX"
    chmod +x "$STAGE_DIR/$BOOTSTRAP_LINUX"
  fi
  cp -f "$GATE" "$STAGE_DIR/$GATE"
  cp -f "$VERIFY" "$STAGE_DIR/$VERIFY"
  # record provenance
  {
    echo "staged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo=$ROOT_DIR"
    echo "branch=$(git branch --show-current 2>/dev/null || echo detached)"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "src_md5=$(md5_of "$SRC")"
    echo "seed_md5=$(md5_of "$SEED")"
  } >"$STAGE_DIR/STAGE.txt"
  note "STAGE_DIR=$STAGE_DIR"
  note "contents:"
  find "$STAGE_DIR" -type f | sort | sed 's/^/  /'
  echo "$STAGE_DIR"
}

# ── derive loop (the actual folklore body) ──────────────────────────────────
# Runs inside WORK_ROOT which must contain SRC and a starting ELF at SEED path.
derive_fixed_point() {
  local work_root="$1"
  local start_elf="$2"   # absolute path to g0
  local out_dir="$work_root/out"
  mkdir -p "$out_dir"
  ulimit -s "$STACK_KB" 2>/dev/null || true

  local src="$work_root/$SRC"
  [[ -f "$src" ]] || die "derive: missing $src"
  [[ -x "$start_elf" ]] || die "derive: missing executable start $start_elf"

  local prev="$start_elf"
  local prev_md5
  prev_md5="$(md5_of "$prev")"
  note "g0 md5=$prev_md5  path=$prev"

  local i=1
  local cand=""
  local cand_md5=""
  while [[ "$i" -le "$MAX_GENS" ]]; do
    cand="$out_dir/g${i}.elf"
    note "compiling g${i}: $(basename "$prev") → $(basename "$cand")"
    if ! "$prev" "$src" "$cand" >"$out_dir/g${i}.log" 2>&1; then
      tail -20 "$out_dir/g${i}.log" >&2 || true
      die "g${i} compile failed (see $out_dir/g${i}.log)"
    fi
    chmod +x "$cand"
    cand_md5="$(md5_of "$cand")"
    note "g${i} md5=$cand_md5  bytes=$(wc -c <"$cand")"
    if [[ "$cand_md5" == "$prev_md5" && "$i" -ge 2 ]]; then
      # prev was already a fixed point; ship prev (which equals cand)
      note "SETTLED at g$((i-1))==g${i}  md5=$cand_md5"
      echo "$cand"
      return 0
    fi
    # special-case i==1: g0→g1 may differ; continue
    prev="$cand"
    prev_md5="$cand_md5"
    i=$((i + 1))
  done

  # After loop, check last two generations if we produced >=2
  if [[ -x "$out_dir/g$((MAX_GENS-1)).elf" && -x "$out_dir/g${MAX_GENS}.elf" ]]; then
    local a b
    a="$(md5_of "$out_dir/g$((MAX_GENS-1)).elf")"
    b="$(md5_of "$out_dir/g${MAX_GENS}.elf")"
    if [[ "$a" == "$b" ]]; then
      note "SETTLED at g$((MAX_GENS-1))==g${MAX_GENS} md5=$a"
      echo "$out_dir/g${MAX_GENS}.elf"
      return 0
    fi
  fi
  die "did not reach fixed point within MAX_GENS=$MAX_GENS (see $out_dir)"
}

# Refined settle: after each new gen i>=2, compare g{i-1} vs g{i}
derive_fixed_point_v2() {
  local work_root="$1"
  local start_elf="$2"
  local out_dir="$work_root/out"
  mkdir -p "$out_dir"
  ulimit -s "$STACK_KB" 2>/dev/null || true

  local src="$work_root/$SRC"
  [[ -f "$src" && -x "$start_elf" ]] || die "derive_v2: bad inputs"

  # g0 is start; produce g1.. until g{n}==g{n+1}
  local -a md5s=()
  local -a paths=()
  paths[0]="$start_elf"
  md5s[0]="$(md5_of "$start_elf")"
  note "g0 md5=${md5s[0]}  path=${paths[0]}"

  local i=1
  while [[ "$i" -le "$MAX_GENS" ]]; do
    local cand="$out_dir/g${i}.elf"
    local prev="${paths[$((i-1))]}"
    note "compiling g${i} from g$((i-1))"
    if ! "$prev" "$src" "$cand" >"$out_dir/g${i}.log" 2>&1; then
      tail -30 "$out_dir/g${i}.log" >&2 || true
      die "g${i} compile failed"
    fi
    chmod +x "$cand"
    paths[$i]="$cand"
    md5s[$i]="$(md5_of "$cand")"
    note "g${i} md5=${md5s[$i]}  bytes=$(wc -c <"$cand")"
    if [[ "$i" -ge 2 && "${md5s[$i]}" == "${md5s[$((i-1))]}" ]]; then
      note "SETTLED: g$((i-1)) == g${i}  md5=${md5s[$i]}"
      # Also prove the gate property: settled ELF self-reproduces once more
      local repro="$out_dir/repro.elf"
      if ! "$cand" "$src" "$repro" >"$out_dir/repro.log" 2>&1; then
        tail -20 "$out_dir/repro.log" >&2 || true
        die "settled candidate could not self-compile"
      fi
      chmod +x "$repro"
      local repro_md5
      repro_md5="$(md5_of "$repro")"
      [[ "$repro_md5" == "${md5s[$i]}" ]] \
        || die "settled md5 ${md5s[$i]} != self-repro $repro_md5 (not a fixed point)"
      note "SELF-REPRO ok md5=$repro_md5"
      printf '%s\n' "$cand"
      return 0
    fi
    i=$((i + 1))
  done
  die "no fixed point within MAX_GENS=$MAX_GENS — dump: ${md5s[*]}"
}

install_seed() {
  local cand="$1"
  local dest="$2"
  local settled_md5_file="${3:-}"
  [[ -x "$cand" ]] || die "install: not executable: $cand"
  # M1 hard stop: refuse install without a recorded settle md5
  if [[ -n "$settled_md5_file" ]]; then
    [[ -f "$settled_md5_file" ]] \
      || die "M1 missing: no $settled_md5_file — refuse install (would ship unsettled ELF)"
    local want got
    want="$(tr -d '[:space:]' <"$settled_md5_file")"
    got="$(md5_of "$cand")"
    [[ "$want" == "$got" ]] \
      || die "M1 mismatch: SETTLED.md5=$want but candidate md5=$got — refuse install"
    note "M1 ok settle_md5=$want"
  fi
  cp -f "$cand" "$dest"
  chmod +x "$dest"
  note "installed $(md5_of "$dest") → $dest"
}

run_execute() {
  [[ "${SOUNIO_SEED_REFRESH_EXECUTE:-}" == "1" ]] \
    || die "refusing --execute without SOUNIO_SEED_REFRESH_EXECUTE=1
  Rebuild consumes cluster. Founder sets the env var deliberately.
  Print the recipe with: bash scripts/dev/refresh_lean_seed.sh --print"

  require_inputs
  local work_root=""
  local start_elf=""

  case "$VIA" in
    slurm)
      [[ -x "$SRUN" || -f "$SRUN" ]] || die "missing $SRUN"
      if [[ -z "$STAGE_DIR" ]]; then
        STAGE_DIR="$(run_stage | tail -1)"
      fi
      [[ -d "$STAGE_DIR" ]] || die "stage dir missing: $STAGE_DIR"
      work_root="$STAGE_DIR"
      start_elf="$STAGE_DIR/$SEED"
      note "launching derive on Slurm via $SRUN (partition=$SLURM_PARTITION)"
      # shellcheck disable=SC2016
      bash "$SRUN" --partition="$SLURM_PARTITION" --time="$SLURM_TIME" -- "
        set -euo pipefail
        ulimit -s $STACK_KB || true
        cd '$STAGE_DIR'
        # inline minimal derive so the node needs no /workspace
        SRC='$SRC'
        SEED='$SEED'
        OUT=out
        mkdir -p \"\$OUT\"
        prev=\"\$SEED\"
        prev_md5=\$(md5sum \"\$prev\" | awk '{print \$1}')
        echo \"[slurm-derive] g0 md5=\$prev_md5\"
        i=1
        while [ \"\$i\" -le $MAX_GENS ]; do
          cand=\"\$OUT/g\${i}.elf\"
          echo \"[slurm-derive] compiling g\${i}\"
          \"\$prev\" \"\$SRC\" \"\$cand\" >\"\$OUT/g\${i}.log\" 2>&1 || {
            tail -30 \"\$OUT/g\${i}.log\" >&2; exit 1
          }
          chmod +x \"\$cand\"
          cand_md5=\$(md5sum \"\$cand\" | awk '{print \$1}')
          echo \"[slurm-derive] g\${i} md5=\$cand_md5\"
          if [ \"\$i\" -ge 2 ] && [ \"\$cand_md5\" = \"\$prev_md5\" ]; then
            # self-repro check
            \"\$cand\" \"\$SRC\" \"\$OUT/repro.elf\" >\"\$OUT/repro.log\" 2>&1
            chmod +x \"\$OUT/repro.elf\"
            repro_md5=\$(md5sum \"\$OUT/repro.elf\" | awk '{print \$1}')
            [ \"\$repro_md5\" = \"\$cand_md5\" ] || {
              echo \"[slurm-derive] FAIL self-repro \$repro_md5 != \$cand_md5\" >&2
              exit 1
            }
            echo \"\$cand_md5\" >\"\$OUT/SETTLED.md5\"
            echo \"\$cand\" >\"\$OUT/SETTLED.path\"
            cp -f \"\$cand\" \"\$OUT/seed.elf\"
            chmod +x \"\$OUT/seed.elf\"
            echo \"[slurm-derive] SETTLED g\$((i-1))==g\${i} md5=\$cand_md5\"
            exit 0
          fi
          prev=\"\$cand\"
          prev_md5=\"\$cand_md5\"
          i=\$((i + 1))
        done
        echo \"[slurm-derive] FAIL no fixed point in $MAX_GENS gens\" >&2
        exit 1
      "
      [[ -f "$STAGE_DIR/out/SETTLED.md5" ]] \
        || die "M1 FAIL: slurm derive produced no SETTLED.md5 — refuse install"
      local settled_md5
      settled_md5="$(tr -d '[:space:]' <"$STAGE_DIR/out/SETTLED.md5")"
      note "slurm settled md5=$settled_md5 (M1)"
      install_seed "$STAGE_DIR/out/seed.elf" "$ROOT_DIR/$SEED" "$STAGE_DIR/out/SETTLED.md5"
      ;;
    local-locked)
      note "LOCAL-LOCKED path — holds souc-build-lock; can stress the pod"
      work_root="$ROOT_DIR"
      start_elf="$ROOT_DIR/$SEED"
      local cand settled_file
      settled_file="$ROOT_DIR/out/SETTLED.md5"
      mkdir -p "$ROOT_DIR/out"
      cand="$(bash "$LOCK" bash -c "
        set -euo pipefail
        $(declare -f note die md5_of derive_fixed_point_v2)
        ROOT_DIR='$ROOT_DIR'
        SRC='$SRC'
        MAX_GENS='$MAX_GENS'
        STACK_KB='$STACK_KB'
        c=\$(derive_fixed_point_v2 '$work_root' '$start_elf')
        md5sum \"\$c\" | awk '{print \$1}' > '$settled_file'
        printf '%s\n' \"\$c\"
      ")"
      [[ -f "$settled_file" ]] || die "M1 FAIL: local derive wrote no SETTLED.md5"
      install_seed "$cand" "$ROOT_DIR/$SEED" "$settled_file"
      ;;
    *)
      die "--execute requires --via-slurm or --local-locked"
      ;;
  esac

  note "post-install verification"
  (cd "$ROOT_DIR" && bash "$GATE")
  (cd "$ROOT_DIR" && bash "$VERIFY")
  note "EXECUTE PASS — commit $SEED when ready"
  note "md5=$(md5_of "$ROOT_DIR/$SEED")"
}

# ── argv ────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --print) MODE=print; shift ;;
    --check) MODE=check; shift ;;
    --stage) MODE=stage; shift ;;
    --execute) MODE=execute; shift ;;
    --cost) MODE=cost; shift ;;
    --via-slurm) VIA=slurm; shift ;;
    --local-locked) VIA=local-locked; shift ;;
    --stage-dir=*) STAGE_DIR="${1#*=}"; shift ;;
    --partition=*) SLURM_PARTITION="${1#*=}"; shift ;;
    --time=*) SLURM_TIME="${1#*=}"; shift ;;
    *) die "unknown arg: $1 (see --help)" ;;
  esac
done

case "$MODE" in
  print)  print_recipe; print_cost ;;
  cost)   print_cost ;;
  check)  run_check ;;
  stage)  run_stage >/dev/null; note "STAGE_DIR=$STAGE_DIR" ;;
  execute) run_execute ;;
  *) die "bad mode $MODE" ;;
esac
