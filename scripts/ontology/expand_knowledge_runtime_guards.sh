#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ontology/expand_knowledge_runtime_guards.sh [--cache-dir DIR] [--manifest FILE] <input.sio> <output.sio>

Pre-native bridge for dynamic Knowledge<T where {...}> runtime guards.

The supported pre-native slice detects numeric proof contexts:

  Knowledge<Patient where { age >= 18 }>
  Knowledge<Patient where { age >= 18, glucose >= 126 }>
  Knowledge<Range where { score <= 100 }>
  Knowledge<Assay where { repeats == 3 }>
  Knowledge<Dose where { amount >= 500 <mg> }>

and rewrites:

  value: Patient { ... }

to:

  value: __sounio_knowledge_guard_Patient_age_ge_18(Patient { ... })

while injecting a helper that asserts the constraint at runtime. This is an
executable bridge toward backend guard lowering, not native backend insertion.
After inserting the guard, the expander lowers the proof-context type to plain
`Knowledge<T>` so the existing backend can execute the guarded program.

When --cache-dir is supplied, the fully expanded source is cached as a
deterministic text .guardcache keyed by the input bytes and this expander's
source bytes. This is still a pre-native cache, not backend guard metadata.

When --manifest is supplied, the expander writes a small deterministic TSV
audit manifest listing the guard diagnostics preserved in the expanded source.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/ontology/expand_knowledge_runtime_guards.sh"

cache_dir=""
manifest_path=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir)
      if [[ $# -lt 2 ]]; then
        echo "error: --cache-dir requires a directory" >&2
        exit 2
      fi
      cache_dir="$2"
      shift 2
      ;;
    --manifest)
      if [[ $# -lt 2 ]]; then
        echo "error: --manifest requires a file path" >&2
        exit 2
      fi
      manifest_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "error: unsupported option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

INPUT="$1"
OUTPUT="$2"

case "$INPUT" in
  /*) INPUT_ABS="$INPUT" ;;
  *) INPUT_ABS="$ROOT_DIR/$INPUT" ;;
esac
case "$OUTPUT" in
  /*) OUTPUT_ABS="$OUTPUT" ;;
  *) OUTPUT_ABS="$ROOT_DIR/$OUTPUT" ;;
esac
if [[ -n "$manifest_path" ]]; then
  case "$manifest_path" in
    /*) MANIFEST_ABS="$manifest_path" ;;
    *) MANIFEST_ABS="$ROOT_DIR/$manifest_path" ;;
  esac
else
  MANIFEST_ABS=""
fi

if [[ ! -f "$INPUT_ABS" ]]; then
  echo "error: input not found: $INPUT" >&2
  exit 2
fi

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

write_guard_manifest() {
  local expanded_file="$1"
  local manifest_file="$2"
  if [[ -z "$manifest_file" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$manifest_file")"
  local manifest_input_hash="${input_hash:-}"
  local manifest_expander_hash="${expander_hash:-}"
  if [[ -z "$manifest_input_hash" ]]; then
    manifest_input_hash="$(hash_file "$INPUT_ABS")"
  fi
  if [[ -z "$manifest_expander_hash" ]]; then
    manifest_expander_hash="$(hash_file "$SCRIPT_PATH")"
  fi
  {
    printf '# Sounio Knowledge runtime guard diagnostics manifest v1\n'
    printf '# input: %s\n' "$INPUT"
    printf '# output: %s\n' "$OUTPUT"
    printf '# input_sha256: %s\n' "$manifest_input_hash"
    printf '# expander_sha256: %s\n' "$manifest_expander_hash"
    printf 'type\tfield\top\tthreshold\tunit\tconstraint\n'
    while IFS= read -r guard_constraint; do
      if [[ "$guard_constraint" =~ ^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*(\>\=|\>|\<\=|\<|\=\=)[[:space:]]*([^[:space:]]+)([[:space:]]*\<([A-Za-z_][A-Za-z0-9_]*)\>)?$ ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" "${BASH_REMATCH[4]}" "${BASH_REMATCH[6]:-}" "$guard_constraint"
      else
        printf 'unknown\tunknown\tunknown\tunknown\t\t%s\n' "$guard_constraint"
      fi
    done < <(sed -n 's/.*Knowledge runtime guard diagnostic: //p' "$expanded_file")
  } >"$manifest_file"
}

if [[ -n "$cache_dir" ]]; then
  case "$cache_dir" in
    /*) CACHE_DIR_ABS="$cache_dir" ;;
    *) CACHE_DIR_ABS="$ROOT_DIR/$cache_dir" ;;
  esac
  mkdir -p "$CACHE_DIR_ABS"
  input_hash="$(hash_file "$INPUT_ABS")"
  expander_hash="$(hash_file "$SCRIPT_PATH")"
  input_base="$(basename "$INPUT_ABS" .sio)"
  cache_file="$CACHE_DIR_ABS/${input_base}.${input_hash:0:16}.${expander_hash:0:16}.guardcache"
  if [[ -f "$cache_file" ]]; then
    mkdir -p "$(dirname "$OUTPUT_ABS")"
    cp "$cache_file" "$OUTPUT_ABS"
    write_guard_manifest "$OUTPUT_ABS" "$MANIFEST_ABS"
    printf 'knowledge-runtime-guard cache HIT %s\n' "$cache_file" >&2
    exit 0
  fi
else
  CACHE_DIR_ABS=""
  input_hash=""
  expander_hash=""
  cache_file=""
fi

MATCH_TEXT="$(awk '
BEGIN { in_match = 0; text = "" }
{
    line = $0
    if (in_match == 0 && line ~ /Knowledge<[[:alnum:]_]+[[:space:]]+where[[:space:]]*\{/) {
        in_match = 1
        text = line
        if (line ~ /\}>/) {
            print text
            exit 0
        }
        next
    }
    if (in_match != 0) {
        text = text " " line
        if (line ~ /\}>/) {
            print text
            exit 0
        }
    }
}
' "$INPUT_ABS" | tr '\n' ' ')"
if [[ -z "$MATCH_TEXT" ]]; then
  mkdir -p "$(dirname "$OUTPUT_ABS")"
  cp "$INPUT_ABS" "$OUTPUT_ABS"
  write_guard_manifest "$OUTPUT_ABS" "$MANIFEST_ABS"
  exit 0
fi

TYPE_NAME="$(printf '%s\n' "$MATCH_TEXT" | sed -E 's/.*Knowledge<([[:alnum:]_]+)[[:space:]]+where[[:space:]]*\{.*/\1/')"
CONSTRAINT_TEXT="$(printf '%s\n' "$MATCH_TEXT" | sed -E 's/.*where[[:space:]]*\{(.*)\}>.*/\1/')"

if [[ -z "$TYPE_NAME" || -z "$CONSTRAINT_TEXT" ]]; then
  echo "error: failed to parse Knowledge<T where {...}> numeric constraints" >&2
  exit 2
fi

FIELDS=()
THRESHOLDS=()
UNITS=()
OPS=()
OP_NAMES=()
FAIL_OPS=()
while IFS= read -r raw_constraint; do
  constraint="$(printf '%s\n' "$raw_constraint" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
  if [[ -z "$constraint" ]]; then
    continue
  fi
  if [[ "$constraint" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\>\=[[:space:]]*([0-9]+([.][0-9]+)?)([[:space:]]*\<([A-Za-z_][A-Za-z0-9_]*)\>)?$ ]]; then
    FIELDS+=("${BASH_REMATCH[1]}")
    THRESHOLDS+=("${BASH_REMATCH[2]}")
    UNITS+=("${BASH_REMATCH[5]:-}")
    OPS+=(">=")
    OP_NAMES+=("ge")
    FAIL_OPS+=("<")
  elif [[ "$constraint" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\>[[:space:]]*([0-9]+([.][0-9]+)?)([[:space:]]*\<([A-Za-z_][A-Za-z0-9_]*)\>)?$ ]]; then
    FIELDS+=("${BASH_REMATCH[1]}")
    THRESHOLDS+=("${BASH_REMATCH[2]}")
    UNITS+=("${BASH_REMATCH[5]:-}")
    OPS+=(">")
    OP_NAMES+=("gt")
    FAIL_OPS+=("<=")
  elif [[ "$constraint" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\<\=[[:space:]]*([0-9]+([.][0-9]+)?)([[:space:]]*\<([A-Za-z_][A-Za-z0-9_]*)\>)?$ ]]; then
    FIELDS+=("${BASH_REMATCH[1]}")
    THRESHOLDS+=("${BASH_REMATCH[2]}")
    UNITS+=("${BASH_REMATCH[5]:-}")
    OPS+=("<=")
    OP_NAMES+=("le")
    FAIL_OPS+=(">")
  elif [[ "$constraint" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\<[[:space:]]*([0-9]+([.][0-9]+)?)([[:space:]]*\<([A-Za-z_][A-Za-z0-9_]*)\>)?$ ]]; then
    FIELDS+=("${BASH_REMATCH[1]}")
    THRESHOLDS+=("${BASH_REMATCH[2]}")
    UNITS+=("${BASH_REMATCH[5]:-}")
    OPS+=("<")
    OP_NAMES+=("lt")
    FAIL_OPS+=(">=")
  elif [[ "$constraint" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*(==|=)[[:space:]]*([0-9]+([.][0-9]+)?)([[:space:]]*\<([A-Za-z_][A-Za-z0-9_]*)\>)?$ ]]; then
    FIELDS+=("${BASH_REMATCH[1]}")
    THRESHOLDS+=("${BASH_REMATCH[3]}")
    UNITS+=("${BASH_REMATCH[6]:-}")
    OPS+=("==")
    OP_NAMES+=("eq")
    FAIL_OPS+=("!=")
  elif [[ "$constraint" =~ subclass_of ]]; then
    continue
  else
    echo "error: unsupported runtime guard constraint: $constraint" >&2
    exit 2
  fi
done < <(printf '%s\n' "$CONSTRAINT_TEXT" | tr ',;' '\n')

if [[ "${#FIELDS[@]}" -eq 0 ]]; then
  mkdir -p "$(dirname "$OUTPUT_ABS")"
  cp "$INPUT_ABS" "$OUTPUT_ABS"
  write_guard_manifest "$OUTPUT_ABS" "$MANIFEST_ABS"
  exit 0
fi

HELPER="__sounio_knowledge_guard_${TYPE_NAME}"
for idx in "${!FIELDS[@]}"; do
  sanitized_threshold="$(printf '%s' "${THRESHOLDS[$idx]}" | sed -E 's/[^A-Za-z0-9_]+/_/g')"
  HELPER="${HELPER}_${FIELDS[$idx]}_${OP_NAMES[$idx]}_${sanitized_threshold}"
  if [[ -n "${UNITS[$idx]}" ]]; then
    HELPER="${HELPER}_${UNITS[$idx]}"
  fi
done
TMP_WRAPPED="$(mktemp)"
TMP_LOWERED="$(mktemp)"
TMP_EXPANDED="$(mktemp)"
trap 'rm -f "$TMP_WRAPPED" "$TMP_LOWERED" "$TMP_EXPANDED"' EXIT

awk -v type_name="$TYPE_NAME" -v helper="$HELPER" '
function brace_delta(s,    opens, closes) {
    opens = gsub(/\{/, "{", s)
    closes = gsub(/\}/, "}", s)
    return opens - closes
}
{
    line = $0
    if (in_value == 0 && line ~ "value:[[:space:]]*" type_name "[[:space:]]*\\{") {
        sub("value:[[:space:]]*" type_name "[[:space:]]*\\{", "value: " helper "(" type_name " {", line)
        in_value = 1
        depth = brace_delta(line)
        if (depth == 0) {
            line = line ")"
            in_value = 0
        }
        print line
        next
    }
    if (in_value != 0) {
        depth += brace_delta(line)
        if (depth == 0) {
            line = line ")"
            in_value = 0
        }
        print line
        next
    }
    print line
}
' "$INPUT_ABS" >"$TMP_WRAPPED"

awk -v type_name="$TYPE_NAME" '
{
    line = $0
    if (in_type == 0 && line ~ "Knowledge<" type_name "[[:space:]]+where[[:space:]]*\\{") {
        prefix = line
        sub("Knowledge<" type_name "[[:space:]]+where[[:space:]]*\\{.*", "Knowledge<" type_name ">", prefix)
        if (line ~ /}>/) {
            suffix = line
            sub(".*}>", "", suffix)
            print prefix suffix
            next
        }
        print prefix
        in_type = 1
        next
    }
    if (in_type != 0) {
        if (line ~ /}>/) {
            suffix = line
            sub(".*}>", "", suffix)
            if (suffix != "") {
                print suffix
            }
            in_type = 0
        }
        next
    }
    print line
}
' "$TMP_WRAPPED" >"$TMP_LOWERED"

{
  printf 'fn %s(value: %s) -> %s with Panic {\n' "$HELPER" "$TYPE_NAME" "$TYPE_NAME"
  for idx in "${!FIELDS[@]}"; do
    if [[ -n "${UNITS[$idx]}" ]]; then
      threshold_literal="${THRESHOLDS[$idx]}"
      if [[ "$threshold_literal" != *.* ]]; then
        threshold_literal="${threshold_literal}.0"
      fi
      printf '    let __sounio_value_%s: %s = value.%s\n' "${FIELDS[$idx]}" "${UNITS[$idx]}" "${FIELDS[$idx]}"
      printf '    let __sounio_threshold_%s: %s = %s\n' "${FIELDS[$idx]}" "${UNITS[$idx]}" "$threshold_literal"
      printf '    let __sounio_ratio_%s = __sounio_value_%s / __sounio_threshold_%s\n' "${FIELDS[$idx]}" "${FIELDS[$idx]}" "${FIELDS[$idx]}"
      printf '    if __sounio_ratio_%s %s 1.0 {\n' "${FIELDS[$idx]}" "${FAIL_OPS[$idx]}"
      printf '        // Knowledge runtime guard diagnostic: %s.%s %s %s <%s>\n' "$TYPE_NAME" "${FIELDS[$idx]}" "${OPS[$idx]}" "${THRESHOLDS[$idx]}" "${UNITS[$idx]}"
      printf '        assert(__sounio_ratio_%s %s 1.0)\n' "${FIELDS[$idx]}" "${OPS[$idx]}"
      printf '    }\n'
    else
      printf '    if value.%s %s %s {\n' "${FIELDS[$idx]}" "${FAIL_OPS[$idx]}" "${THRESHOLDS[$idx]}"
      printf '        // Knowledge runtime guard diagnostic: %s.%s %s %s\n' "$TYPE_NAME" "${FIELDS[$idx]}" "${OPS[$idx]}" "${THRESHOLDS[$idx]}"
      printf '        assert(value.%s %s %s)\n' "${FIELDS[$idx]}" "${OPS[$idx]}" "${THRESHOLDS[$idx]}"
      printf '    }\n'
    fi
  done
  printf '    return value\n'
  printf '}\n\n'
  cat "$TMP_LOWERED"
} >"$TMP_EXPANDED"

mkdir -p "$(dirname "$OUTPUT_ABS")"
if [[ -n "$CACHE_DIR_ABS" ]]; then
  tmp_cache="$cache_file.tmp.$$"
  {
    printf '// Sounio pre-native Knowledge runtime guard .guardcache v1\n'
    printf '// input: %s\n' "$INPUT"
    printf '// input_sha256: %s\n' "$input_hash"
    printf '// expander_sha256: %s\n\n' "$expander_hash"
    cat "$TMP_EXPANDED"
  } >"$tmp_cache"
  mv "$tmp_cache" "$cache_file"
  cp "$cache_file" "$OUTPUT_ABS"
  write_guard_manifest "$OUTPUT_ABS" "$MANIFEST_ABS"
  printf 'knowledge-runtime-guard cache MISS %s\n' "$cache_file" >&2
else
  cp "$TMP_EXPANDED" "$OUTPUT_ABS"
  write_guard_manifest "$OUTPUT_ABS" "$MANIFEST_ABS"
fi
