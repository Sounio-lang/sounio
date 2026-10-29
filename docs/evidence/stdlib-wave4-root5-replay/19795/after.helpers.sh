set -euo pipefail
sdk_inventory(){
 (cd "$1" || exit 1
  LC_ALL=C find . -printf '%y\t%m\t%p\t%l\0' | LC_ALL=C sort -z) || return 1
 return 0
}
verify_binding(){
 local dir=$1
 (cd "$dir" || exit 1
  sha256sum -c binding/IMMUTABLE.sha256 || exit 1) || return 1
 (cd "$build_cwd" || exit 1
  sha256sum -c "$build_manifest" || exit 1) || return 1
 sdk_inventory "$historical" | cmp - "$dir/binding/historical.inventory.argv0" || return 1
 sdk_inventory "$current" | cmp - "$dir/binding/current.inventory.argv0" || return 1
 return 0
}
parse_raw_ready(){
 local file=$1 expected=$2 line extra
 IFS= read -r line < "$file" || return 1
 read -r raw_sha raw extra <<< "$line" || return 1
 [[ "$raw_sha" =~ ^[0-9a-f]{64}$ && "$raw" == "$expected" && -z "${extra:-}" ]] || return 1
 printf '%s  %s\n' "$raw_sha" "$raw" | cmp - "$file" || return 1
 return 0
}
