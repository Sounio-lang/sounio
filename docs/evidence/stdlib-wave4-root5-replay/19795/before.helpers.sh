set -euo pipefail
sdk_inventory(){ (cd "$1"; LC_ALL=C find . -printf '%y\\t%m\\t%p\\t%l\\0' | LC_ALL=C sort -z); }
verify_binding(){
 local dir=$1
 (cd "$dir"; sha256sum -c binding/IMMUTABLE.sha256)
 (cd "$build_cwd"; sha256sum -c "$build_manifest")
 sdk_inventory "$historical" | cmp - "$dir/binding/historical.inventory.argv0"
 sdk_inventory "$current" | cmp - "$dir/binding/current.inventory.argv0"
}
parse_raw_ready(){
 local file=$1 expected=$2 extra
 local ready=$file expected_raw=$expected
  read -r raw_sha raw extra < "$ready"
  [[ "$raw_sha" =~ ^[0-9a-f]{64}$ && "$raw" == "$expected_raw" && -z "${extra:-}" && -x "$raw" && ! -L "$raw" ]] || return 1
}
