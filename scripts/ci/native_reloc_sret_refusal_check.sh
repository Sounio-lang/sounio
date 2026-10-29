#!/usr/bin/env bash
# scripts/ci/native_reloc_sret_refusal_check.sh MADAROS
#
# Discriminating test for the relocation refusal on the sret witness route
# (compile_native_v2_sret_witness_to_file). That route goes straight from
# apply_relocations_into to native_v2_write_min_elf64_to_file; on d52b82d4 (the
# #2803 head first reviewed) nothing on it checked NC_RELOC_UNKNOWN_KIND_COUNT,
# so an unknown-kind site was written out as a rel32 placeholder.
#
#   inject   MADAROS --native-v2-emit-sret-unknown-reloc OUT adds one kind-9
#            relocation before apply_relocations_into. It must report rc 20,
#            name kind 9 at 1 site, and OUT must NOT exist afterwards. The file
#            is the evidence: an rc alone does not prove nothing was written.
#   control  MADAROS --native-v2-emit-sret OUT (same route, nothing injected)
#            must write OUT, and OUT must exit 14 -- so a refusal that fired on
#            everything, or a route that writes nothing, cannot pass.
#
# Runs in well under a second per invocation (no source is compiled).
set -uo pipefail

MADAROS="${1:?usage: $0 MADAROS}"
[[ -x "$MADAROS" ]] || { echo "FAIL  not executable: $MADAROS" >&2; exit 2; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
fail=0

out="$WORK/sret_unknown_reloc.elf"
rm -f "$out"
timeout 30 "$MADAROS" --native-v2-emit-sret-unknown-reloc "$out" >"$WORK/inject.log" 2>&1
proc_rc=$?
echo "inject: madaros exit=$proc_rc"
sed 's/^/inject| /' "$WORK/inject.log"
if grep -q "FAIL to_file label=sret-unknown-reloc rc=20" "$WORK/inject.log"; then
  echo "PASS  inject: route returned rc=20"
else
  echo "FAIL  inject: route did not return rc=20"
  fail=1
fi
if grep -q "native relocation of unknown kind 9 at 1 site(s)" "$WORK/inject.log"; then
  echo "PASS  inject: diagnostic names kind 9 and 1 site"
else
  echo "FAIL  inject: no diagnostic naming kind 9 at 1 site"
  fail=1
fi
if [[ -e "$out" ]]; then
  echo "FAIL  inject: $out EXISTS ($(stat -c %s "$out") bytes) -- an image with an unpatched site was written"
  fail=1
else
  echo "PASS  inject: output file absent"
fi

ctl="$WORK/sret_control.elf"
timeout 30 "$MADAROS" --native-v2-emit-sret "$ctl" >"$WORK/control.log" 2>&1
if [[ -s "$ctl" ]]; then
  chmod +x "$ctl"
  timeout 10 "$ctl"
  ctl_rc=$?
  if [[ "$ctl_rc" == 14 ]]; then
    echo "PASS  control: uninjected sret route wrote its ELF and it exits 14"
  else
    echo "FAIL  control: uninjected sret ELF exited $ctl_rc, expected 14"
    fail=1
  fi
else
  sed 's/^/control| /' "$WORK/control.log"
  echo "FAIL  control: uninjected sret route wrote no ELF"
  fail=1
fi

if [[ "$fail" == 0 ]]; then
  echo "RESULT PASS native_reloc_sret_refusal_check"
else
  echo "RESULT FAIL native_reloc_sret_refusal_check"
fi
exit "$fail"
