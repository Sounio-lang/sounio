#!/bin/bash
set -euo pipefail
export PATH=/tmp/sounio-stable-api-bin:$PATH
base=/home/devsounio/sounio-artifact-archive/2026-10-20/source-expanded-59
mirror=/workspace/sounio-artifact-archive/2026-10-20/source-expanded-59
hot=/var/lib/cephfs-slurm/training/sounio-foundry/runs
cd "$base"
test "$(id -u)" = 1000
test "$(id -g)" = 1000
test -f CLEANUP_CONSUMERS_READY
test -f GITHUB_CUSTODY_VERIFIED
test ! -e REMOVAL_COMPLETE
test "$(wc -l < REMOVAL_JOURNAL.tsv)" = 2
test "$(awk -F '\t' 'NR==2{print $1}' REMOVAL_JOURNAL.tsv)" = INTENT
cp -p REMOVAL_JOURNAL.tsv REMOVAL_FIRST_ATTEMPT.tsv
printf '7c392f4c3366235b19e020c3ca28a7fecfdad695de7e8657516de24bff2955d0  SOURCE.sha256\n3964be916049b0491574b75b98acf94d4314073944a24ce110baa65443915479  FILES.sha256\n816f9db03ff77f06e37d4454409441ce6d800fdc6601cea080dcaa04591eeb08  COHORT_RUNS.txt\n' | sha256sum --check --status
declare -A metadata
while IFS=$'\t' read -r run bytes mode mtime uid gid nlink; do
 test "$run" != run || continue
 metadata["$run"]="$bytes $mode $mtime $uid $gid $nlink"
done < ORIGINAL_METADATA.tsv
mirror_journal(){
 sync -f REMOVAL_JOURNAL.tsv
 tar -cf - REMOVAL_JOURNAL.tsv REMOVAL_AUTHORIZATION.txt QUOTA_BEFORE.txt | kubectl -n beagle exec -i sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- tar -xf - -C "$mirror"
}
mirror_journal
while IFS= read -r run; do
 src="$hot/$run/source.tar.gz"
 test -f "$src" && test ! -L "$src"
 test "$(stat -c '%s %a %Y %u %g %h' "$src")" = "${metadata[$run]}"
 expected=$(sha256sum "payload/$run/source.tar.gz" | cut -d' ' -f1)
 actual=$(sha256sum "$src" | cut -d' ' -f1)
 test "$expected" = "$actual"
 if ! awk -F '\t' -v r="$run" '$1=="INTENT" && $2==r{found=1} END{exit !found}' REMOVAL_JOURNAL.tsv; then
 printf 'INTENT\t%s\t%s\t%s\n' "$run" "$expected" "$(date -u +%FT%TZ)" >> REMOVAL_JOURNAL.tsv
 fi
 mirror_journal
 rm -- "$src"
 test ! -e "$src"
 jq -n --arg run "$run" --arg sha "$expected" --arg a "$base/payload/$run/source.tar.gz" --arg b "$mirror/payload/$run/source.tar.gz" '{scope:"SOURCE_ONLY",run:$run,sha256:$sha,custody_a:$a,custody_b:$b,release:"https://github.com/Sounio-lang/sounio/releases/tag/archive-sounio-source59-20261020",asset:($run+".source.tar.gz.gpg"),restore:"Restore with hash verification without overwriting an existing file; recovery key retained separately"}' | tee "$hot/$run/ARCHIVED_SOURCE.json" >/dev/null
 printf 'REMOVED_AND_INDEXED\t%s\t%s\t%s\n' "$run" "$expected" "$(date -u +%FT%TZ)" >> REMOVAL_JOURNAL.tsv
 mirror_journal
 printf 'REMOVED_AND_INDEXED %s\n' "$run"
done < COHORT_RUNS.txt
df -B1 "$hot" > QUOTA_AFTER.txt
printf 'APPROVED_EXACT59_SOURCE_ONLY_REMOVAL_COMPLETE\n' > REMOVAL_COMPLETE
 tar -cf - REMOVAL_JOURNAL.tsv REMOVAL_AUTHORIZATION.txt QUOTA_BEFORE.txt QUOTA_AFTER.txt REMOVAL_COMPLETE | kubectl -n beagle exec -i sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- tar -xf - -C "$mirror"
cat REMOVAL_COMPLETE QUOTA_AFTER.txt
