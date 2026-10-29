#!/bin/bash
#SBATCH --job-name=stdlib-frontiers-observability
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/root-independent-frontiers01/observability-%j.stdout
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/root-independent-frontiers01/observability-%j.stderr
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
RAW=$R/results/madaros
R=$B/root-integrated05
S=$R/source
D=$B/root-independent-frontiers01
I=$D/observability-input03
OUT=$D/observability03
mkdir "$OUT"
ulimit -s 524288
ulimit -c 0
finish(){ r=$?;trap - EXIT;set +e;(cd "$S";sha256sum -c "$R/results/SOURCE.sha256") > "$OUT/SOURCE.final.verify" 2>&1 || r=1;sha256sum -c "$OUT/INPUTS.sha256" > "$OUT/INPUTS.final.verify" 2>&1 || r=1;printf '%s\n' "$r" > "$OUT/STATUS";exit "$r"; }
trap finish EXIT
(cd "$S";sha256sum -c "$R/results/SOURCE.sha256") > "$OUT/SOURCE.before.verify"
find "$I" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/INPUTS.sha256"
sha256sum "$RAW" "$CXX" "$P/tools/node" "$S/tests/reference/dp54_dense_events_reference.cpp" "$D/native-retained01/bin/graph.regions0.elf" "$D/native-retained01/bin/graph.regions1.elf" "$D/native-retained01/bin/dp54.regions0.elf" "$D/native-retained01/bin/dp54.regions1.elf" >> "$OUT/INPUTS.sha256"
"$P/tools/node" - "$S" "$OUT" <<'JS'
const fs=require('fs'),crypto=require('crypto');const[s,out]=process.argv.slice(2);const text=fs.readFileSync(s+'/tests/reference/dp54_dense_events_reference.cpp','utf8');
const start=text.indexOf('int invalid_control(');const end=text.indexOf('\n}\n',start);if(start<0||end<0)throw Error('actual invalid_control boundary absent');
const body=text.slice(start,end+3);fs.writeFileSync(out+'/DP54_INVALID_FUNCTION.sha256',crypto.createHash('sha256').update(body).digest('hex')+'  actual_invalid_control\n');
const head='#include <array>\n#include <fstream>\n#include <iostream>\n#include <sstream>\n#include <stdexcept>\n#include <string>\n';
const main='\nint main(int argc,char**argv){try{if(argc!=2)return 64;std::ifstream f(argv[1]);std::string line,last;while(std::getline(f,line))last=line;if(last!="DP54_INVALID_PASS")return 1;return invalid_control(argv[1]);}catch(const std::exception&e){std::cerr<<e.what()<<"\\n";return 1;}}\n';
fs.writeFileSync(out+'/dp54_reference.cpp',head+body+main);
JS
for name in graph_reference mutate;do "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$I/$name.cpp" -o "$OUT/$name";done
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$OUT/dp54_reference.cpp" -o "$OUT/dp54_reference"
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
printf 'case\tmode\tphase\tstatus\n' > "$OUT/RESULTS.tsv"
for mode in 0 1;do
 "$OUT/dp54_reference" "$D/native-retained01/dp54.regions$mode.stdout" > "$OUT/dp54.regions$mode.oracle.stdout" 2> "$OUT/dp54.regions$mode.oracle.stderr"
 printf 'dp54\t%s\toracle\t0\n' "$mode" >> "$OUT/RESULTS.tsv"
 timeout --kill-after=30s 300 env SOUNIO_LOOP_REGIONS="$mode" "$RAW" check "$I/graph_metrics.sio" > "$OUT/graph.regions$mode.check.stdout" 2> "$OUT/graph.regions$mode.check.stderr"
 timeout --kill-after=30s 600 env SOUNIO_LOOP_REGIONS="$mode" "$RAW" compile "$I/graph_metrics.sio" -o "$OUT/graph.regions$mode.elf" > "$OUT/graph.regions$mode.compile.stdout" 2> "$OUT/graph.regions$mode.compile.stderr"
 if grep -E 'error\[E|error:|parse error|undefined identifier|field initializer mismatch|would have dropped|cannot compile an assignment' "$OUT/graph.regions$mode.compile.stdout" "$OUT/graph.regions$mode.compile.stderr";then exit 1;fi
 chmod u+x "$OUT/graph.regions$mode.elf"
 timeout --kill-after=30s 120 "$OUT/graph.regions$mode.elf" > "$OUT/graph.regions$mode.native.stdout" 2> "$OUT/graph.regions$mode.native.stderr"
 "$OUT/graph_reference" "$OUT/graph.regions$mode.native.stdout" > "$OUT/graph.regions$mode.oracle.stdout" 2> "$OUT/graph.regions$mode.oracle.stderr"
 printf 'graph_companion\t%s\tcheck_compile_native_oracle\t0\n' "$mode" >> "$OUT/RESULTS.tsv"
 for family in graph dp54;do
  input=$OUT/graph.regions$mode.native.stdout
  verifier=$OUT/graph_reference
  if test "$family" = dp54;then input=$D/native-retained01/dp54.regions$mode.stdout;verifier=$OUT/dp54_reference;fi
  for mutation in wrong drop extra;do
   "$OUT/mutate" "$family" "$mutation" "$input" "$OUT/$family.regions$mode.$mutation.txt"
   rc=0
   "$verifier" "$OUT/$family.regions$mode.$mutation.txt" > "$OUT/$family.regions$mode.$mutation.oracle.stdout" 2> "$OUT/$family.regions$mode.$mutation.oracle.stderr" || rc=$?
   test "$rc" = 1
   printf '%s\t%s\t%s_refuse\t%s\n' "$family" "$mode" "$mutation" "$rc" >> "$OUT/RESULTS.tsv"
  done
 done
done
sha256sum "$OUT/graph.regions0.elf" "$OUT/graph.regions1.elf" "$OUT/graph_reference" "$OUT/dp54_reference" "$OUT/mutate" > "$OUT/ELFS.sha256"
echo FRONTIERS_GRAPH_OBSERVABILITY_AND_DP54_EXISTING_CPP23_VALIDATED
