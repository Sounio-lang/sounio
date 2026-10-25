# SIO-273 four native scientific workflows handoff

Notion: https://app.notion.com/p/4043f37f93718106bd04d063f20df465
Owner Devpass; status In progress until reviewed PR leaves draft; Done only after
human merge. This lane does not merge, auto-merge or enqueue main.

Worktree /workspace/sounio-scientific-native-workflows-20261025
Branch devpass/scientific-native-workflows-20261025
Original preserved dependency HEAD0708a91d; initial producer/custody commit1e25e451ae;
main8eca8e0c clean merge c9946adb; portable gate/guide commit7c5b2cbcae.
Scientific example bodies and independent numerical oracle unchanged throughout.

Retained17451 gate06 bothengines PASS59s. Currentmain17533 and finalportable17534
both COMPLETED0:0/58s, allfourcases botha292/895, C++science/graphics/pixels/PNG
roundtrip,10specificcontrols perengine; complete source/compiler integrity PASS.
Final stagedsource catalog matches ownWT. Actualstdout/PNG/SVG allcmpidentical
originalgate06. Full source/ELFs/PPMs/mutantinputs remain onworker, no deletions.
Final worker /scratch/sounio-stdlib-inference-scale-20261024/SIO273-main02.

Two compact bundles and manifests are documented in README. Original bundle
includes fouractualPNG/SVGpairs; new362623B bundle adds freshcurrent-main receipts
and byte-reuse checks, avoidsduplicatingimages/ELFs. Both copiedarchives extracted
and internalmanifests independently verified. Scope/science limits in exampleguide
remain mandatory. No broadcalibration/fullcorpus/certificationclaim.

Root owns shared deep_science_gate.sh/workflows. Proposed additive focused block:
rc=0
SCIENCE_NATIVE_OUTPUT="$out/scientific-native-workflows" SCIENCE_NATIVE_SEED_BIN="${SEED_RAW_BIN:-$repo/bin/souc-lean-single-x86_64}" MADAROS_RAW_BIN="$MADAROS_RAW_BIN" timeout 1200 bash scripts/stdlib/scientific_native_workflows_gate.sh > "$out/scientific-native-workflows.gate.stdout" 2> "$out/scientific-native-workflows.gate.stderr" || rc=$?
record focused-gate scientific-native-workflows "$rc"

Root final review and central exact-block/fullcomposition replay remain pending.
No push or PR by this lane before root final review. Claims canonicalstate
/tmp/sounio-coord/2493585255 via ABSOLUTE /workspace/sounio/bin/sounio-coord.
