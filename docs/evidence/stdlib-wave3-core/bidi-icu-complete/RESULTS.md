# Qualified diagnostic result

Slurm 19261 completed 0 in 3m55s with 4 CPUs on dl380-dl380-proxmox.
Every expected/actual phase in receipts/RESULTS.tsv agrees.
Slurm 19263 completed 0 in 40s and verified durable archive content against live
source, scripts, controls, and all three candidate ICU libraries.

| SDK / control | Official cases executed | Base errors | Mixed scalar-level errors | Visual-map errors | Inventory errors | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pristine ICU 79.1 | 861948 | 0 | 101 | 30 | 0 | 1 |
| Prior two-assignment candidate | 861948 | 0 | 93 | 26 | 0 | 1 |
| Complete isolated candidate | 861948 | 0 | 0 | 0 | 0 | 0 |
| Candidate + wrong observed level | 861948 | 0 | 1 | 0 | 0 | 1 |
| Candidate + wrong observed base | 861948 | 1 | 0 | 0 | 0 | 1 |
| Candidate + wrong observed order | 861948 | 0 | 0 | 1 | 0 | 1 |
| Candidate + missing observation | 861948 | 0 | 0 | 0 | 1 | 1 |
| Candidate + duplicate observation | 861948 | 0 | 0 | 0 | 1 | 1 |

The collector distinguishes read/executed cases from compared observations.
The missing/duplicate controls compare 861947/861949 observations respectively,
while executing every official input once. Each mutant fires once, at the
original character-test line 67/mode 2. All canonical expected files stay intact.

The unchanged locked reference also returns 0 for the full candidate and 1 for
both before SDKs. It reports 1993675 exact mixed scalar-level checks and 2943927
visual-map scalar entries. It separately reports 950252 optimized unidirectional
level observations, of which 165604 differ from unoptimized official levels;
these are the pre-existing explicit ICU contract, not a new waiver or subset.

The 12 SDK controls return 12/0 for the candidate; pristine and the prior SDK
return 12/8. Counts mean cases/failures. They detect a repair that merely drops
future pairs on overflow but fails to undo earlier completed pairs.

The existing primary gate refuses the candidate prefix with status 2 and the
message in receipts/policy.candidate-as-primary.stderr. No forbidden output
directory was created. No compiler check/build/native Sounio execution is
claimed by this ICU-only candidate gate.

## Exact identities

* Gate script:7b2f466742c70491c847db556f750ec03af06dd2be71c59a04d70d0e80a7e9ed.
* Collector:5187f494b13ad695ee64891378f199e5ee1b2559a3783c9f9143eb7585a6942b.
* SDK controls:cbf1401ce2cf38be3c87eb0b679221f8c500f0b41f5617b9c01cbd2f5e624aef.
* Complete source-input manifest:9584e4a2d84434a258e8ae900ea003f03bede2f5c38baf94b2158d13c517015b.
* Derived ubidi.cpp:87265ac33c060b59e7550a6fe00905dc9984c7916d62a3a92ffc08faf010c9e7.
* Derived ubidiimp.h:0078529043507a38f14eb8c7572a025a8c449e9fc3b6f62f990937f82e3dbf06.
* Full candidate patch:90c45bd8e8b489f0d645d200381063e3549af50cc0053ae5d8fd7e3b0d8ee467.
* libicuuc.so.79.1:66dc40ec868d99722b7f1dac83c8d00b7f82e8f539eaf091f8a7f41a2ae108ce.
* libicui18n.so.79.1:34ec92e926330c0d99b6af3abf7129ef28491f7f101a4b1e704cb81663cd0423.
* libicudata.so.79.1:2f26922e9964e3ca2da1e08fc9dde67feeb99b153a4a2b74ca7fc3ec13a0dba2.

The SDK prefix is
/scratch/sounio-stdlib-wave3-20261027/core/bidi-icu-complete03/candidate/install.
The source and exact worker are its sibling source/ and ../../worker.sbatch.
The private build stays marked Diagnostic candidate only by SCOPE.

## Time and RSS observations

These are single scheduled observations, not comparative benchmarks.
All phase stderr files contain /usr/bin/time -v.

| Phase | Wall time | Maximum RSS (reported kbytes) |
| --- | --- | ---: |
| Candidate SDK make, 4 CPUs | 3:06.59 | 671376 |
| Pristine full collector | 0:02.20 | 3984 |
| Two-assignment full collector | 0:02.22 | 4008 |
| Complete candidate full collector | 0:02.21 | 4768 |
| Complete candidate unchanged locked reference | 0:02.25 | 4820 |

The candidate performs additional allocation and tracking. No no-allocation,
constant-space, general ICU performance, or whole-library API claim is made.

## Durable custody

/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/bidi-icu-complete01

* final-sdk-source-receipts-19261.tar.gz:
  bf3715acfff7eb49fc2c4e51a8c37922d04d96f5c49820d291541cf19630b310
* before-failed-history-19235-19247-19253.tar.gz:
  33004a2061772335f883ebdce24b10584e6721e327399cb48438563b184bdf6a

The final archive contains exact input scripts/controls/corpus archive, full
derived ICU source, private installed SDK, executable judges, timings, receipts,
hash manifests, and worker. Build objects are omitted. The history archive
preserves full before collection and failed attempts 19247/19253, including the
partial candidate SDK. The pristine SDK and the earlier two-assignment SDK are
referenced read-only and remain in their pre-existing durable directories.

Final guards verified INPUTS, PRISTINE, BEFORE, SOURCE_TREE, CANDIDATE and
EXECUTABLES manifests and the worker SOURCE manifest. The durable archive job
also compared extracted source/script/control/three-library bytes with live
qualification files. ARCHIVES.sha256 and ARCHIVES.final.verify accompany them.

This commit authorizes root to adopt these immutable new diagnostic files into
its isolated worktree. It does not authorize primary-oracle promotion, edits to
this worktree/branch, publication upstream, push, or merge into main.
