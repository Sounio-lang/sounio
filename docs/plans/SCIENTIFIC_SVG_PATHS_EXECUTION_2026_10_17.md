<!-- docs:meta
topic_id: repo.docs.plans.scientific-svg-paths-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.scientific-svg-paths-execution-2026-10-17
-->

# Compact clipped SVG paths — execution 2026-10-17

## Accepted native run

Run `stdlib-path-20261017-v1`, Slurm **14889**:
**COMPLETED 0:0**, elapsed **00:01:12**.
JUnit **21 tests, 0 failures, 0 skips, 0 errors**.
Snapshot SHA-256:
`158ea3a8124aead7316af2eb8e247b45f44ae8adf3053fb7054bc7a34af10892`.

Pinned committed Madaros v0.80 decompressed ELF SHA-256:
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
Compiler gzip SHA-256:
`feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a`.
No current-source compiler rebuild is claimed.
2 CPUs, 8192 MiB, GPU off, five-minute limit, clean-v1, UID/GID 1000/1000.

Evidence root:
`/var/lib/cephfs-slurm/training/sounio-foundry/runs/stdlib-path-20261017-v1`.
Isolated worktree: `/workspace/sounio-sprint-stats-viz`.
Original remote-main baseline:
`f5b2c5850be9f3d3995719ffcade44a6d851658a`.
Remote main fetched/rechecked:
`106f7a988d5b007472d0eead724f58ac50f92200`.
Committed stdlib changes since the baseline are optimize/nelder_mead.sio and
polynomial/lib.sio, outside the focused renderer dependency closure.
The active canonical checkout was not edited. No commit/push/PR/merge or CI.

## Change

Only the clipped-line serialization in viz_html.sio changes behavior.
Each chart emits at most one path; each visible segment remains an independent
M/L subpath with the same integer endpoints and ordering. Repeated stroke/color
attributes disappear. The clipPath, round caps and audit counts remain.

Serialization is streaming, one pass, O(n), capped at 1023 segments. No extra
coordinate buffer, path-string allocation or scene storage is introduced.
Canvas and numerical clipping code are unchanged.
The new fixture exercises full 1024-sample capacity, gaps/NaN/infinity, outside
segments, zero-length segments, empty and one-sample series.
The former zero-length SVG assertion tied to line attributes moves from the old
clipping fixture to the new path fixture; its Canvas pixel assertion remains.
Contract and compatibility details: `stdlib/viz/SVG_PATHS.md`.

## Before comparison

Run `stdlib-path-20261017-before1`, Slurm **14890**:
FAILED **1:0**, elapsed **00:01:11**.
JUnit **20 passes, 1 failure, zero skips/errors**.
The new path fixture compiles and runs, but fails its expected path markup check.
Snapshot:
`34cebd85d366e16e02fd313e7f68e1c7f19146468685a0145aa03938fda5f5f5`.
Package manifests differ only in `stdlib/viz/viz_html.sio`.
The before package contains the exact prior accepted renderer, with identical
new fixture/input data, compiler and other packaged files.

This is a format/size optimization. Before failure is a structural output
requirement, not a newly discovered numerical clipping failure.

## Independent C++23 comparison and measured sizes

`tests/reference/svg_path_reference.cpp` is built separately on t560 with:

```sh
g++ -std=c++23 -O2 -Wall -Wextra -Werror -pedantic svg_path_reference.cpp -o svg-path-reference
./svg-path-reference BEFORE_NATIVE_LOG AFTER_NATIVE_LOG ARTIFACT_DIRECTORY
./svg-path-reference --sizes BEFORE.svg.gz AFTER.svg.gz
```

This comparator was authored after the native snapshots and is separate evidence,
not a claimed file inside those snapshots. Its source/binary identity and logs
are saved with the delivery. It extracts SVG documents, parses clipped line/path
geometry into canonical ordered segments and requires exact equality of endpoint
integers, styles, masks, audit counts and all remaining SVG content. Compact paths
must contain independent M/L pairs and at most one path per clipped group.

Comparison passed on native fixtures 13 (automatic DFFITS), 16 (explicit DFFITS),
18 (clipping/Compton), 19 (legend), 20 (interval forests), 21 (full capacity and
edge cases). Unchanged legend/forest documents compare byte-for-byte.

| Complete SVG artifact | Before bytes | After bytes | Reduction percent |
|---|---:|---:|---:|
| Dense 1024 samples / 1023 segments | 111053 | 18050 | 83.7465 |
| DFFITS with automatic axes | 7668 | 6027 | 21.4006 |
| DFFITS with explicit axes | 7431 | 5790 | 22.0832 |
| Physical Compton consumer | 4633 | 4177 | 9.8424 |
| Zero-length point | 616 | 615 | 0.1623 |
| Dense SVG, gzip -n default compression | 7379 | 4792 | 35.0590 |

All bytes and percentages above are measured by the C++23 comparator.
Across the new edge fixture's seven documents, 1027 line elements become three
nonempty paths containing exactly 1027 subpaths; empty/outside/invalid cases
have no paths and retain their sizes/audits.

Two deliberate negative controls also fail the comparator with exit 1:
altering one point coordinate reports changed canonical geometry; replacing
its M with L reports invalid independent-subpath grammar. These controls are
isolated copies of native logs; original job evidence is unchanged.

## Artifact inspection and limits

Exact SVGs, both raw and gzip dense artifacts, and source/log/hash custody:
`/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/compact-svg-2026-10-17/`.
Quick Look previews of the actual compact DFFITS SVG and degenerate-point SVG
were visually inspected; the isolated round-capped point remains visible.
The PNGs are thumbnails with square white padding, not Canvas exports.

Independent subpaths preserve endpoints/caps and prevent bridging across gaps.
SVG semantics references:
https://www.w3.org/TR/SVG2/paths.html and
https://www.w3.org/TR/SVG2/painting.html#LineCaps .
A compound path changes paint grouping. Bitwise browser antialias/compositing
equivalence at overlaps is not claimed. Alpha, wide strokes, markers, CSS and
per-segment DOM interaction require their own contract. XML consumers selecting
clipped line elements must now parse path subpaths.

The gain established here is artifact size and element count. These runs do not
measure kernel latency, compiler RSS, browser performance or binary size.
Native elapsed times include compilation and are not a runtime speed comparison.

The five-track sprint remains active. Next: log/symlog contracts and domain
transforms, then broader statistical assumption/resampling procedures, alongside
math boundaries, genomics count matrices and physics covariance.
