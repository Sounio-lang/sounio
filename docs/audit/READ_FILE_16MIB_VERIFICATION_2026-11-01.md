# read_file >16 MiB verification
Date: 2026-11-01

## Scope
This note records a focused verification for GitHub issue #2832: legacy `lean_single` `read_file` previously returned a silently truncated string when reading files larger than 16 MiB.

The verification covered Linux x86-64 `lean_single` only. ARM64 was not validated here.

## Environment
- Repository checkout: `/Users/demetriosagourakis/dev/sounio-worktrees/warp-SIO-31`
- Linux execution environment: `souc-linux` Lima VM, `x86_64`
- Test directory inside the VM: `/tmp/sounio-readfile-2832-m43slbot`
- Source file under test: `self-hosted/compiler/lean_single.sio`
- Source digest:
  - `4e961e8e6be359ea2e6eb515fd97f0b85eb7e2f251c8c52ea76a575cfb278079  self-hosted/compiler/lean_single.sio`

## Test fixture
A temporary NUL-free file was created with 20,096,256 bytes:

- Path: `/tmp/sounio-readfile-2832-m43slbot/large.dat`
- Digest:
  - `86b6025c1c166101dda4f0799bf12bbd60a25fa26e2ace2554576de094e280b1  large.dat`
- Size:
  - `20096256`
- Tail marker:
  - `END-OF-FILE!`

The Sounio probe checked:
- `str_len(read_file(path)) == 20096256`
- byte `16777216` remains readable and equals `a`, proving the read crossed the old 16 MiB boundary
- the final 12 bytes match `END-OF-FILE!`

## Rebuild
The compiler was rebuilt from current source using the shipped seed:

```sh
limactl shell souc-linux bash -c 'ROOT=/Users/demetriosagourakis/dev/sounio-worktrees/warp-SIO-31; D=/tmp/sounio-readfile-2832-m43slbot; "$ROOT/bin/souc-lean-single-x86_64" "$ROOT/self-hosted/compiler/lean_single.sio" "$D/lean-current" > "$D/build-current.log" 2>&1'
```

Result:

- build exit: `0`
- rebuilt compiler digest:
  - `81937ae1e193af363baa05fb9d868fadf114bf86a72ccab65eae5d424ef2f069  /tmp/sounio-readfile-2832-m43slbot/lean-current`

The rebuilt compiler was byte-for-byte identical to the shipped `bin/souc-lean-single-x86_64` seed in this checkout.

## Verification run
The rebuilt compiler was used directly through the raw `lean_single` CLI:

```sh
limactl shell souc-linux bash -c 'D=/tmp/sounio-readfile-2832-m43slbot; "$D/lean-current" "$D/probe.sio" "$D/probe-current"; chmod +x "$D/probe-current"; "$D/probe-current"'
```

Observed output:

```text
bytes=20096256
PASS full length, beyond-boundary byte, and tail
```

Exit codes:

- probe compile exit: `0`
- probe run exit: `0`

## Conclusion
The current `lean_single` source implements the #2832 fix for Linux x86-64: `read_file` reads past the former 16 MiB boundary and returns the complete file contents for the verified 20,096,256-byte fixture.

Remaining unverified in this receipt:
- ARM64 `read_file` behaviour
- interrupted or failing `read(2)` paths
- files requiring multiple capacity doublings beyond the tested 20,096,256-byte size
