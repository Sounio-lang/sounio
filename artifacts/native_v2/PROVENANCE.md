# souc-mc v2.1.0-native-v2 — provenance

The modular self-hosted compiler that compiles `.sio` SOURCE to a runnable native
x86-64 ELF via `--native-v2-compile` (exit code = program return value).

- Release asset: `souc-mc-v2.1.0-native-v2.elf` (85 MB, gitignored — reproducible, not committed)
- md5: `d8bbb5e957ca76a9e936af5dcc5c30e0`
- Source tip: `feat/native-v2-source-bridge` @ `bdbab4e30` (this commit's lineage)
- Reproduce:
      scripts/dev/souc-build-lock.sh ./bin/souc self-hosted/compiler/main.sio souc-mc.elf
  (bootstrap `bin/souc` = c634b38f tuple-match fixed point, 96528f1d7)
- Verify (17/17): `tests/native_v2_capgate/run.sh souc-mc.elf`
