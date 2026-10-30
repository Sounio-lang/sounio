Internal: the test-suite harness banner (`harness: engine=... elf=... md5=...`)
now names the engine `bin/souc` will actually run. With no override set,
`bin/souc` defaults to Madaros whenever a Madaros ELF resolves — the committed
prebuilt included — but the banner inferred the engine from
`artifacts/self-hosted/madaros` alone and reported `engine=lean_single`
(hashing the committed seed) for runs the Madaros prebuilt had produced. The
banner now mirrors the dispatcher's own resolution order
(`SOUNIO_TEST_SOUC_BIN`, `SOUNIO_SOUC_BIN` ELF override,
`SOUNIO_SOUC_ENGINE` forced legacy under all three spellings the dispatcher
accepts (`lean_single`, `lean-single`, `leansingle`), then Madaros resolution
with lean_single fallback), materializing the committed prebuilt before
hashing — a checkout that refreshed the tracked `.gz` but left an older ELF
is reported as the compiler that is about to run — and refusing to continue
when that materialization fails and no higher-priority override resolves.
