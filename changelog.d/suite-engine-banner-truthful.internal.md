Internal: the test-suite harness banner (`harness: engine=... elf=... md5=...`)
now names the engine `bin/souc` will actually run. With no override set,
`bin/souc` defaults to Madaros whenever a Madaros ELF resolves — the committed
prebuilt included — but the banner inferred the engine from
`artifacts/self-hosted/madaros` alone and reported `engine=lean_single`
(hashing the committed seed) for runs the Madaros prebuilt had produced. The
banner now mirrors the dispatcher's own resolution order
(`SOUNIO_TEST_SOUC_BIN`, `SOUNIO_SOUC_BIN` ELF override,
`SOUNIO_SOUC_ENGINE=lean_single`, then Madaros resolution with lean_single
fallback), materializing the committed prebuilt first so the reported md5
comes from the ELF that is about to run.
