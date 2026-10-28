# Extracted normal serial component on integrated raw8abe

Slurm 19540 COMPLETED 0 in 4m31s on DL380, 4 CPUs/8 GiB. This is a replay of the literal serial executor and its helpers with the new compiler, plus one-million-row file and TCP selections. It does not qualify the complete public normal_pipeline import, certified WorkerFn, or the parallel pipeline.

The raw compiler is 8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932, with SOUNIO_LOOP_REGIONS=0 explicitly set. The public-module/pool emission gate remains blocked separately by closure119. No compiler, runtime, library or original Sounio fixture is changed for this replay.

## Source concordance

file.sio and tcp.sio are byte-identical copies of the previously qualified normal-abort-component02 inputs. The original abort selection covered 257 rows. The earlier million-row results came from normal-serial-component02 and normal-serial-tcp-component01, not from that abort selection.

CONCORDANCE.json compares all four extracted normal_* functions against the actual services WIP file. Every function is byte-identical: normal_header, normal_next, normal_emit_values and normal_reader_serial. The serial executor is 3309 bytes, SHA256 24459b1e2589515ac4f79b4e3d2e5959dde2c215144faf57a4ba374fdd8cda94. The complete current WIP normal_pipeline file is SHA256 4d3f1dccd1b4e210dc7c7e6937016d181baea782473c460265c95c8cd237492a, recorded against HEAD 911dfe9ff8ec96475ca11b1e91011f8ec7f9aedd with uncommitted_source=true. This receipt does not disguise that WIP as a committed source revision.

The numerical kernels use the canonical normal_scalar source SHA256 8df264dce09a54f059227dc58d5d0a4c5ab870600c62e6c6e6bc9f4d74b3f008. The unchanged qualified C++23/Arrow oracle and its source, SDK, dynamic dependencies and runtime DSO are verified by the original dependency manifests before and after the replay.

## One million rows over each inlet

The C++ generator produces sample_id,z_milli input, with 1,000,000 actual rows and 257 distinct z values. The real Sounio kernels produce cdf, sf, logcdf and logsf; the six output columns include the two input columns. The C++ oracle verifies all 1,000,000 rows and 62,500 batches of 16, preserving order and numerical values. The file and TCP Arrow artifacts are byte-identical.

| Serial inlet | 257-row native RSS | 1M-row native RSS | 1M native wall time |
| --- | ---: | ---: | ---: |
| File | 3248 KiB | 3212 KiB | 97.51 s |
| TCP loopback | 3976 KiB | 3976 KiB | 99.13 s |

These are one execution each, not selected retries or a parallel speedup benchmark. The C++ resource checker passes within each inlet. Its legacy third argument duplicates the serial scale resource file; MEMORY_SCOPE explicitly excludes any parallel runtime claim. These two sizes establish the measured memory behavior for this selection, not a Lean proof of asymptotic bounds.

The pthread interposer records its loaded marker L and zero create events in all four executions; an absent trace is not counted as zero. The Slurm job MaxRSS is 7466716 KiB across the complete compiler/oracle/harness workload, separate from the measured Sounio native process values above.

## Error and ownership controls

All original abort/error controls remain: wrong header, excess columns, sample ordering, fraction/exponent/sign syntax, range/overflow, UTF-8, quoted EOF and maximum row length. TCP deadline110, byte-budget105 and truncated-EOF22 controls retain their expected statuses. Failed streams are aborted and do not carry a successful Arrow EOS marker.

The close-error interposer actually closes the descriptor and reports EIO. The work error remains22 and cleanup_status remains5; resources are closed and no successful EOS is written. It is not a generic conversion of22 to success.

The TCP source duplicates its socket owner into the native file reader. The original socket read probe is EBUSY16 while the reader lease exists; the original socket and listener are closed before parsing proceeds. The one-million-row run therefore exercises the reader's independent lifetime, monotonic absolute deadline and explicit byte budget. Peer input is real TCP loopback data, not a substituted file.

Input/source, compiler, executable, SDK/dependency, final committed API source and artifact guards pass. The exact executed script is run-19540.sh. The first preparation attempt19539 failed2 from a shell function-trap syntax error before any Sounio test; it is preserved as failed-19539.sh.txt and is not a numerical or compiler verdict. The fresh corrected attempt was validated with bash -n before submission. No Sounio witness was changed.

## Durable custody

Slurm19544 COMPLETED0 in25seconds, 1GiB/2CPU; MaxRSS217936KiB. It rechecks the completed19540 manifests and original dependency guards, archives the failure and successful replay, source trees, C++ oracle/runtime/interposers, Arrow SDK, raw8abe and source-concordance snapshot, then tests gzip and reads back the complete archive for SHA verification.

Archive: /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/normal-integrated-export05/normal-integrated.tar.gz

SHA256: cc82bae2b6f42c205c00f1bf25998f246c3485dbcb41b39611e627d1bce2e723

Size: 82,386,352 bytes. Adjacent metadata retains the accounting and every guard/readback. run-19544.sbatch is the exact custody script. The unrelated services WIP is preserved, and no push or main merge occurs.
