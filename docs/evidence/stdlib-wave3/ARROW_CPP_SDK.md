# Official Arrow C++ oracle SDK

The interoperability oracle uses official Apache Arrow C++ 26.0.0 packages for
Ubuntu 24.04 amd64. It includes Arrow Flight and JSON, LZ4, and Zstd support.
The production Arrow implementation remains Sounio.

Run `bash scripts/stdlib/provision_arrow_cpp.sh ABS_FRESH_OUTPUT` to extract the
four pinned packages without installing system packages. The output contains
`root`, package/version manifests, library linkage, and SHA256 identities.
Set `ARROW_CPP_PREFIX` to the resulting root for the interoperability gate.
`ARROW_CPP_DEB_CACHE` and `ARROW_CPP_RUNTIME_CACHE` accept archived package
inputs for disconnected replay. Runtime libraries are Ubuntu dependencies;
their exact downloaded identities and versions are recorded separately.

The pinned package hashes came from the Apache signed Noble package index.
The download is documented at [Apache Arrow installation](https://arrow.apache.org/install/).
The captured InRelease was verified with gpgv and signing fingerprint
BE7EF45DBAD38E4EECED390E9CBA4EF977CA20B8. Its SHA256 is
9afa0df025a5f16918af7a211df7a1544f42a07b8d51bbda176cb4649b675808.
The corresponding Packages.gz SHA256 is
38cd7e834717fbb530566833ddc6433479130f07a2d43c2b421a84eb43e89187;
it matched the signed InRelease entry. The four package hashes are checked
before extraction and retained in scripts/stdlib/arrow_sdk_packages.tsv.

Slurm 18103 qualified C++23 compilation and native execution against the SDK:
a record-batch IPC stream roundtrip, FlightDescriptor serialization roundtrip,
the expected registered extension types, and compile-time version/features.
The exact native marker was ARROW26_SDK_CPP_OK JSON LZ4 ZSTD FLIGHT.
Slurm 18115 qualified the reusable extraction script with the same C++ probe.
The probe, executable, bound libraries, input manifests, and final verifications
are retained in the wave3 sdk packet.

SDK qualification is separate from Sounio producer interoperability.
Slurm 18101 is retained as a failed compiler invocation: strict warning flags
were applied to third-party headers. The corrected invocation uses -isystem
for SDK headers while retaining -Wall -Wextra -Werror for the oracle source.
No PyArrow or Python numerical implementation is used by this SDK path.
