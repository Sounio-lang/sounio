<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-apple-cpu-dependency-latency-interface-material-parity-20260828
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-apple-cpu-dependency-latency-interface-material-parity-20260828
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Apple CPU Dependency-Latency Interface: Material Receipt

Date: 2026-08-28

## Status

This is a sealed MATERIAL_PARITY receipt from the canonical Apple target. It
opens parity for Sounio ingestion. It does not contain a semantic feasibility
verdict or a dependency-latency cost.

    producer_language=C++
    producer_role=MATERIAL_PARITY
    parity_receipt_valid=true
    material_observation_ready=false
    classification_requested=false
    semantic_verdict_emitted=false
    cost_present=false
    parity_open=true
    claim_ready=false

The frozen Sounio authority artifact remains the only source of requirement
meaning. Sounio must ingest these facts before classification.

## Frozen Parent

    garden_commit=30237723bc53bbee48a93893be4da5b5f2118053
    sounio_executable_commit=c924d0014c88af8873eeaa3ca5d2c11cf468a167
    authority_commit=ba85ed0689484f747e392783de4f912001153360
    sounio_source_sha256=d8c7e6f9410c36f6858fb2379efa010a5adbaa32c615d89edc3e764a0606a6be
    sounio_semantics_sha256=6819916ac4240923a149dd95ee9dcbeaba8d3826b7452dd819e177ff62ce8c7f
    authority_result=UNASSESSED

## Material Coordinate

The login locator is the canonical identity. The IP is only the pinned
transport address and OpenSSH host-key alias.

    login_locator=demetriosagourakis@sounio-language-macbook
    tailnet_identity=sounio-language-macbook
    transport_address=100.91.184.41
    material_hostname=Sounio-Language-MacBook
    material_os=Darwin
    material_os_release=27.0.0
    material_architecture=arm64
    material_model=Mac17,7
    material_cpu=Apple M5 Max
    material_target=J714c
    hardware_sha256=842335c33446d81dc1887fae3e89222fcfcd98a071262dac8bfe8fec65b6a8b6

    compiler=Apple clang version 21.0.0 (clang-2100.3.27.1)
    compiler_target=arm64-apple-darwin27.0.0
    xcode=27.0
    xcode_build=27A5228h
    toolchain_sha256=3414db5a78a90d416dd1460d7daaa8f89b7e34def45f9a0a9bf243df58ae5c6d

The ED25519 host key was checked against the T560 pinned known-hosts file:

    host_key_sha256=686f543fab35171a25ece14ff3f1d5f92c54f120bcee3bca4c440425d2ee31e5

## Execution

    cpp=tools/pireus/apple_cpu_dependency_latency_interface_material_parity.cpp
    cpp_sha256=b0ac066a1b2bb085296d05b25eac0e8c25d38c8c662d911b806af32c7d6e075f
    runner=scripts/ci/pireus_apple_cpu_dependency_latency_interface_material_parity.sh
    runner_sha256=bc2a028fbe968511e50ceb1306342158af5053d6d2290ab76f063a4dde1c8e76
    command_sha256=1f4b9f7299694e2c274a85c9f21572b58755e4a0d0ce4f20046034fb5ebdfb40

The runner failed closed on an initial return-copy attempt because MagicDNS was
not available inside the T560 network namespace. The corrected runner kept
demetriosagourakis@sounio-language-macbook as the login locator and passed
100.91.184.41 only as OpenSSH HostName and HostKeyAlias. Hardware and toolchain
identity were revalidated before the successful write and execution.

The successful command was:

    ./scripts/ci/pireus_apple_cpu_dependency_latency_interface_material_parity.sh

It compiled with -O3, strict floating-point flags, -Wall, -Wextra, -Werror, and
-arch arm64; executed 128 warmups and 1,001 samples; returned the artifacts
through T560; and verified every returned hash against the remote hashes.

## Raw Facts

The C++ producer emitted six family records without classifying them:

    CORE_CYCLE_COUNTER:
      kperf image loaded, required KPC symbols incomplete, accepted samples=0
    PROCESS_PMU_CYCLE_EVENT:
      PROC_PIDTHREADCOUNTS reads=1001, median cycles delta=1338
      domain=THREAD_PERF_LEVEL_AGGREGATE, migration observable=false
    SYSTEM_TRACE_CYCLE_EVENT:
      xctrace paths executable, trace executed=false, event configuration=ABSENT
    ARCHITECTURAL_TIMER_TICK:
      mach_absolute_time median tick delta=3, timebase=125/3
    OS_MONOTONIC_TIME:
      CLOCK_UPTIME_RAW reads=1001, median nanosecond delta=84
    FREQUENCY_DERIVED_ESTIMATE:
      hw.cpufrequency unavailable, errno=2, native cycle claim=false

measurand_validated=false is preserved. None of the cycle, tick, or nanosecond
deltas is promoted here to the requested dependency-latency cost.

## Sealed Result

    binary_sha256=924a4d3122a6c6e7ac75ac773fda8ec6aebeeba607c61c3f420a5dcee23f67ae
    summary_sha256=42701c319b81b3372098b53a8bb100e29c40a85f24be611eb9896d1675dc0913
    samples_sha256=8fe07f2b44174af64ba8014b151a86c8a37589e35b7753a5b037cb75e5cae582
    remote_hashes_sha256=a03e6202f57ecc0de1148d9b34965344dd22213058bd6ceea01ca805918ad1d6
    result_sha256=2df233f73c400e4a1330c5e26fbbef6f92f3340adf59a05a56eee797c0c43a0c
    material_receipt_sha256=038176404c65e19c4c3424c0081f7b8c660638ad788ee300a32c67a84c705109

The repository evidence is:

    receipt=docs/research/evidence/pireus_apple_cpu_dependency_latency_interface_material_parity_20260828.txt
    summary=docs/research/evidence/pireus_apple_cpu_dependency_latency_interface_material_parity_summary_20260828.txt
    samples=docs/research/evidence/pireus_apple_cpu_dependency_latency_interface_material_parity_samples_20260828.tsv

## Loom Decisions

    preexec_frame_sha256=40809e87bce77853d49e380a5a5d3e29cc5acfc0f31e97727e5ee53ab3f438e4
    preexec_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN
    seal_frame_sha256=5a342796f5adfb25f361137624fe9eeb8d3f7e1d9855fad9dbe0d4e7ad3aee1f
    seal_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN
    evidence_write_command_sha256=39256c99c3854d4f4f0548dc59a5562ae3256492bd0e464793b5a48c55030d8e
    evidence_write_frame_sha256=fdde93b229292bb1c18b4fd2d08381f01199f4498dd19704ab03ef7005dab001
    evidence_write_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN
    commit_command_sha256=f869c833c164d6bc01701b8a7ea24bf9edbf53b90bd8b6a5a3e0cfdef8a0fb43
    commit_frame_sha256=c9c935fb4df1987a4c02d75fc77d97cc777459d7d060d144e57c10a5c18cb637
    commit_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN

The next causal step is a new Sounio executable that verifies the sealed receipt
and maps the raw family facts into the already frozen classifier. Until that
step runs, material_observation_ready=false, classification_requested=false,
and CLAIM_READY=false.
