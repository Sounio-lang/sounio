<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-28-pireus-u250-dual-card-admission
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-28-pireus-u250-dual-card-admission
-->

# Pireus: Dual AMD Alveo U250 Admission

> **Status**: Garden | **Date**: 2026-08-28 | **Authority**: founder direction

## Butterfly

The founder owns two AMD Alveo U250 cards. Both are canonical Pireus targets.
The target is therefore not "an FPGA-shaped accelerator" and not the one card
currently visible to a device plugin. It is a declared two-instance material
fleet whose members must acquire distinct physical identities before Pireus may
say the fleet exists operationally.

## Boundary Discovered Before Semantics

A read-only census on 2026-08-28 found one reachable U250 on the DL380 host:

```text
host=HP ProLiant DL380 Gen10
cpu=Intel Xeon Gold 6262V
observed_user_pf=0000:d8:00.1
observed_management_pf=0000:d8:00.0
observed_serial=22000321B01F
observed_shell=xilinx_u250_gen3x16_xdma_shell_4_1
observed_xrt=2.23.0
kubernetes_allocatable_sounio.dev/u250=1
```

No second U250 identity was reachable through the cluster census. This is an
inventory fact, not a semantic result and not a parity receipt. It must not be
used to create the first expected output retrospectively.

The first Sounio executable therefore starts with:

```text
declared_card_count=2
material_slot_count=2
discovered_card_count=0
admitted_card_count=0
status=INVENTORY_OPEN
parity_open=false
claim_ready=false
```

Only after that result is frozen may a C++ material collector submit the
reachable card as `MATERIAL_PARITY`. The expected post-freeze intermediate
state is one admitted identity out of two, still not a complete fleet.

## Canonical Contract

The first executable must define the AMD Alveo U250 target family and exactly
two material slots. A material observation is admissible only when it binds:

1. a unique non-zero physical identity to one declared slot;
2. exactly one management PF and one user PF for that card;
3. AMD/Xilinx U250 PCI identities for both functions;
4. a ready U250 shell and an identified XRT toolchain;
5. four external DDR banks totalling at least 64 GiB;
6. a sealed receipt produced by C++ in the `MATERIAL_PARITY` role.

Two slots may not share an identity. An unsealed observation may be retained as
transport evidence but cannot increment the admitted count. The fleet reaches
material completeness only with two distinct admitted cards.

No cost, speedup, lowering, kernel-correctness, thermal-headroom, or
availability claim follows from inventory admission.

## Existing Hardware Work

The HLS and XRT surfaces under `hardware/fpga/u250_catastrophe_scan/` remain the
implementation line. Pireus does not create a competing FPGA stack. After the
Sounio freeze, those C++/HLS artifacts may provide material observations and
kernel parity, but they cannot define the target semantics or expected result.

Historical gates that invoke Python are not admissible Pireus authorities.
They remain historical evidence until replaced or enclosed by the native Loom
Guardian and a Sounio-first receipt path. Python and Rust are prohibited, and a
disposable language may not be substituted as an oracle.

## Required Negative Tests

The executable and its gate must reject or contain:

- a declaration with one or three cards;
- a material observation before the Sounio semantics are frozen;
- a Python command proposed as an oracle before process launch;
- an external LLM receipt promoted beyond `REVIEW_ONLY`;
- a C++ receipt promoted to `SEMANTIC_AUTHORITY`;
- an unsealed card observation counted as admitted;
- two slots bound to the same physical identity;
- missing or duplicated PCI functions;
- a non-U250 PCI identity or a shell that is not ready;
- promotion to `CLAIM_READY` from inventory completeness alone.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This seed establishes `GARDEN`. The next artifact is the Sounio executable with
the zero-observation result above. Live U250 facts remain outside its authority
until the resulting Sounio source and semantics are frozen by hash.

## Connections

- [`2026-07-26-fpga-acceleration-opportunity.md`](2026-07-26-fpga-acceleration-opportunity.md)
- [`u250_catastrophe_scan_fpga_spec_2026-07-26.md`](../../../research/u250_catastrophe_scan_fpga_spec_2026-07-26.md)
- [`san_imagenet_fpga_dl380_spec_2026-08-02.md`](../../../research/san_imagenet_fpga_dl380_spec_2026-08-02.md)
- [`hardware/fpga/u250_catastrophe_scan/`](../../../../hardware/fpga/u250_catastrophe_scan/)
