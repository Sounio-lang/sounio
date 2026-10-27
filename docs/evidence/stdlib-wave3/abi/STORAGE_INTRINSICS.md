# Storage descriptor component

Slurm 18689 qualified the descriptor/factory component against the sealed seed. Source equality was independently checked before this commit:

- self-hosted/ir/storage_intrinsics.sio: 4e7b87fa75f22c52de72efdb68b245ed9af95a98d8b5251849710fde09dc9475
- tests/run-pass/wave3_storage_descriptors.sio: d7fb06511fdae88e0351caa8cb46db3eb79dad01fec2a5c2014661b6d78d684e

Receipt snapshot: /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/storage-root-component/source/ROOT_INPUTS.sha256. Component overlay SHA256 a08c9d173fff8202d3de578fb25e37b8a93747f2dd1eeb42d9c8de5ed1a19733. The native component fixture reports STORAGE_DESCRIPTOR_COMPONENT_OK.

This module describes admitted layouts and constructs ordinary protected IR calls. It is not yet imported by production lowering and has no registered runtime hook. Its proof flags are policy inputs to this component fixture; the checker must derive them before any user-facing typed move API is admitted. This receipt does not qualify actual memory movement, lifetime propagation, region barriers or HeapVec ownership.
