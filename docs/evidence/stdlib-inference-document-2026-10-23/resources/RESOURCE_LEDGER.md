# Resource observations ledger — 2026-10-23

This ledger preserves distinct observations, not a generalized speed claim.

## Paired resource03: Slurm16425

Root: /scratch/sounio-stdlib-advanced-20261023/resource_compare03
REQUEST.txt names resource_base03 and resource_after03. The same compiler was
used for both: ee0f341e6e710d35c162084023cbb10ce72d2b7ca29726ea6e3a5994f080eded.
Three alternating paired trials; one warmup per variant; exact whole stdout
comparison; pre/post source/compiler/ELF inventories. QR batch1; CountGLM batch25.
Reported medians from the existing C++ analyzer (not recomputed here):
- QR wall3.13→3.12s/exec; maxRSS106496→8192KiB; PASS.
- CountGLM wall.5972→.6564s/exec; maxRSS42508→3188KiB; PASS.
CountGLM second individual pair time ratio1.12185159973 exceeds1.10; the frozen
rule applies to median, whose ratio1.09912926993 passes narrowly. Preserve this
noise and marginal timing outcome. Thresholds remain10%time/5%RSS. CPU and all
per-pair values remain in REPORT.tsv. GNUtime resolution is centisecond; RSS is
maximum process/descendant, not aggregate; driver can dominate small programs.

## Arena failure and same-input completed replay

Before Slurm16411:
qualification01/pilot_B31/n16_alpha.25/batch0
B31,n16,16 datasets intended in one process. Original terminated181 after11
completed datasets(0..10), wall2:18.68,maxRSS1917016KiB. Actual stderr:
'madaros: arena full (allocation of 48 bytes)'. This is the generated fixed
runtime arena exhaustion, not proof of Slurm OOM or handle-table182. Partial
output is not a completed calibration estimate.

After Slurm16420:
pilot_replay_hot01, source check06/source, same exact original driver.sio and
same compiler. CountGLM dc61d4e... and LSS7950de... are recorded in INPUTS.sha256.
Native0, wall3:24.80,maxRSS205936KiB; all16datasets complete/available, summary
status_0=16. Shared prefix of the original partial stdout was byte-compared equal
by dispatch.sh. This demonstrates repaired completion for this workload, not a
before/after speed ratio: the failed run did not complete the same amount of work.
The16 available pilot datasets are not full B199/n256 scientific calibration.

## Final repeat resource04

Slurm16479 submitted. Source resource_after04 is a byte-identical read-only copy
of full02/source; full02 was not modified. Baseline resource_base03 and compiler
are unchanged from resource03. Same frozen harness/analyzer, same thresholds and
same fixtures. Outputs resource_compare04; acceptance pending its own READY and
REPORT, not inferred from03. Job and identities retained in resource_tools04.
Host-login sbatch failed before queue submission because controller was unreachable;
worker runuser sounio(uid1000) submission succeeded as16479.

## Integration read-only review

Count workflow retains raw table, explicit key alignment, normalized log offsets,
real profile failures/status comments; masks positive-edge profiles and failed
genes separately from Poisson selections. Visible footer states Poisson alpha0,
masked counts and no CI/FDR. Full scientific-document preflight precedes output.
ODE workflow consumes only accepted-state observer records, exposes bracket and
per-record metadata, checks monitor/solver status, and labels sampled bracket,
no exact event time, no dense output, no conservation proof. Both close their
owned buffers. No new concrete integration blocker found in this source review;
actual native/concordance evidence belongs to central gates, not this review.

## Artifact SHA256 custody

```text
e56558cf4b50cece5ca5b1dacdbbbd3c7349a345e78760092fd6935035d680d5  /scratch/sounio-stdlib-advanced-20261023/resource_compare03/REQUEST.txt
27673bef3ea2bcece03358de9074a6c46f7bb2d989d7181266f03a59ebcd536e  /scratch/sounio-stdlib-advanced-20261023/resource_compare03/METRICS.tsv
306c8e8bd521e7dc0e48fc2c009d94422e674fa62457ae7a7d74d4e7690be7d2  /scratch/sounio-stdlib-advanced-20261023/resource_compare03/REPORT.tsv
f8dbbe07203e0b52ccd787ce9f6e54385ded8d4e8311bdaefa23abe8693ce005  /scratch/sounio-stdlib-advanced-20261023/resource_compare03/FILES.sha256
059df237269bb26f985e0344e43dbced03d9f10cf4f68f10bc1a28001cadba7c  /scratch/sounio-stdlib-advanced-20261023/resource_compare03/READY
2182499f72c91a7a6ccb0613a1ff2c95d7f52f7b11db577acf276085f273938c  /scratch/sounio-stdlib-advanced-20261023/qualification01/pilot_B31/n16_alpha.25/batch0/driver.sio
96136f020037e047458e36d8e3b562c4e2c1f82a7f981184dc6f72dc318a83a6  /scratch/sounio-stdlib-advanced-20261023/qualification01/pilot_B31/n16_alpha.25/batch0/driver.elf
6a12886315e85fbbdd7fede3b7a503bc7872cff09b41b234fce8cab38e02ac76  /scratch/sounio-stdlib-advanced-20261023/qualification01/pilot_B31/n16_alpha.25/batch0/datasets.tsv
10355dfb63788a7343be9825776d321d8499848445fedc8382689cf332e3b3b6  /scratch/sounio-stdlib-advanced-20261023/qualification01/pilot_B31/n16_alpha.25/batch0/native.stderr
8942b2803316755a88a26bbedd473a76af1e46dfbf8f074c178dfb82b919ca0a  /scratch/sounio-stdlib-advanced-20261023/qualification01/pilot_B31/n16_alpha.25/batch0/native.resource
b3e4cb16554c2bcfcb76d89cf9d4f65c36bf2d0783fd119db8846fa5f11ed2f4  /scratch/sounio-stdlib-advanced-20261023/pilot_replay_hot01/driver.elf
0a4e286331a78048981888ca1b7ce718cf33ac684b17d55cb39daa193cdcd3cc  /scratch/sounio-stdlib-advanced-20261023/pilot_replay_hot01/native.stdout
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  /scratch/sounio-stdlib-advanced-20261023/pilot_replay_hot01/native.stderr
15dde1022d5ecbd5164cd57644156c59a966566d37d624a6a9b608f9884cc0bf  /scratch/sounio-stdlib-advanced-20261023/pilot_replay_hot01/native.resource
7b21c770890c789d063fedb24876ac3b854b1ab4ba8ecc835c7c8a498d4b4af6  /scratch/sounio-stdlib-advanced-20261023/pilot_replay_hot01/summary.tsv
9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa  /scratch/sounio-stdlib-advanced-20261023/pilot_replay_hot01/STATUS
fe166bf24c3f3bc6106cfec9f1a3fc8b9189ac897921201f20e63e59af986138  /scratch/sounio-stdlib-advanced-20261023/pilot_replay_hot01/INPUTS.sha256
20d55661b8fc0d5b6d244316aa6a06068529c59722e65dbfa43fc17cd479a273  /scratch/sounio-stdlib-advanced-20261023/pilot_replay_hot01/INPUTS-after.verify
0c918267841aee68134db6b6e52c77adc3c9e04f7858e47168729a76e4db47f2  /scratch/sounio-stdlib-advanced-20261023/pilot_replay_hot01/shared-prefix.stdout
03353dd1509a8e9e56f40198e8eaa921a859373b5e0c2cebdaf6fc8bbd86d502  /scratch/sounio-stdlib-advanced-20261023/resource_tools04/job.sh
94d4bf47b3ea97309a2ca916ec5551eadf0ba62dc9b602e3041fa6dc2ceef76a  /scratch/sounio-stdlib-advanced-20261023/resource_tools04/advanced_resource_compare.sh
f1da82c69f79c5df7b2cb3c666cf0cc8aa94b91fb472dae1b310138b50281547  /scratch/sounio-stdlib-advanced-20261023/resource_tools04/stdlib_resource_compare.cpp
001b59a1d95776a29d0d4b8dc178403a63cffd1e211f65e98131eec7e095e056  /scratch/sounio-stdlib-advanced-20261023/resource_tools04/candidate.SOURCE.sha256
```

## Final repeat completion (supersedes pending04 above)

Slurm16479 COMPLETED0:0 elapsed2:57; READY=RESOURCE_COMPARE_PASS. Existing C++
analyzer medians: QR3.16→3.13s/exec,RSS106496→8192KiB; CountGLM(batch25)
.5912→.6336s/exec,RSS42508→2788KiB. CountGLM median wallratio1.07171853857
passes unchanged1.10 threshold; pair3alone1.12855209743 exceeds that level, so
this is a median-rule acceptance, not a claim that every pair speeds up. Median
CPU .56→.60s/exec. QRCPU3.16→3.13s/exec. All native stdout bytes matched; complete
source/compiler/ELF pre/post identity checks passed. BatchMaxRSS796080K includes
compilation and is not a kernel RSS measure. Final evidence remains descriptive
three-pair/one-worker, not a universal speed/calibration claim.

### Final04 custody

```text
17854b2e6756c7ccdbaa0def2a2b8591c850be839203f3699fe5fd5a1cc9d769  /scratch/sounio-stdlib-advanced-20261023/resource_compare04/REQUEST.txt
f85f9fddb7afe21dfe96d6b1a545b362c74349b84050ee32a382743154c07208  /scratch/sounio-stdlib-advanced-20261023/resource_compare04/METRICS.tsv
5a08f8e8d6d31be281c26478b2d7c1ae55f67775d8bfa5dd1c1ce0a50566908c  /scratch/sounio-stdlib-advanced-20261023/resource_compare04/REPORT.tsv
a03135b73d6a0e2ec91117dbbfe60dc801d15da77bedbbdfbde5b1d39d5b1eb0  /scratch/sounio-stdlib-advanced-20261023/resource_compare04/FILES.sha256
059df237269bb26f985e0344e43dbced03d9f10cf4f68f10bc1a28001cadba7c  /scratch/sounio-stdlib-advanced-20261023/resource_compare04/READY
49f3a6c2f8587f928ce982169ec7e324b6ae0270060106ddcbabbcbcb29fc95f  /scratch/sounio-stdlib-advanced-20261023/resource_compare04/COMPILER.pre.sha256
49f3a6c2f8587f928ce982169ec7e324b6ae0270060106ddcbabbcbcb29fc95f  /scratch/sounio-stdlib-advanced-20261023/resource_compare04/COMPILER.post.sha256
fb9fb9eccdfcb2b495a9cf440b24f38aa43fa780f4ef28c790a68edd77f846b8  /scratch/sounio-stdlib-advanced-20261023/resource_compare04/NATIVE.pre.sha256
fb9fb9eccdfcb2b495a9cf440b24f38aa43fa780f4ef28c790a68edd77f846b8  /scratch/sounio-stdlib-advanced-20261023/resource_compare04/NATIVE.post.sha256
```
