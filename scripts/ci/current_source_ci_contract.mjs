#!/usr/bin/env node
// Repository wiring contract, not a general shell/YAML execution verifier.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import path from 'node:path';
const root=process.argv[2];
assert(root,'usage: current_source_ci_contract.mjs REPO [--selftest]');
const workflow=fs.readFileSync(path.join(root,'.github/workflows/ci.yml'),'utf8');
const action=fs.readFileSync(path.join(root,'.github/actions/current-source-madaros/action.yml'),'utf8');
const expectedAction=`name: Receive verified current-source Madaros
description: Download this run's compiler and verify exact checkout inputs before execution.
runs:
  using: composite
  steps:
    - uses: actions/download-artifact@v4
      with:
        name: current-source-madaros-${'${{ github.run_id }}'}-${'${{ github.sha }}'}
        path: ${'${{ runner.temp }}'}/stdlib-deep-compiler
    - name: Verify exact source and compiler bytes
      shell: bash
      run: |
        set -euo pipefail
        bash scripts/ci/stdlib_deep_compiler_receipt.sh verify "$GITHUB_WORKSPACE" "$RUNNER_TEMP/stdlib-deep-compiler" "$GITHUB_SHA"
        chmod 755 "$RUNNER_TEMP/stdlib-deep-compiler/madaros.elf"
`;
function check(w,a){
 assert.equal(a,expectedAction,'receipt action changed: review the execution contract explicitly');
 function job(name){const match=w.match(new RegExp('^  '+name+':\\n([\\s\\S]*?)(?=^  [a-zA-Z0-9_-]+:|$(?![\\s\\S]))','m'));assert(match,'missing '+name);return match[1];}
 for(const consumer of ['madaros-witness-gate','gate-wave-0','madaros-current-source-deref-f64','stdlib-deep-science','sco-current-source']){
  const b=job(consumer);
  assert(b.includes('    needs: [impact, current-source-madaros]\n'),consumer+' must depend directly on producer');
  assert.equal(b.split('      - uses: ./.github/actions/current-source-madaros\n').length,2,consumer+' must verify exactly once');
  assert(!b.includes('      - uses: ./.github/actions/current-source-madaros\n        if:'),'conditional receive');
 }
 const producer=job('current-source-madaros');
 assert(producer.includes('bash scripts/ci/build_modular_madaros.sh /tmp/madaros-ci.elf'),'missing source build');
 assert(producer.includes('stdlib_deep_compiler_receipt.sh prepare'),'missing producer receipt');
 assert(producer.includes('uses: actions/upload-artifact@v4'),'missing upload');
 assert(!job('madaros-current-source-deref-f64').includes('sco_exact_integral_center_gate.sh'),'SCO still serializes f64');
 for(const gate of ['sco_stack_contract.sh','sco_step4_runtime_gate.sh','sco_corpus_gate.sh','sco_etau_install_gate.sh','sco_strategy_probe_gate.sh','sco_egraph_metadata_gate.sh','sco_exact_integral_center_gate.sh'])assert(job('sco-current-source').includes(gate),'dropped '+gate);
}
check(workflow,action);
if(process.argv[3]==='--selftest'){
 const mutations=[
 [workflow.replace('needs: [impact, current-source-madaros]','needs: impact'),action],
 [workflow.replace('      - uses: ./.github/actions/current-source-madaros\n',''),action],
 [workflow.replace('bash scripts/ci/sco_exact_integral_center_gate.sh','true'),action],
 [workflow,action.replace('shell: bash','shell: bash -seo pipefail {0}')],
 [workflow,action.replace('run: |','run: >')],
 [workflow,action.replace('        bash scripts/ci/stdlib_deep_compiler_receipt.sh verify','        : bash scripts/ci/stdlib_deep_compiler_receipt.sh verify')],
 [workflow,action.replace('set -euo pipefail','set -n')],
 [workflow,action.replace('    - name: Verify exact source and compiler bytes','    - if: false\n      name: Verify exact source and compiler bytes')],
 ];
 for(const [i,args] of mutations.entries())assert.throws(()=>check(...args),'accepted mutation '+i);
 console.log('CURRENT_SOURCE_CI_CONTRACT_MUTATIONS_PASS controls='+mutations.length);
}
console.log('CURRENT_SOURCE_CI_CONTRACT_PASS');
