#!/usr/bin/env node
// Static consistency contract across the legacy evaluator and additive Bash gate.
// Read Python as text only; no Python interpreter or numerical work is used here.
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
const root=process.argv[2];
if(!root)throw Error("usage: ci_decision_contract.mjs REPO [--selftest]");
const nativeGuard="    if: ${{ !cancelled() && (needs.impact.outputs.compiler == 'true' || needs.impact.outputs.full == 'true') && needs.source-bootstrap-selfhost-linux-x86_64.result == 'success' }}";
const fullGuard="    if: ${{ !cancelled() && (needs.impact.outputs.compiler == 'true' || needs.impact.outputs.runtime == 'true' || needs.impact.outputs.stdlib == 'true' || needs.impact.outputs.tests == 'true' || needs.impact.outputs.full == 'true') && needs.native-selfhost-linux-x86_64.result == 'success' }}";
function check(workflow,decision,additive){
 const lines=workflow.split(/\r?\n/), start=lines.indexOf("  ci-decision:");
 const mac=lines.indexOf("  native-selfhost-macos-arm64:");
 assert(mac>=0 && lines.slice(mac+1,mac+8).includes(nativeGuard),"native Mac cannot cancel stale workflow");
 const suite=lines.indexOf("  full-test-suite:");
 assert(suite>=0 && lines.slice(suite+1,suite+8).includes(fullGuard),"full suite cannot cancel stale workflow");
 assert(start>=0,"missing ci-decision");
 let end=lines.findIndex((s,i)=>i>start&&/^  [A-Za-z0-9_-]+:/.test(s));
 if(end<0)end=lines.length;
 const block=lines.slice(start,end);
 const n=block.indexOf("    needs:");
 assert(n>=0,"missing needs");
 const needs=[];
 for(const line of block.slice(n+1)){
  if(!line.trim()||line.trim().startsWith("#"))continue;
  const m=/^      - ([A-Za-z0-9_-]+)$/.exec(line);
  if(!m)break;
  assert(!needs.includes(m[1]),"duplicate workflow need");
  needs.push(m[1]);
 }
 assert(needs.includes("impact"),"missing impact need");
 const maps=[...decision.matchAll(/^    required = \{\n([\s\S]*?)^    \}/gm)];
 assert.equal(maps.length,1,"expected one legacy required map");
 const keys=[...maps[0][1].matchAll(/^        "([A-Za-z0-9_-]+)":/gm)].map(m=>m[1]);
 assert(keys.length>0,"empty legacy required map");
 assert.equal(new Set(keys).size,keys.length,"duplicate legacy key");
 const addition="stdlib-deep-science";
 assert(!keys.includes(addition),"additive job unexpectedly duplicated in legacy map");
 // Require the additive decision unconditionally after the existing evaluator.
 const text=block.join("\n");
 assert(text.includes("run: python3 scripts/ci/evaluate_ci_decision.py"),"legacy evaluator disconnected");
 assert(/- name: Require selected deep scientific replay\n        if: always\(\)\n        env:\n          NEEDS_JSON: \$\{\{ toJSON\(needs\) \}\}\n        run: bash scripts\/ci\/stdlib_deep_ci_decision\.sh/.test(text),"additive decision disconnected or conditionally skipped");
 assert(additive.includes('result=$(jq -r \'.["stdlib-deep-science"].result // "missing"\''),"additive job lookup missing");
 // The layered aggregators are part of the verdict too: a red fast or deep
 // gate must fail ci-decision, not merely show up as its own red check.
 const expected=new Set([...keys,addition,"fast-pr-gate","deep-compiler-gate"]);
 const actual=new Set(needs.filter(x=>x!=="impact"));
 const missing=[...actual].filter(x=>!expected.has(x)).sort();
 const extra=[...expected].filter(x=>!actual.has(x)).sort();
 assert(!missing.length&&!extra.length,"needs/evaluator mismatch missing="+missing+" extra="+extra);
 return {legacy:keys.length,additive:1};
}
const workflow=fs.readFileSync(path.join(root,".github/workflows/ci.yml"),"utf8");
const decision=fs.readFileSync(path.join(root,"scripts/ci/evaluate_ci_decision.py"),"utf8");
const additive=fs.readFileSync(path.join(root,"scripts/ci/stdlib_deep_ci_decision.sh"),"utf8");
const result=check(workflow,decision,additive);
if(process.argv[3]==="--selftest"){
 const controls=[
  [workflow.replace(fullGuard,fullGuard.replace("!cancelled()","always()")),decision,additive],
  [workflow.replace(nativeGuard,nativeGuard.replace("!cancelled()","always()")),decision,additive],
  [workflow.replace("      - stdlib-deep-science\n",""),decision,additive],
  [workflow.replace("run: bash scripts/ci/stdlib_deep_ci_decision.sh","run: true"),decision,additive],
  [workflow.replace("name: Require selected deep scientific replay\n        if: always()","name: Require selected deep scientific replay\n        if: success()"),decision,additive],
  [workflow.replace("      - full-test-suite\n",""),decision,additive],
  [workflow,decision.replace('        "contracts": True,\n',""),additive],
  [workflow,decision,additive.replaceAll('["stdlib-deep-science"]','["unrelated"]')],
  [workflow.replace("      - fast-pr-gate\n",""),decision,additive],
  [workflow.replace("      - deep-compiler-gate\n",""),decision,additive],
 ];
 for(const [i,args]of controls.entries())assert.throws(()=>check(...args),"accepted mutation "+i);
 console.log("CI_DECISION_CONTRACT_MUTATIONS_PASS controls="+controls.length);
}
console.log("CI_DECISION_CONTRACT_PASS legacy="+result.legacy+" additive="+result.additive);

