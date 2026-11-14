#!/usr/bin/env node
// Text/provenance validation only; all mathematical proofs are checked by Lean.
import fs from 'node:fs';
function check(source,audit){
 if(/^\s*(?:axiom|constant)\s/m.test(source)||/\b(?:sorry|admit|native_decide)\b/.test(source.replace(/\/-[\s\S]*?-\//g,'').replace(/--[^\n]*/g,'')))throw Error('forbidden proof/declaration escape');
 const names=[...source.matchAll(/^theorem (\w+)/gm)].map(m=>'Sounio.IntervalFinite.'+m[1]);
 const lines=[...audit.matchAll(/'([^']+)' depends on axioms: \[([^\]]*)\]/g)];
 const actual=new Map();const allowed=new Set(['propext','Classical.choice','Quot.sound']);
 for(const [,name,axioms]of lines){if(actual.has(name))throw Error('duplicate audit row');for(const a of axioms.split(',').map(s=>s.trim()).filter(Boolean)){if(!allowed.has(a))throw Error('unapproved axiom '+a);}actual.set(name,axioms);}
 if(actual.size!==names.length||names.some(n=>!actual.has(n)))throw Error('missing/unknown theorem audit');
 if(!audit.includes('Rounding.rd_le')||!audit.includes('Rounding.le_ru'))throw Error('missing explicit primitive-premise inventory');
 return names.length;
}
if(process.argv[2]==='--selftest'){
 const source='theorem witness : True := by trivial\n';
 const good="'Sounio.IntervalFinite.witness' depends on axioms: [propext]\nRounding.rd_le Rounding.le_ru\n";
 check(source,good);
 const bad=[
  [source,good.replace('propext','sorryAx')],
  [source,good.replace('propext','Lean.trustCompiler')],
  [source,good.replace('propext','hiddenEnclosureAxiom')],
  [source,good.replace("'Sounio.IntervalFinite.witness' depends on axioms: [propext]\n",'')],
  [source,good+good],
  [source,good.replace('Rounding.le_ru','unavailable')],
  ['axiom fabricated : False\n'+source,good],
  [source.replace('trivial','sorry'),good],
  [source.replace('trivial','native_decide'),good]
 ];
 for(const [s,a]of bad){let rejected=false;try{check(s,a);}catch{rejected=true;}if(!rejected)throw Error('negative control accepted');}
 console.log('INTERVAL_AXIOM_INVENTORY_SELFTEST_PASS controls='+bad.length);
}else{
 const [source,audit]=process.argv.slice(2);if(!source||!audit)throw Error('SOURCE AUDIT_STDOUT required');
 console.log('INTERVAL_AXIOM_INVENTORY_PASS theorems='+check(fs.readFileSync(source,'utf8'),fs.readFileSync(audit,'utf8'))+' allowed=propext,Classical.choice,Quot.sound primitive_rounding_premises=explicit_NOT_ESTABLISHED_hardware');
}
