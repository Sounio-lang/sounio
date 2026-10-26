#!/usr/bin/env node
// Administrative census and receipt checks; theorems are checked by Lean.
import fs from 'node:fs';
const ns='Sounio.IntervalExtended.MulDiv.',allowed=new Set(['propext','Classical.choice','Quot.sound']);
function check(source,audit,expected=53){
 const code=source.replace(/\/-[\s\S]*?-\//g,'').replace(/--[^\n]*/g,'');
 if(/\b(?:sorry|admit|native_decide|axiom|constant|unsafe|partial|implemented_by|extern)\b/.test(code))throw Error('forbidden proof/declaration escape');
 const names=[...source.matchAll(/^theorem (\w+)/gm)].map(m=>ns+m[1]);
 if(names.length!==expected||new Set(names).size!==expected)throw Error('wrong source theorem census');
 const rows=new Map();
 for(const [,n,a]of audit.matchAll(/'([^']+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)/g)){
  if(rows.has(n))throw Error('duplicate theorem audit');
  for(const ax of(a??'').split(',').map(x=>x.trim()).filter(Boolean))if(!allowed.has(ax))throw Error('unapproved axiom '+ax);
  rows.set(n,a??'');
 }
 if(rows.size!==expected||names.some(n=>!rows.has(n)))throw Error('missing or unexpected theorem audit');
 if(!audit.includes('Rounding.rd_le')||!audit.includes('Rounding.le_ru'))throw Error('missing conditional rounding premise inventory');
 return names.length;
}
function selftest(){
 const s='theorem witness : True := by trivial\n',a="'"+ns+"witness' depends on axioms: [propext]\nRounding.rd_le Rounding.le_ru\n";
 check(s,a,1);check(s,a.replace('depends on axioms: [propext]','does not depend on any axioms'),1);
 const bad=[[s,a.replace('propext','sorryAx')],[s,a.replace('propext','Lean.trustCompiler')],[s,a.replace('propext','hiddenEnclosureAxiom')],[s,a+a],[s,a.replace(ns+'witness',ns+'wrong')],[s,a.replace('Rounding.le_ru','missing')],['',a],[s,a.replace(/'.*?' depends on axioms: \[.*?\]/,'')],[s.replace('trivial','sorry'),a],[s.replace('trivial','native_decide'),a],['axiom fabricated : False\n'+s,a]];
 for(const [x,y]of bad){let refused=false;try{check(x,y,1)}catch{refused=true}if(!refused)throw Error('counterfactual accepted')}
 console.log('INTERVAL_MULDIV_AXIOM_SELFTEST_PASS controls='+bad.length);
}
function prepare(source,out){
 const s=fs.readFileSync(source,'utf8');
 const specs=[
  ['zero-infinity','if x=0 then .finite 0','if x=0 then .posInf','zero_times_entire',true],
  ['omit-lower-corners','def low4 (a b c d : ExtendedRat) := minE (minE a b) (minE c d)','def low4 (a b c d : ExtendedRat) := minE a b','product_corners'],
  ['omit-upper-corners','def high4 (a b c d : ExtendedRat) := maxE (maxE a b) (maxE c d)','def high4 (a b c d : ExtendedRat) := maxE a b','product_corners'],
  ['positive-touch','if b=0 then .empty else make (some b⁻¹) none','if b=0 then .empty else make none (some b⁻¹)','touching_positive_reciprocal'],
  ['negative-touch','else if b=0 then make none (some a⁻¹)','else if b=0 then make (some a⁻¹) none','touching_negative_reciprocal'],
  ['zero-pole','if b=0 then .empty else make (some b⁻¹) none','if b=0 then entire else make (some b⁻¹) none','zero_point_reciprocal'],
  ['division-decoration','⟨i,if containsZero b.interval then meet d .trv else d⟩','⟨i,d⟩','div_zero_containing_decoration'],
  ['negative-singleton','  | _,_ => .empty\n\ntheorem ofEndpoints_contains','  | .negInf,.negInf => entire\n  | _,_ => .empty\n\ntheorem ofEndpoints_contains','negative_infinite_singleton_empty'],
  ['positive-singleton','  | _,_ => .empty\n\ntheorem ofEndpoints_contains','  | .posInf,.posInf => entire\n  | _,_ => .empty\n\ntheorem ofEndpoints_contains','positive_infinite_singleton_empty'],
  ['negative-sign','| .negInf,.negInf => .posInf','| .negInf,.negInf => .negInf','negative_unbounded_product'],
  ['infinity-reciprocal','if 0<a then make (some 0) (some a⁻¹)','if 0<a then make (some a⁻¹) (some a⁻¹)','positive_ray_reciprocal'],
  ['finite-reversed','else if 0<a ∨ b<0 then make (some b⁻¹) (some a⁻¹)','else if 0<a ∨ b<0 then make (some a⁻¹) (some b⁻¹)','reciprocal_ofFinite']
 ];
 const rows=[];
 for(const [name,old,replacement,target,all]of specs){
  if(!s.includes(old)||(!all&&s.split(old).length!==2))throw Error('ambiguous mutation '+name);
  const m=all?s.replaceAll(old,replacement):s.replace(old,replacement);
  if(m===s)throw Error('ineffective mutation');
  const ts=[...m.matchAll(/^theorem (\w+)/gm)].map(x=>({name:x[1],line:m.slice(0,x.index).split('\n').length}));
  const i=ts.findIndex(x=>x.name===target);if(i<0)throw Error('missing target');
  const file=out+'/mutant-'+name+'.lean';fs.writeFileSync(file,m);
  rows.push([name,file,ts[i].line,(ts[i+1]?.line??m.split('\n').length)-1].join('\t'));
 }
 fs.writeFileSync(out+'/mutants.tsv',rows.join('\n')+'\n');
}
if(process.argv[2]==='--selftest')selftest();
else if(process.argv[2]==='--prepare-mutants')prepare(process.argv[3],process.argv[4]);
else if(process.argv[2]==='--check-mutant-error'){
 const [file,lo,hi]=process.argv.slice(3),s=fs.readFileSync(file,'utf8');
 if(![...s.matchAll(/\.lean:(\d+):\d+: error:/g)].some(m=>+m[1]>=+lo&&+m[1]<=+hi))throw Error('required mathematical theorem refusal absent');
 console.log('INTERVAL_MULDIV_SEMANTIC_MUTANT_REJECTED');
}else{
 const [source,audit]=process.argv.slice(2);if(!source||!audit)throw Error('SOURCE AUDIT required');
 console.log('INTERVAL_MULDIV_AXIOM_PASS theorems='+check(fs.readFileSync(source,'utf8'),fs.readFileSync(audit,'utf8'))+' allowed=propext,Classical.choice,Quot.sound primitive_rounding_premises=explicit_NOT_ESTABLISHED_IEEE');
}
