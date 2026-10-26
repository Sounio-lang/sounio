#!/usr/bin/env node
// Administrative source/receipt validation only; mathematical proofs use Lean.
import fs from 'node:fs';
const namespace='Sounio.IntervalExtended.';
const allowed=new Set(['propext','Classical.choice','Quot.sound']);
function strip(src){
 let out='',i=0,depth=0,string=false;
 while(i<src.length){const a=src[i],b=src[i+1];
  if(depth){if(a==='/'&&b==='-'){depth++;out+='  ';i+=2;}else if(a==='-'&&b==='/'){depth--;out+='  ';i+=2;}else{out+=a==='\n'?'\n':' ';i++;}}
  else if(string){if(a==='\\'){out+='  ';i+=2;}else if(a==='"'){string=false;out+=' ';i++;}else{out+=a==='\n'?'\n':' ';i++;}}
  else if(a==='/'&&b==='-'){depth=1;out+='  ';i+=2;}
  else if(a==='-'&&b==='-'){while(i<src.length&&src[i]!=='\n'){out+=' ';i++;}}
  else if(a==='"'){string=true;out+=' ';i++;}else{out+=a;i++;}
 }
 if(depth||string)throw Error('unterminated source comment/string');return out;
}
function check(source,audit,expected=40){
 const code=strip(source);
 if(/\b(?:sorry|admit|native_decide|axiom|constant|unsafe|partial|implemented_by|extern)\b/u.test(code))throw Error('forbidden proof/declaration escape');
 const names=[...code.matchAll(/^theorem (\w+)/gm)].map(m=>namespace+m[1]);
 if(names.length!==expected||new Set(names).size!==expected)throw Error('wrong/nonunique source theorem census');
 const rows=new Map();
 for(const [,name,list] of audit.matchAll(/'([^']+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)/g)){
  if(rows.has(name))throw Error('duplicate audit theorem');
  const axioms=(list??'').split(',').map(s=>s.trim()).filter(Boolean);
  for(const a of axioms)if(!allowed.has(a))throw Error('unapproved axiom '+a);
  rows.set(name,axioms);
 }
 if(rows.size!==expected||names.some(n=>!rows.has(n)))throw Error('missing/unknown theorem audit');
 if(!audit.includes('Rounding.rd_le')||!audit.includes('Rounding.le_ru'))throw Error('missing conditional rounding premise inventory');
 return names.length;
}
function selftest(){
 const s='theorem witness : True := by trivial\n';
 const a="'Sounio.IntervalExtended.witness' depends on axioms: [propext]\nRounding.rd_le Rounding.le_ru\n";
 check(s,a,1);check(s,a.replace('depends on axioms: [propext]','does not depend on any axioms'),1);
 const bad=[[s,a.replace('propext','sorryAx')],[s,a.replace('propext','Lean.trustCompiler')],
  [s,a.replace('propext','hiddenEnclosureAxiom')],[s,a.replace('Sounio.IntervalExtended.witness','Sounio.IntervalExtended.unknown')],
  [s,a+a],[s,a.replace('Rounding.le_ru','missing')],[s.replace('trivial','sorry'),a],
  [s.replace('trivial','native_decide'),a],['axiom fabricated : False\n'+s,a],['',a]];
 for(const [src,out]of bad){let refused=false;try{check(src,out,1)}catch{refused=true}if(!refused)throw Error('counterfactual accepted')}
 check('/- outer /- axiom hidden -/ sorry -/\n'+s,a,1);
 console.log('INTERVAL_EXTENDED_AXIOM_SELFTEST_PASS controls='+bad.length);
}
function mutants(source,out){
 const s=fs.readFileSync(source,'utf8');
 const specs=[
  ['subtract-bound','some (x+y)','some (x-y)','lower_add'],
  ['infinite-upper','| .span _ none _ => .posInf','| .span _ none _ => .negInf','entire_endpoints'],
  ['infinite-member','match x with | .finite q => Contains i q | _ => False','match x with | .finite q => Contains i q | _ => True','negative_infinity_not_member'],
  ['upgrade-empty','| .empty => meet d .trv','| .empty => d','cap_empty_rank']
 ];
 const index=[...s.matchAll(/^theorem (\w+)/gm)].map(m=>({name:m[1],line:s.slice(0,m.index).split('\n').length}));
 const rows=[];
 for(const [name,old,replacement,theorem]of specs){
  if(s.split(old).length!==2)throw Error('nonunique mutant anchor '+name);
  const mutated=s.replace(old,replacement);if(mutated===s)throw Error('ineffective mutant');
  const file=out+'/mutant-'+name+'.lean';fs.writeFileSync(file,mutated);
  const i=index.findIndex(t=>t.name===theorem);if(i<0)throw Error('missing target theorem');
  rows.push([name,file,index[i].line,(index[i+1]?.line??s.split('\n').length)-1].join('\t'));
 }
 fs.writeFileSync(out+'/mutants.tsv',rows.join('\n')+'\n');
}
if(process.argv[2]==='--selftest')selftest();
else if(process.argv[2]==='--prepare-mutants')mutants(process.argv[3],process.argv[4]);
else if(process.argv[2]==='--check-mutant-error'){
 const [file,min,max]=process.argv.slice(3);const s=fs.readFileSync(file,'utf8');
 const lines=[...s.matchAll(/\.lean:(\d+):\d+: error:/g)].map(m=>Number(m[1]));
 if(!lines.some(n=>n>=Number(min)&&n<=Number(max)))throw Error('expected mathematical theorem refusal absent');
 console.log('INTERVAL_EXTENDED_REAL_MUTANT_REJECTED');
}else{
 const [source,audit]=process.argv.slice(2);if(!source||!audit)throw Error('SOURCE AUDIT required');
 console.log('INTERVAL_EXTENDED_AXIOM_INVENTORY_PASS theorems='+check(fs.readFileSync(source,'utf8'),fs.readFileSync(audit,'utf8'))+' allowed=propext,Classical.choice,Quot.sound primitive_rounding_premises=explicit_NOT_ESTABLISHED_IEEE');
}
