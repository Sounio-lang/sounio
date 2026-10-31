#!/usr/bin/env node
// Source inventory, not a compiler or a runtime coverage claim.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
export function maskSource(s) {
  let out='', state='code', depth=0;
  for(let i=0;i<s.length;i++) {
    const c=s[i], n=s[i+1];
    if(state==='line') {if(c==='\n')state='code';out+=c==='\n'?'\n':' ';continue;}
    if(state==='block') {
      if(c==='/'&&n==='*'){depth++;out+='  ';i++;continue;}
      if(c==='*'&&n==='/'){if(--depth===0)state='code';out+='  ';i++;continue;}
      out+=c==='\n'?'\n':' ';continue;
    }
    if(state==='string') {
      if(c==='\\'){out+='  ';i++;continue;}
      if(c==='"')state='code';
      out+=c==='\n'?'\n':' ';continue;
    }
    if(c==='/'&&n==='/'){state='line';out+='  ';i++;continue;}
    if(c==='/'&&n==='*'){state='block';depth=1;out+='  ';i++;continue;}
    if(c==='"'){state='string';out+=' ';continue;}
    out+=c;
  }
  return out;
}
export function walk(dir, strict=true) {
  if(!fs.existsSync(dir))return [];
  const out=[];
  for(const e of fs.readdirSync(dir,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name))) {
    const p=path.join(dir,e.name);
    if(e.isDirectory())out.push(...walk(p,strict));
    else if(e.isFile())out.push(p);
    else if(e.isSymbolicLink()&&strict)throw Error('inventory refuses unbound symlink: '+p);
  }
  return out;
}
export function inventory(stdlib,root=repo) {
  stdlib=path.resolve(stdlib);
  if(!fs.statSync(stdlib).isDirectory())throw Error('stdlib is not a directory');
  const all=walk(stdlib), sources=all.filter(p=>p.endsWith('.sio'));
  const disabled=all.filter(p=>p.endsWith('.sio.disabled'));
  const rel=p=>path.relative(stdlib,p).split(path.sep).join('/');
  const testFiles=walk(path.join(root,'tests'),false).filter(p=>p.endsWith('.sio'));
  const witnesses=new Map();
  for(const p of testFiles) {
    const code=maskSource(fs.readFileSync(p,'utf8'));
    for(const m of code.matchAll(/\buse\s+([A-Za-z_][\w]*(?:::[A-Za-z_][\w]*)*)/g)) {
      const key=m[1], set=witnesses.get(key)||new Set();
      set.add(path.relative(root,p));witnesses.set(key,set);
    }
  }
  const records=sources.map(p=>{
    const raw=fs.readFileSync(p), text=raw.toString('utf8'), code=maskSource(text), relative=rel(p);
    const module=relative.replace(/\.sio$/,'').replaceAll('/','::');
    const keys=[module];if(/::(?:lib|mod)$/.test(module))keys.push(module.replace(/::(?:lib|mod)$/,''));
    const candidates=[...new Set(keys.flatMap(k=>[...(witnesses.get(k)||[])]))].sort();
    const api=[...code.matchAll(/\bpub\s+(?:(?:linear|opaque|unsafe)\s+)?(fn|struct|enum|type|const)\s+([A-Za-z_]\w*)/g)]
      .map(m=>({kind:m[1],name:m[2],line:code.slice(0,m.index).split('\n').length}));
    const stub=path.basename(p)==='mod.sio'&&/^\s*module\s+[^\n]+\s*$/.test(code);
    return {path:relative,sha256:crypto.createHash('sha256').update(raw).digest('hex'),module,
      public_declarations:api,import_witness_candidates:candidates,
      evidence_status:'source_only',stub_module_declaration:stub,
      has_code:code.trim().length>0};
  });
  const mods=records.filter(r=>path.basename(r.path)==='mod.sio'),libs=records.filter(r=>path.basename(r.path)==='lib.sio');
  const stubs=mods.filter(r=>r.stub_module_declaration),active=[...libs,...mods.filter(r=>!r.stub_module_declaration)];
  const lanes=['nn','onn','qnn','snn','spnn','quantnn','math'];
  const hyper=p=>{const top=p.split('/')[0];return lanes.includes(top)&&(top!=='nn'&&top!=='math'||/quaternion|octonion|sedenion|hypercomplex|g2|clifford|spiking|quantum/i.test(p));};
  const hyperActive=records.map(r=>r.path).filter(hyper),hyperDisabled=disabled.map(rel).filter(hyper),hyperStub=stubs.map(r=>r.path).filter(hyper);
  return {schema:'sounio.stdlib.inventory.v1',generated_at_utc:new Date().toISOString(),
    command:'bash scripts/stdlib/scan_stdlib.sh',stdlib_path:stdlib,
    evidence_note:'Static declaration/import candidates are not typecheck, runtime, or API coverage evidence.',
    counts:{sio_files:records.length,disabled_files:disabled.length,stub_mod_files:stubs.length,
      mod_files:mods.length,lib_files:libs.length,module_entrypoints:mods.length+libs.length,
      active_module_entrypoints:active.length,hyper_active_files:hyperActive.length,
      hyper_disabled_files:hyperDisabled.length,hyper_stub_mod_files:hyperStub.length},
    stub_mod_paths:stubs.map(r=>r.path),active_entrypoint_paths:active.map(r=>r.path).sort(),
    hyper_active_paths:hyperActive,hyper_disabled_paths:hyperDisabled,hyper_stub_mod_paths:hyperStub,
    hyper_lane_counts:Object.fromEntries(lanes.map(l=>[l,{active_files:hyperActive.filter(p=>p.startsWith(l+'/')).length,
      disabled_files:hyperDisabled.filter(p=>p.startsWith(l+'/')).length,stub_mod_files:hyperStub.filter(p=>p.startsWith(l+'/')).length}])),
    hyper_lane_paths:Object.fromEntries(lanes.map(l=>[l,{active:hyperActive.filter(p=>p.startsWith(l+'/')),
      disabled:hyperDisabled.filter(p=>p.startsWith(l+'/')),stub_mod:hyperStub.filter(p=>p.startsWith(l+'/'))}])),
    modules:[...new Set(records.map(r=>r.path.split('/')[0]))].sort().map(name=>({name,
      files:records.filter(r=>r.path.startsWith(name+'/')),evidence_status:'source_only'})),
    source_files:records};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  let stdlib=path.join(repo,'stdlib'),out='',quiet=false;
  const args=process.argv.slice(2);
  try {
    for(let i=0;i<args.length;i++) {
      const a=args[i];
      if(a==='--help'){console.log('scan_stdlib.sh [--stdlib-path PATH] [--json-out PATH] [--quiet]');process.exit(0);}
      if(a==='--quiet'){quiet=true;continue;}
      if(a==='--stdlib-path'||a==='--json-out'){if(!args[i+1]||args[i+1].startsWith('--'))throw Error(a+' requires a path');const v=args[++i];if(a==='--stdlib-path')stdlib=v;else out=v;continue;}
      throw Error('unknown argument: '+a);
    }
    const value=inventory(stdlib);
    if(out){fs.mkdirSync(path.dirname(path.resolve(out)),{recursive:true});fs.writeFileSync(out,JSON.stringify(value,null,2)+'\n');}
    if(!quiet)console.log(JSON.stringify({schema:value.schema,counts:value.counts,modules:value.modules.length,json_out:out}));
  }catch(e){console.error(e.message);process.exitCode=1;}
}
