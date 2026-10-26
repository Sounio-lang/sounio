#!/usr/bin/env node
// Operational metadata only. File identities do not attest universal execve routing.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const [op, out, ...args] = process.argv.slice(2);
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const utc = () => new Date().toISOString();
const inside = (p, root) => p === root || p.startsWith(root + path.sep);
function regular(p) { const s = fs.lstatSync(p); if (!s.isFile() || s.isSymbolicLink()) throw Error(`not regular: ${p}`); return s; }
function identity(p) { p = path.resolve(p); const real_path = fs.realpathSync(p); const s = regular(real_path); return {path:p, real_path, sha256:sha(fs.readFileSync(real_path)), bytes:s.size}; }
function put(p, x) { fs.writeFileSync(p, JSON.stringify(x, null, 2)+'\n', {flag:'wx',mode:0o600}); }
function read(p) { regular(p); return JSON.parse(fs.readFileSync(p,'utf8')); }
function ancestors(p) { for (let q=p;;q=path.dirname(q)) { const s=fs.lstatSync(q); if (!s.isDirectory() || s.isSymbolicLink()) throw Error(`unsafe directory: ${q}`); if (q===path.dirname(q)) break; } }
function same(a) { const b=identity(a.path); if (JSON.stringify(a)!==JSON.stringify(b)) throw Error(`identity changed: ${a.path}`); }
function control() { ancestors(out); const c=read(path.join(out,'capture.json')); if(c.output!==out) throw Error('capture root mismatch'); return c; }
function createBelowOutput(dir) {
  if (!inside(dir,out)) throw Error('capture directory escapes output root');
  let current=out;
  for(const component of path.relative(out,dir).split(path.sep)) {
    current=path.join(current,component);
    try { fs.mkdirSync(current,{mode:0o700}); } catch(e) { if(e.code!=='EEXIST') throw e; }
    const st=fs.lstatSync(current);
    if(!st.isDirectory()||st.isSymbolicLink()) throw Error('unsafe directory: '+current);
  }
}
function phase(c, rel, name) {
  if (!['check','run'].includes(name) || !rel.startsWith('tests/stdlib/') || !rel.endsWith('.sio') || path.posix.normalize(rel)!==rel || rel.split('/').some(x=>x==='..'||x==='.'||x==='')) throw Error('unsafe case key');
  if (!c.sources.some(x=>x.path===path.join(c.root,rel))) throw Error('case outside initial source census');
  return path.join(out,rel,name);
}
function walk(root, found=[]) { for(const d of fs.readdirSync(root,{withFileTypes:true})) { const p=path.join(root,d.name); if(d.isDirectory()) walk(p,found); else if(d.isFile()&&p.endsWith('.sio')) found.push(identity(p)); else if(d.isSymbolicLink()) { /* source symlinks are bound by their resolved file identity below */ if(p.endsWith('.sio')) found.push(identity(p)); } } return found; }
try {
  if(op==='init') {
    const [root, wrapper, stdlib] = args;
    if(!path.isAbsolute(out) || path.normalize(out)!==out || out.endsWith(path.sep)) throw Error('artifacts directory must be an absolute normalized fresh path');
    if(fs.existsSync(out) || (()=>{try{fs.lstatSync(out);return true;}catch(e){if(e.code==='ENOENT')return false;throw e;}})()) throw Error('artifacts directory already exists');
    ancestors(path.dirname(out));
    for(const input of [root,stdlib]) { const real=fs.realpathSync(input); if(inside(out,real)||inside(real,out)) throw Error(`artifacts/input overlap: ${real}`); }
    const callers=['scripts/stdlib/run_stdlib_e2e.sh','scripts/stdlib/stdlib_e2e_diagnostics.mjs','scripts/stdlib/emit_stdlib_e2e_result.mjs','scripts/lib/resolve_souc.sh'].map(p=>identity(path.join(root,p)));
    callers.push(identity(wrapper));
    const sources=walk(path.join(root,'tests/stdlib')).sort((a,b)=>a.path.localeCompare(b.path));
    fs.mkdirSync(out,{mode:0o700});
    put(path.join(out,'capture.json'),{schema:'sounio.stdlib.e2e.diagnostics.v1',output:out,root,started_at_utc:utc(),identity_scope:'Test source and caller/wrapper file identities; not an execve witness or imported-source closure. Run output combines compiler and native execution.',callers,sources});
  } else if(op==='begin') {
    const c=control(), [rel,name,...command]=args, dir=phase(c,rel,name);
    c.callers.forEach(same); c.sources.filter(x=>x.path===path.join(c.root,rel)).forEach(same);
    createBelowOutput(dir); ancestors(dir);
    put(path.join(dir,'start.json'),{test_path:rel,phase:name,command,started_at_utc:utc(),source:identity(path.join(c.root,rel))});
    process.stdout.write(path.join(dir,'combined.output'));
  } else if(op==='finish') {
    const c=control(), [rel,name,rc]=args, dir=phase(c,rel,name); ancestors(dir);
    if(!/^(0|[1-9][0-9]{0,2})$/.test(rc)||Number(rc)>255) throw Error('invalid exit status');
    const start=read(path.join(dir,'start.json')); regular(path.join(dir,'combined.output')); same(start.source); c.callers.forEach(same);
    put(path.join(dir,'result.json'),{...start,completed_at_utc:utc(),exit_code:Number(rc),combined_output:identity(path.join(dir,'combined.output'))});
  } else if(op==='verify') {
    const c=control(),[rel,name,rc]=args,dir=phase(c,rel,name); ancestors(dir);
    const result=read(path.join(dir,'result.json')); if(result.test_path!==rel||result.phase!==name||result.exit_code!==Number(rc)) throw Error('missing/mismatched completed capture'); regular(result.combined_output.path); same(result.combined_output); same(result.source);
  } else if(op==='finalize') {
    const c=control(); c.callers.forEach(same); c.sources.forEach(same);
    const now=walk(path.join(c.root,'tests/stdlib')).sort((a,b)=>a.path.localeCompare(b.path));
    if(JSON.stringify(now)!==JSON.stringify(c.sources)) throw Error('test source census changed');
    put(path.join(out,'complete.json'),{schema:c.schema,completed_at_utc:utc(),source_and_caller_identity_unchanged:true,selection_and_results:'See unchanged schema-v1 E2E result; capture is supplemental operational metadata.'});
  } else throw Error(`unknown operation: ${op}`);
} catch(e) { console.error(`error: E2E artifact capture: ${e.message}`); process.exitCode=65; }
