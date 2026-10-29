#!/usr/bin/env node
// Operational metadata only. File identities do not attest universal execve routing.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
const runtimeFixturePath=path.join(path.dirname(fileURLToPath(import.meta.url)),'runtime_fixtures.mjs');
const runtimeFixtures=fs.existsSync(runtimeFixturePath)?await import(pathToFileURL(runtimeFixturePath).href):null;
const BIDI_TEST='tests/stdlib/wave3-bidi/conformance.sio';
const BIDI_GOLDEN='tests/fixtures/wave4_bidi_runtime/official-corpus.json';
import {argvSnapshot,verifyArgvSnapshot} from './argv_inputs.mjs';
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
  if (!['check','run','compile','native'].includes(name) || !rel.startsWith('tests/stdlib/') || !rel.endsWith('.sio') || path.posix.normalize(rel)!==rel || rel.split('/').some(x=>x==='..'||x==='.'||x==='')) throw Error('unsafe case key');
  if (!c.sources.some(x=>x.path===path.join(c.root,rel))) throw Error('case outside initial source census');
  return path.join(out,rel,name);
}

function fixtureBinding(c,rel,name,command) {
  const helper=path.join(c.root,'scripts/stdlib/runtime_fixtures.mjs');
  if(name!=='native'||rel!==BIDI_TEST) {
    if(command.includes(helper))throw Error('runtime fixture helper outside registered phase');
    return null;
  }
  const output=path.join(out,rel,'runtime-fixture');
  if(!runtimeFixtures||command.length!==12||command[0]!=='node'||command[1]!==path.join(c.root,'scripts/stdlib/bounded_command.mjs')||!/^([1-9][0-9]*)$/.test(command[2])||!/^([1-9][0-9]*)$/.test(command[3])||command[4]!=='--'||command[5]!=='node'||command[6]!==helper||command[7]!=='run'||command[8]!==c.root||command[9]!==rel||command[11]!==output)throw Error('runtime fixture native command binding');
  const compiler=read(path.join(phase(c,rel,'compile'),'result.json'));
  const cc=compiler.command;
  if(compiler.exit_code!==0||cc.length!==10||cc[0]!=='node'||cc[1]!==command[1]||!/^([1-9][0-9]*)$/.test(cc[2])||cc[3]!==command[3]||cc[4]!=='--'||cc[5]!==c.callers.at(-1).path||cc[6]!=='compile'||cc[7]!==path.join(c.root,rel)||cc[8]!=='-o'||cc[9]!==command[10])throw Error('runtime fixture compile output binding');
  same(compiler.source);same(compiler.combined_output);
  regular(command[10]);const originalElf=identity(command[10]);
  if(originalElf.bytes===0||fs.readFileSync(command[10]).subarray(0,4).toString('hex')!=='7f454c46')throw Error('runtime fixture compile output is not ELF');
  const golden=path.join(c.root,BIDI_GOLDEN);
  const helperPin=c.callers.find(x=>x.path===helper),goldenPin=c.callers.find(x=>x.path===golden);
  if(!helperPin||!goldenPin)throw Error('runtime fixture helper/golden absent from initial capture');
  same(helperPin);same(goldenPin);
  return {schema:'sounio.stdlib.runtime-fixture.binding.v1',fixture:'unicode18-bidi',test_path:rel,helper:helperPin,golden:goldenPin,originalElf,executedElfPath:path.join(output,'program.elf'),output};
}
function fixtureSummary(c,start,rc) {
  const b=start.runtime_fixture;if(!b)return null;
  const expectedOutput=path.join(out,BIDI_TEST,'runtime-fixture'),helper=path.join(c.root,'scripts/stdlib/runtime_fixtures.mjs');
  const command=start.command;
  if(start.test_path!==BIDI_TEST||start.phase!=='native'||b.schema!=='sounio.stdlib.runtime-fixture.binding.v1'||b.output!==expectedOutput||b.helper.path!==helper||b.golden.path!==path.join(c.root,BIDI_GOLDEN)||b.originalElf.path!==command[10]||command.length!==12||command[0]!=='node'||command[1]!==path.join(c.root,'scripts/stdlib/bounded_command.mjs')||!/^([1-9][0-9]*)$/.test(command[2])||!/^([1-9][0-9]*)$/.test(command[3])||command[4]!=='--'||command[5]!=='node'||command[6]!==helper||command[7]!=='run'||command[8]!==c.root||command[9]!==BIDI_TEST||command[11]!==expectedOutput)throw Error('runtime fixture terminal command/output binding');
  if(!c.callers.some(x=>JSON.stringify(x)===JSON.stringify(b.helper))||!c.callers.some(x=>JSON.stringify(x)===JSON.stringify(b.golden)))throw Error('runtime fixture initial pins differ');
  // Recheck the compile envelope without touching its intentionally removed
  // temporary output. The original ELF identity was sealed by native start.
  const compiled=read(path.join(phase(c,BIDI_TEST,'compile'),'result.json')),cc=compiled.command;
  if(compiled.exit_code!==0||cc.length!==10||cc[0]!=='node'||cc[1]!==command[1]||!/^([1-9][0-9]*)$/.test(cc[2])||cc[3]!==command[3]||cc[4]!=='--'||cc[5]!==c.callers.at(-1).path||cc[6]!=='compile'||cc[7]!==path.join(c.root,BIDI_TEST)||cc[8]!=='-o'||cc[9]!==b.originalElf.path)throw Error('runtime fixture terminal compile binding');
  same(compiled.source);same(compiled.combined_output);
  const result=runtimeFixtures.verifyBidiArtifacts(c.root,BIDI_TEST,expectedOutput,b,rc===0?0:1);
  result.manifest=identity(result.manifest.path);return result;
}
function verifyPhaseCapture(c,rel,name) {
  const dir=phase(c,rel,name);ancestors(dir);
  const names=fs.readdirSync(dir).sort();
  if(JSON.stringify(names)!==JSON.stringify(['combined.output','result.json','start.json']))throw Error('unfinished or unexpected phase files: '+rel+'/'+name);
  const raw=path.join(dir,'combined.output');regular(raw);
  const start=read(path.join(dir,'start.json')),result=read(path.join(dir,'result.json'));
  if(start.source.path!==path.join(c.root,rel)||!Array.isArray(start.command)||start.command.length===0||start.test_path!==rel||start.phase!==name||result.test_path!==start.test_path||result.phase!==start.phase||JSON.stringify(result.command)!==JSON.stringify(start.command)||result.started_at_utc!==start.started_at_utc||JSON.stringify(result.source)!==JSON.stringify(start.source)||JSON.stringify(result.runtime_fixture)!==JSON.stringify(start.runtime_fixture)||result.combined_output.path!==raw||!Number.isInteger(result.exit_code)||result.exit_code<0||result.exit_code>255)throw Error('terminal phase receipt mismatch: '+rel+'/'+name);
  same(start.source);same(result.combined_output);
  const summary=fixtureSummary(c,start,result.exit_code);
  if(JSON.stringify(summary)!==JSON.stringify(result.runtime_fixture_artifacts??null))throw Error('runtime fixture terminal artifact seal differs');
  if(name==='native'&&rel===BIDI_TEST&&!start.runtime_fixture)throw Error('registered runtime fixture lacks native binding');
  return result;
}

function verifyAllCaptures(c) {
  let count=0;
  function visit(dir) {
    const rel=path.relative(out,dir).split(path.sep).join('/');
    const match=/^(tests\/stdlib\/.+\.sio)\/(check|run|compile|native)$/.exec(rel);
    if(rel===BIDI_TEST+'/runtime-fixture') {
      const result=verifyPhaseCapture(c,BIDI_TEST,'native');
      if(!result.runtime_fixture)throw Error('runtime fixture tree lacks native binding');
      return;
    }
    if(match) {verifyPhaseCapture(c,match[1],match[2]);count++;return;}
    for(const name of fs.readdirSync(dir)) {
      const child=path.join(dir,name), st=fs.lstatSync(child);
      if(st.isSymbolicLink()) throw Error('symlink in capture tree: '+child);
      if(st.isDirectory()) {if(name==='runtime-fixture'&&child!==path.join(out,BIDI_TEST,'runtime-fixture'))throw Error('runtime fixture tree outside registered path');visit(child);}
      else if(!(dir===out&&['capture.json','argv-inputs.after.json'].includes(name)&&st.isFile())) throw Error('unexpected capture file: '+child);
    }
  }
  visit(out);return count;
}
function walk(root, found=[]) { for(const d of fs.readdirSync(root,{withFileTypes:true})) { const p=path.join(root,d.name); if(d.isDirectory()) walk(p,found); else if(d.isFile()&&p.endsWith('.sio')) found.push(identity(p)); else if(d.isSymbolicLink()) { /* source symlinks are bound by their resolved file identity below */ if(p.endsWith('.sio')) found.push(identity(p)); } } return found; }
try {
  if(op==='init') {
    const [root, wrapper, stdlib] = args;
    if(!path.isAbsolute(out) || path.normalize(out)!==out || out.endsWith(path.sep)) throw Error('artifacts directory must be an absolute normalized fresh path');
    if(fs.existsSync(out) || (()=>{try{fs.lstatSync(out);return true;}catch(e){if(e.code==='ENOENT')return false;throw e;}})()) throw Error('artifacts directory already exists');
    ancestors(path.dirname(out));
    for(const input of [root,stdlib]) { const real=fs.realpathSync(input); if(inside(out,real)||inside(real,out)) throw Error(`artifacts/input overlap: ${real}`); }
    const callers=['scripts/stdlib/run_stdlib_e2e.sh','scripts/stdlib/argv_inputs.mjs','scripts/stdlib/stdlib_e2e_diagnostics.mjs','scripts/stdlib/emit_stdlib_e2e_result.mjs','scripts/stdlib/bounded_command.mjs','scripts/lib/resolve_souc.sh'].map(p=>identity(path.join(root,p)));
    const fixtureHelper=path.join(root,'scripts/stdlib/runtime_fixtures.mjs');
    if(fs.existsSync(fixtureHelper))callers.push(identity(fixtureHelper));
    const fixtureGolden=path.join(root,BIDI_GOLDEN);
    if(fs.existsSync(fixtureGolden))callers.push(identity(fixtureGolden));
    callers.push(identity(wrapper));
    const sources=walk(path.join(root,'tests/stdlib')).sort((a,b)=>a.path.localeCompare(b.path));
    fs.mkdirSync(out,{mode:0o700});
    put(path.join(out,'capture.json'),{schema:'sounio.stdlib.e2e.diagnostics.v1',output:out,root,started_at_utc:utc(),identity_scope:'Test source and caller/wrapper file identities; not an execve witness or imported-source closure. Compile and native execution have separate bounded phase receipts.',callers,sources,argv_inputs:sources.map(s=>argvSnapshot(root,path.relative(root,s.path)))});
  } else if(op==='begin') {
    const c=control(), [rel,name,...command]=args, dir=phase(c,rel,name);
    c.callers.forEach(same); c.sources.filter(x=>x.path===path.join(c.root,rel)).forEach(same);
    c.argv_inputs.filter(x=>x.test_path===rel).forEach(x=>verifyArgvSnapshot(c.root,x));
    createBelowOutput(dir); ancestors(dir);
    const binding=fixtureBinding(c,rel,name,command);
    put(path.join(dir,'start.json'),{test_path:rel,phase:name,command,started_at_utc:utc(),source:identity(path.join(c.root,rel)),...(binding?{runtime_fixture:binding}:{})});
    process.stdout.write(path.join(dir,'combined.output'));
  } else if(op==='finish') {
    const c=control(), [rel,name,rc]=args, dir=phase(c,rel,name); ancestors(dir);
    if(!/^(0|[1-9][0-9]{0,2})$/.test(rc)||Number(rc)>255) throw Error('invalid exit status');
    const start=read(path.join(dir,'start.json')); regular(path.join(dir,'combined.output')); same(start.source); c.callers.forEach(same);
    const summary=fixtureSummary(c,start,Number(rc));
    put(path.join(dir,'result.json'),{...start,completed_at_utc:utc(),exit_code:Number(rc),combined_output:identity(path.join(dir,'combined.output')),...(summary?{runtime_fixture_artifacts:summary}:{})});
  } else if(op==='verify') {
    const c=control(),[rel,name,rc]=args,dir=phase(c,rel,name); ancestors(dir);
    const result=verifyPhaseCapture(c,rel,name);if(result.exit_code!==Number(rc))throw Error('missing/mismatched completed capture');
  } else if(op==='finalize') {
    const c=control(); c.callers.forEach(same); c.sources.forEach(same);
    const argv_inputs_after=c.argv_inputs.map(x=>argvSnapshot(c.root,x.test_path));
    put(path.join(out,'argv-inputs.after.json'),argv_inputs_after);
    if(JSON.stringify(argv_inputs_after)!==JSON.stringify(c.argv_inputs))throw Error('argv input changed during capture');
    const now=walk(path.join(c.root,'tests/stdlib')).sort((a,b)=>a.path.localeCompare(b.path));
    if(JSON.stringify(now)!==JSON.stringify(c.sources)) throw Error('test source census changed');
    const captured_phases=verifyAllCaptures(c);
    put(path.join(out,'complete.json'),{schema:c.schema,captured_phases,completed_at_utc:utc(),source_and_caller_identity_unchanged:true,argv_inputs_after,argv_inputs_unchanged:true,selection_and_results:'See unchanged schema-v1 E2E result; capture is supplemental operational metadata.'});
  } else throw Error(`unknown operation: ${op}`);
} catch(e) { console.error(`error: E2E artifact capture: ${e.message}`); process.exitCode=65; }
