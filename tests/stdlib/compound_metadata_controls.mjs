// Metadata and invocation controls; qualification belongs on Slurm.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {compoundGates,validateCompoundReceipt,runCompoundGates} from '../../scripts/stdlib/compound_gates.mjs';
import {createCompilerBinding,compilerEnvironment,verifyCompilerBinding} from '../../scripts/stdlib/compiler_binding.mjs';
const gate=compoundGates[0],sha='a'.repeat(64),compiler={path:'/bound/compiler',sha256:sha};
const inputs=gate.members.map(path=>({path,sha256:sha}));
const valid={code:0,signal:null,error:null,status:'0\n',stdout:gate.marker+'\n',manifest:sha+'  '+gate.members[0]+'\n',compilerManifest:sha+'  '+compiler.path+'\n'};
validateCompoundReceipt(gate,valid,inputs,compiler);
for(const [name,patch] of [
 ['producer-nonzero',{code:1}],['producer-signal',{signal:'SIGTERM'}],['spawn-error',{error:'failed'}],
 ['status-absent',{status:''}],['status-nonzero',{status:'1'}],
 ['marker-missing',{stdout:''}],['marker-duplicate',{stdout:valid.stdout+valid.stdout}],
 ['manifest-malformed',{manifest:'bad'}],['member-missing',{manifest:''}],
 ['member-digest-wrong',{manifest:'b'.repeat(64)+'  '+gate.members[0]+'\n'}],
 ['compiler-missing',{compilerManifest:''}],['compiler-digest-wrong',{compilerManifest:'b'.repeat(64)+'  '+compiler.path+'\n'}],
 ['compiler-path-wrong',{compilerManifest:sha+'  /other/compiler\n'}],['compiler-duplicate',{compilerManifest:valid.compilerManifest+valid.compilerManifest}]
]){assert.throws(()=>validateCompoundReceipt(gate,{...valid,...patch},inputs,compiler),undefined,name);console.log('CONTROL_REFUSED '+name);}
assert.throws(()=>validateCompoundReceipt(gate,valid,[],compiler));
assert.throws(()=>validateCompoundReceipt(gate,valid,inputs,null));
const out=process.argv[2];fs.mkdirSync(out);
const quotedRaw=path.join(out,"raw ' alias.elf");
fs.linkSync(process.env.MADAROS_RAW_BIN,quotedRaw);
const binding=createCompilerBinding(quotedRaw,out);
assert.equal(spawnSync('bash',['-n',binding.launcher]).status,0);
let children=0;
await assert.rejects(runCompoundGates({root:process.cwd(),output:path.join(out,'stopped'),results:[],inputs:[],compiler:binding,shouldStop:()=>true,onChild:()=>children++}),/interrupted/);
assert.equal(children,0);
const env=compilerEnvironment(binding,{...process.env,SOUC_BIN:'/wrong/compiler',SOUC_NATIVE_BIN:'/wrong/compiler',SOUNIO_SOUC_BIN:'/wrong/compiler'});
const run=spawnSync('bash',['scripts/stdlib/run_stdlib_e2e.sh','--filter','tests/stdlib/algebra/test_xor_convolution.sio','--json-out',path.join(out,'standalone.json')],{cwd:process.cwd(),env,encoding:'utf8',timeout:180000});
fs.writeFileSync(path.join(out,'standalone.stdout'),run.stdout||'');fs.writeFileSync(path.join(out,'standalone.stderr'),run.stderr||'');
assert.equal(run.status,0);
const result=JSON.parse(fs.readFileSync(path.join(out,'standalone.json'),'utf8'));
assert.equal(result.context.souc_bin,binding.launcher);
assert.equal(result.totals.pass,1);assert.equal(result.totals.total,1);
assert.ok(fs.statSync(path.join(binding.retained_dir,'algebra/test_xor_convolution.sio.elf')).size>0);
verifyCompilerBinding(binding);
fs.appendFileSync(binding.launcher,'# mutation control\n');
assert.throws(()=>verifyCompilerBinding(binding),/changed/);
console.log('COMPOUND_BINDING_AND_INTERRUPT_CONTROLS_PASS');
