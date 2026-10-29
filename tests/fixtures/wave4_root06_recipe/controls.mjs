// Synthetic metadata/recipe controls. No Sounio/C++ compiler or build recipe runs.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {qualification,layout,publish,explicitError,digest,authorize,verifyPlan} from './metadata.mjs';
import {treeManifest} from '../wave4_variance_shadow/admission.mjs';
if(!process.env.SLURM_JOB_ID)throw Error('Slurm metadata receipt required');
const out=path.resolve(process.argv[2]);fs.mkdirSync(out);
const results=[];
function write(f,s){fs.mkdirSync(path.dirname(f),{recursive:true});fs.writeFileSync(f,s)}
const pin=file=>({file,sha256:digest(file),mode:fs.lstatSync(file).mode&0o7777});
function json(f,v){write(f,JSON.stringify(v,null,2)+'\n');return pin(f)}
function check(name,fn,accept=false){
 let error=null;try{fn()}catch(e){error=e.message}
 if(accept?error:!error)throw Error(name+': unexpected '+(error??'acceptance'));
 results.push({name,expected:accept?'accept':'refuse',observed:error?'refuse':'accept',reason:error});
}
const producer={source:{sha256:'a'.repeat(64)},elf:{sha256:'b'.repeat(64)}};
function receipt(name,role,n){
 const d=path.join(out,name);
 const ids=Array.from({length:n},(_,i)=>name+'-'+String(i).padStart(4,'0'));
 const manifest=json(path.join(d,'cases.json'),{schema:'sounio.qualification.cases.v1',role,ids});
 write(path.join(d,'adapter.txt'),'SYNTHETIC reader fixture, not an actual qualification adapter\n');
 write(path.join(d,'original.txt'),'SYNTHETIC original metadata; no program executed\n');
 const r={schema:'sounio.producer.qualification-adapter.v1',role,completed:true,
 manifest_sha256:manifest.sha256,producer_source_sha256:producer.source.sha256,
 producer_elf_sha256:producer.elf.sha256,failed:0,skipped:0,required:n,passed:n,
 cases:ids.map(id=>({id,status:'pass'})),adapter:pin(path.join(d,'adapter.txt')),
 originals:[pin(path.join(d,'original.txt'))],job_id:'SYNTHETIC_METADATA_ONLY'};
 return {manifest,receipt:json(path.join(d,'receipt.json'),r),r};
}
function change(x,fn){fn(x.r);x.receipt=json(x.receipt.file,x.r)}
for(const[role,n]of [['producer-flow-base104',104],['producer-bounds-base34',34],['producer-metadata-base211',211],['producer-frontiers',4]]){
 const x=receipt(role,role,n);check('exact-'+role,()=>qualification(role,x,producer),true);
}
let x=receipt('rebound','producer-metadata-base211',211);change(x,r=>r.producer_elf_sha256='c'.repeat(64));
check('old211-cannot-rebind-to-new-ELF',()=>qualification('producer-metadata-base211',x,producer));
x=receipt('source','producer-flow-base104',104);change(x,r=>r.producer_source_sha256='c'.repeat(64));
check('wrong-final-producer-source',()=>qualification('producer-flow-base104',x,producer));
x=receipt('missing','producer-flow-base104',103);check('missing-original-case',()=>qualification('producer-flow-base104',x,producer));
x=receipt('duplicate','producer-bounds-base34',34);change(x,r=>r.cases[1].id=r.cases[0].id);
check('duplicate-original-case',()=>qualification('producer-bounds-base34',x,producer));
x=receipt('skip','producer-bounds-base34',34);change(x,r=>{r.skipped=1;r.cases[0].status='skip'});
check('skip-not-pass',()=>qualification('producer-bounds-base34',x,producer));
x=receipt('frontier-empty','producer-frontiers',0);check('frontiers-nonempty',()=>qualification('producer-frontiers',x,producer));
x=receipt('frontier-fail','producer-frontiers',3);change(x,r=>{r.failed=1;r.cases[0].status='fail'});
check('frontier-failed',()=>qualification('producer-frontiers',x,producer));
x=receipt('original-drift','producer-frontiers',2);write(x.r.originals[0].file,'changed');
check('original-receipt-drift',()=>qualification('producer-frontiers',x,producer));
const base={run_root:path.join(out,'run'),repository:path.join(out,'repo'),sdk:[{root:path.join(out,'sdk')}]};
check('private-layout-outside-source',()=>layout(base),true);
check('never-root05',()=>layout({...base,run_root:'/scratch/root-integrated05'}));
check('no-output-under-Git-source',()=>layout({...base,run_root:path.join(base.repository,'run')}));
check('no-output-under-SDK',()=>layout({...base,run_root:path.join(base.sdk[0].root,'run')}));
for(const[name,text,want]of [['error','error: masked compiler failure',true],['coded','error[E038]: failed',true],
 ['drop','would have dropped write',true],['refusal','native refusal',true],['clean','ordinary successful output',false]]){
 if(explicitError(text)!==want)throw Error('diagnostic '+name);
 results.push({name:'diagnostic-'+name,expected:want?'refuse':'accept',observed:want?'refuse':'accept'});
}
const pub=path.join(out,'SYNTHETIC-PUBLICATION');fs.mkdirSync(pub);
const elf=path.join(pub,'metadata-only.elf');write(elf,Buffer.from([0x7f,0x45,0x4c,0x46]));fs.chmodSync(elf,0o755);
const guard={schema:'sounio.root06.validated-plan.v1',status:0,phase:'final',head:'1'.repeat(40),tree:'2'.repeat(40),producer_elf_sha256:'3'.repeat(64),build_manifest_sha256:'4'.repeat(64),layout:{R:pub}};
check('no-ready-before-EXIT-guards',()=>publish(pub,'build',{...guard,phase:'before'},0,elf));
check('no-ready-on-guard-failure',()=>publish(pub,'build',{...guard,status:1},0,elf));
check('no-ready-on-work-failure',()=>publish(pub,'build',guard,1,elf));
if(fs.existsSync(path.join(pub,'RAW_READY')))throw Error('failure published readiness');
check('metadata-publication-after-final-guards',()=>publish(pub,'build',guard,0,elf),true);
check('refuse-existing-readiness',()=>publish(pub,'build',guard,0,elf));
const planFile=path.join(out,'SYNTHETIC-plan.json');write(planFile,'{"synthetic":true}\n');
let auth=json(path.join(out,'auth.json'),{authorized_by:'root',scope:'ROOT06_STAGING_PREFLIGHT_BUILD',plan_sha256:digest(planFile)});
check('manual-auth-record-binding',()=>authorize(planFile,auth),true);
write(planFile,'{"synthetic":"changed"}\n');check('plan-change-revokes-record',()=>authorize(planFile,auth));
// One real Git metadata model validates HEAD/tree/full archive census without executing a producer.
const model=path.join(out,'GIT-METADATA-MODEL'),repo=path.join(model,'repo');fs.mkdirSync(repo,{recursive:true});
const git=process.env.GIT_BIN??'/usr/bin/git';
function command(args,options={}){const r=spawnSync(git,args,{encoding:'utf8',...options});if(r.status!==0)throw Error('Git metadata command failed');return r.stdout}
write(path.join(repo,'self-hosted/compiler/lean_single.sio'),'SYNTHETIC metadata producer source\n');
write(path.join(repo,'self-hosted/compiler/main.sio'),'use parser::ast::*\nfn main() with IO, Mut, Panic, Div, Alloc {\n}\n');
write(path.join(repo,'docs/README'),'full docs input\n');write(path.join(repo,'tools/tool.txt'),'full tools input\n');
fs.symlinkSync('missing-target',path.join(repo,'docs/Lean'));
command(['init','-q',repo]);command(['-C',repo,'add','.']);
command(['-C',repo,'-c','user.name=MetadataControl','-c','user.email=metadata@invalid','commit','-qm','synthetic metadata only']);
const head=command(['-C',repo,'rev-parse','HEAD']).trim(),tree=command(['-C',repo,'rev-parse','HEAD^{tree}']).trim();
const archive=path.join(model,'source.tar');command(['-C',repo,'archive','--format=tar','-o',archive,head]);
const census=path.join(model,'census.z');write(census,spawnSync(git,['-C',repo,'ls-tree','-rz','--full-tree',head]).stdout);
const sdk=path.join(model,'private-sdk');write(path.join(sdk,'only-metadata.txt'),'SYNTHETIC SDK manifest fixture\n');
fs.symlinkSync('missing-library',path.join(sdk,'dangling'));
const sdkManifest=json(path.join(model,'SDK.json'),treeManifest(sdk));
const expectedOrigin=path.join(model,'synthetic-archive-extraction');fs.mkdirSync(expectedOrigin);
const extracted=spawnSync('/bin/tar',['-xf',archive,'-C',expectedOrigin]);if(extracted.status!==0)throw Error('synthetic Git archive extraction failed');
const originManifest=json(path.join(model,'SOURCE.json'),treeManifest(expectedOrigin));
const src=pin(path.join(repo,'self-hosted/compiler/lean_single.sio'));
const fakeELF=path.join(model,'metadata-only-producer.elf');write(fakeELF,Buffer.from([0x7f,0x45,0x4c,0x46]));fs.chmodSync(fakeELF,0o755);
const originalProducer={source:{...producer.source},elf:{...producer.elf}};
producer.source=src;producer.elf=pin(fakeELF);
const qualifications={};
for(const[role,n]of [['producer-flow-base104',104],['producer-bounds-base34',34],['producer-metadata-base211',211],['producer-frontiers',2]])qualifications[role]=receipt('model-'+role,role,n);
const here=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(here,'../../..');
const workerPatch=path.join(model,'worker.patch');write(workerPatch,'+++ b/self-hosted/compiler/main.sio\n');
const preflight=path.join(model,'preflight.sh');write(preflight,'# SYNTHETIC METADATA, NEVER EXECUTED\n');
const env=path.join(model,'env.sh');write(env,'# SYNTHETIC ENV, NEVER SOURCED\n');
const cases=json(path.join(model,'preflight-cases.json'),{schema:'sounio.root06.preflight-cases.v1',cases:[]});
const p={schema:'sounio.root06.recipe.v1',repository:repo,head,tree,run_root:path.join(model,'future-run'),producer,qualifications,
 build_seconds:100,preflight_seconds:10,tools:{node:pin(process.execPath),bash:pin('/bin/bash'),git:pin(git),tar:pin('/bin/tar'),time:pin('/bin/bash'),timeout:pin('/bin/bash'),cxx:pin('/bin/bash')},
 tool_environment:{file:pin(env),context_R:sdk},sdk:[{root:sdk,manifest:sdkManifest}],pristine_manifest:originManifest,build_manifest:originManifest,
 inputs:{archive:pin(archive),git_census:pin(census),capacity:pin(process.env.CAPACITY_PATCH),
 worker_optin:pin(workerPatch),preflight_recipe:pin(preflight),preflight_cases:cases,
 magic:pin(path.join(sdk,'only-metadata.txt')),metadata_helper:pin(path.join(here,'metadata.mjs')),graph_helper:pin(path.join(here,'../wave4_variance_shadow/admission.mjs')),
 build_template:pin(path.join(root,'scripts/stdlib/root06_build_template.sh'))}};
check('complete-Git-review-metadata-only',()=>verifyPlan(p,'review'),true);
check('wrong-actual-HEAD',()=>verifyPlan({...p,head:'0'.repeat(40)},'review'));
const oldCensus=fs.readFileSync(census);write(census,oldCensus.subarray(1));p.inputs.git_census=pin(census);
check('partial-Git-census-refused',()=>verifyPlan(p,'review'));write(census,oldCensus);p.inputs.git_census=pin(census);
check('non-eaa-capacity-refused',()=>verifyPlan({...p,inputs:{...p.inputs,capacity:pin(workerPatch)}},'review'));
if(fs.existsSync(p.run_root))throw Error('review created run root');
const L=layout(p);
fs.mkdirSync(p.run_root);
fs.cpSync(expectedOrigin,L.S_GIT,{recursive:true,verbatimSymlinks:true});
fs.cpSync(L.S_GIT,L.S,{recursive:true,verbatimSymlinks:true});
check('full-Git-copy-before-overlay',()=>verifyPlan(p,'copy'),true);
const relativeLink=path.join(L.S,'docs/Lean');
if(fs.readlinkSync(relativeLink)!=='missing-target')throw Error('relative link not verbatim');
fs.unlinkSync(relativeLink);fs.symlinkSync(path.join(expectedOrigin,'docs/missing-target'),relativeLink);
check('copy-link-absolutization-refused',()=>verifyPlan(p,'copy'));
fs.unlinkSync(relativeLink);fs.symlinkSync('missing-target',relativeLink);
const fileMode=fs.statSync(path.join(L.S,'docs/README')).mode&0o7777;
fs.chmodSync(path.join(L.S,'docs/README'),fileMode^0o100);
check('copy-mode-drift-refused',()=>verifyPlan(p,'copy'));
const privateInput=path.join(model,'mode-pin.txt');write(privateInput,'same bytes\n');
const modePin=pin(privateInput);fs.chmodSync(privateInput,modePin.mode^0o100);
const savedAdapter=qualifications['producer-frontiers'].r.adapter;
qualifications['producer-frontiers'].r.adapter=modePin;
qualifications['producer-frontiers'].receipt=json(qualifications['producer-frontiers'].receipt.file,qualifications['producer-frontiers'].r);
check('private-input-mode-drift',()=>verifyPlan(p,'review'));
qualifications['producer-frontiers'].r.adapter=savedAdapter;
qualifications['producer-frontiers'].receipt=json(qualifications['producer-frontiers'].receipt.file,qualifications['producer-frontiers'].r);
const sdkFile=path.join(sdk,'only-metadata.txt');write(sdkFile,'changed SDK\n');
check('private-SDK-mutation',()=>verifyPlan(p,'review'));
if(fs.existsSync(path.join(p.run_root,'RAW_READY')))throw Error('copy metadata created actual build readiness');
fs.writeFileSync(path.join(out,'RESULT.json'),JSON.stringify({status:'PASS',scope:'synthetic-metadata-and-static-template-only',compiler_invocations:0,real_source_seals:0,build_submissions:0,controls:results},null,2)+'\n');
console.log('ROOT06_RECIPE_METADATA_CONTROLS_OK '+results.length);
