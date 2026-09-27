#!/usr/bin/env node
import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync, execFileSync} from 'node:child_process';
import {overlaps, normalize, conflicts, parse, check, covers} from '../dev/sounio_coord_github.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'coord-github-'));
process.on('exit', () => fs.rmSync(temp, {recursive:true, force:true}));
const git = (...args) => execFileSync('git', args, {cwd:temp, encoding:'utf8'}).trim();
git('init','-q'); git('config','user.name','Coord Test'); git('config','user.email','coord@example.invalid');
fs.writeFileSync(path.join(temp,'seed'),'seed'); git('add','seed'); git('commit','-qm','seed');
git('remote','add','origin','https://github.com/Sounio-lang/sounio.git');
const head = git('rev-parse','HEAD');
const bin = path.join(temp,'fake-bin'); fs.mkdirSync(bin);
const fixture = path.join(temp,'fixture.json');
fs.writeFileSync(path.join(bin,'gh'), `#!/usr/bin/env node
const fs=require('node:fs'); const f=JSON.parse(fs.readFileSync(process.env.COORD_FIXTURE));
const args=process.argv.slice(2); let data;
if(f.patchFail && args.includes('-H')) {process.stderr.write('diff too large');process.exit(1);}
if(f.fail) {process.stderr.write('API unavailable'); process.exit(1);}
if(args.includes('graphql')) {
 const query=args.find(a=>a.startsWith('query='));
 if(query.includes('pullRequests')) data={data:{repository:{defaultBranchRef:{name:'main',target:{oid:'base'}},pullRequests:{nodes:(f.prPages ? f.prPages[args.includes('cursor=next')?1:0] : f.prs)||[],pageInfo:{hasNextPage:!!f.prPages&&!args.includes('cursor=next'),endCursor:'next'}}}}};
 else data={data:{repository:{refs:{nodes:(f.refPages ? f.refPages[args.includes('cursor=next')?1:0] : f.refs)||[],pageInfo:{hasNextPage:!!f.refPages&&!args.includes('cursor=next'),endCursor:'next'}}}}};
} else if(args.some(a=>a.includes('/compare/'))) data=f.compare||{files:[]};
else if(args.some(a=>a.includes('/files'))) { const p=Number(args[1].match(/page=(\\d+)$/)[1]); data=(f.restFiles||[]).slice((p-1)*100,p*100); }
else if(args.some(a=>a.includes('state=closed'))) data=[];
else {process.stdout.write(f.patch||''); process.exit(0);}
process.stdout.write(JSON.stringify(data));
`, {mode:0o755});
const pr = (files=['self-hosted/ir/lower.sio']) => ({number:2708,title:'carrier',url:'https://github.com/Sounio-lang/sounio/pull/2708',headRefName:'carrier',headRefOid:'a'.repeat(40),headRepository:{nameWithOwner:'Sounio-lang/sounio'},body:'',files:{nodes:files.map(path=>({path,changeType:'MODIFIED'})),pageInfo:{hasNextPage:false}}});
function invoke(f, args, {state='state', cli=false}={}) {
 fs.writeFileSync(fixture, JSON.stringify(f));
 const env={...process.env, PATH:`${bin}:${process.env.PATH}`,COORD_FIXTURE:fixture,SOUNIO_COORD_DIR:path.join(temp,state)};
 return spawnSync(cli?'bash':process.execPath, [cli?path.join(root,'bin/sounio-coord'):path.join(root,'scripts/dev/sounio_coord_github.mjs'),...args],{cwd:temp,env,encoding:'utf8'});
}
function result(f,args,code=0,opts) {const r=invoke(f,args,opts); assert.equal(r.status,code,r.stdout+r.stderr);return r.stdout+r.stderr;}
const files=['--files','self-hosted/ir/lower.sio'];
test('path normalization and conservative globs',()=>{
 assert.equal(normalize('./self-hosted/ir/../ir/lower.sio',temp),'self-hosted/ir/lower.sio');
 assert.throws(()=>normalize('../escape',temp));
 for(const scope of ['self-hosted','self-hosted/**','self-hosted/ir/*.sio','self-hosted/ir/low*.sio','.']) assert.ok(overlaps(scope,'self-hosted/ir/lower.sio'),scope);
 assert.ok(!overlaps('stdlib/math','stdlib/maths/a.sio'));
});
test('empty remote succeeds, outage blocks',()=>{result({},files);result({fail:true},files,3);});
test('all open PRs block independent of local store',()=>{
 for(const state of ['machine-a','machine-b']) assert.match(result({prs:[pr()]},['claim','--agent','a','--lane','x','--intent','test',...files],3,{state,cli:true}),/REMOTE_OVERLAP pr:2708/);
});
test('disjoint claim succeeds and existing local collision remains',()=>{
 result({prs:[pr(['other'])]},['claim','--agent','a','--lane','ok','--intent','test',...files],0,{cli:true});
 result({},['claim','--agent','b','--lane','collision','--intent','test',...files],1,{cli:true});
});
test('scope refuses absent or expanded receipt, accepts fresh full preflight',()=>{
 result({},['scope','--agent','a','--lane','hook','--intent','test',...files],3,{state:'scope',cli:true});
 result({},['remote-check',...files],0,{state:'scope',cli:true});
 result({fail:true},['scope','--agent','a','--lane','hook','--intent','test',...files],0,{state:'scope',cli:true});
 result({},['scope','--agent','a','--lane','hook','--intent','test','--files','new-file'],3,{state:'scope',cli:true});
});
test('expired and wrong-head receipts cannot grant writes',()=>{
 for(const field of ['time','head']) {
  const dir=path.join(temp,'stale', 'remote-receipts');
  result({},['remote-check',...files],0,{state:'stale',cli:true});
  const p=path.join(dir,fs.readdirSync(dir)[0]); const value=JSON.parse(fs.readFileSync(p));
  value[field]=field==='time'?0:'wrong'; fs.writeFileSync(p,JSON.stringify(value));
  result({},['scope','--agent','a','--lane','stale','--intent','test',...files],3,{state:'stale',cli:true});
 }
});
test('review exception requires exact current head and reason',()=>{
 result({prs:[pr()]},['--reviewed',`pr:2708@${'a'.repeat(40)}`,'--review-reason','scoped handoff',...files]);
 result({prs:[pr()]},['--reviewed','pr:2708@old','--review-reason','handoff',...files],3);
 assert.throws(()=>parse(['--reviewed','pr:1@x']));
});
test('same branch in a fork is not an own carrier',()=>{
 const p=pr();p.headRefName=git('branch','--show-current');p.headRefOid=head;
 result({prs:[p]},files);
 p.headRepository.nameWithOwner='somebody/sounio';result({prs:[p]},files,3);
});
test('declared future write sets and symbols conflict before implementation',()=>{
 const p=pr(['other']);p.body='Sounio-Coord-Files: self-hosted/ir/**\nSounio-Coord-Symbols: lower_let_stmt';
 result({prs:[p]},files,3);
 assert.deepEqual(conflicts({...p,files:['other'],trustedDeclarations:true},[],['lower_let_stmt']),['symbol:lower_let_stmt']);
 result({prs:[pr(['other'])],patch:'@@ fn lower_let_stmt()'},['--symbol','lower_let_stmt',...files],3);
});
test('paginated PR files and renamed source path are checked',()=>{
 const p=pr(['other']);p.files.pageInfo.hasNextPage=true;
 const restFiles=Array.from({length:100},(_,i)=>({filename:`unrelated/${i}`}));restFiles.push({filename:'self-hosted/ir/lower.sio'});
 result({prs:[p],restFiles},files,3);
 p.files.pageInfo.hasNextPage=false;p.files.nodes[0].changeType='RENAMED';
 result({prs:[p],restFiles:[{filename:'other',previous_filename:'self-hosted/ir/lower.sio'}]},files,3);
});
test('recent branch without a PR blocks; old branch is outside discovery window',()=>{
 const ref={name:'unpublished',target:{oid:'b'.repeat(40),committedDate:new Date().toISOString()}};
 result({refs:[ref],compare:{files:[{filename:'self-hosted/ir/lower.sio'}]}},files,3);
 ref.target.committedDate='2000-01-01T00:00:00Z';result({refs:[ref]},files);
});
test('truncated branch diff cannot silently pass',()=>{
 const refs=[{name:'wide',target:{oid:'b'.repeat(40),committedDate:new Date().toISOString()}}];
 result({refs,compare:{files:Array.from({length:300},()=>({filename:'other'}))}},files,3);
});
test('brief is remote inventory, never permission',()=>{assert.match(result({prs:[pr()]},['--brief']),/INVENTORY_ONLY/);});
test('failed refresh revokes previous receipt',()=>{
 result({},['remote-check',...files],0,{state:'revoke',cli:true});
 result({fail:true},['remote-check',...files],3,{state:'revoke',cli:true});
 result({},['scope','--agent','a','--lane','revoke','--intent','test',...files],3,{state:'revoke',cli:true});
});
test('large PR cap is explicit, head-pinned and not a silent pass',()=>{
 const p=pr(['other']);p.changedFiles=3001;
 assert.match(result({prs:[p]},files,3),/incomplete-diff/);
 result({prs:[p]},['--reviewed',`pr:2708@${p.headRefOid}`,'--review-reason','full external review recorded',...files]);
});
test('invalid remote branch timestamp fails closed',()=>{
 result({refs:[{name:'unknown',target:{oid:'abc',committedDate:'invalid'}}]},files,3);
});

test('PR and branch inventory pagination sees conflicts beyond the first page',()=>{
 result({prPages:[[],[pr()]]},files,3);
 result({refPages:[[],[{name:'page-two',target:{oid:'c'.repeat(40),committedDate:new Date().toISOString()}}]],compare:{files:[{filename:'self-hosted/ir/lower.sio'}]}},files,3);
});

test('branch head advanced after PR inventory is checked separately',()=>{
 const p=pr(['other']);
 result({prs:[p],refs:[{name:'carrier',target:{oid:'b'.repeat(40),committedDate:new Date().toISOString()}}],compare:{files:[{filename:'self-hosted/ir/lower.sio'}]}},files,3);
});

test('fork-only origin cannot silently scan an isolated GitHub store',()=>{
 const execute=(command,args)=>args.includes('--show-toplevel')?temp:args.includes('get-url')?'https://github.com/other/sounio.git':'origin';
 assert.throws(()=>check(parse(files),execute),/canonical Sounio-lang/);
});

test('fork body declarations cannot manufacture ownership; changed files still count',()=>{
 const p=pr(['other']);p.headRepository.nameWithOwner='untrusted/sounio';
 p.body='Sounio-Coord-Files: .\nSounio-Coord-Symbols: lower_let_stmt';
 result({prs:[p]},['--symbol','lower_let_stmt',...files]);
 p.files.nodes=[{path:'self-hosted/ir/lower.sio',changeType:'MODIFIED'}];result({prs:[p]},files,3);
});
test('directory/glob receipts cover child writes, never the reverse',()=>{
 assert.ok(covers('self-hosted/ir/**','self-hosted/ir/lower.sio'));
 assert.ok(!covers('self-hosted/ir/lower.sio','self-hosted/ir/**'));
 result({},['remote-check','--files','self-hosted/ir/**'],0,{state:'glob',cli:true});
 result({},['scope','--agent','a','--lane','glob','--intent','test',...files],0,{state:'glob',cli:true});
 result({},['remote-check',...files],0,{state:'narrow',cli:true});
 result({},['scope','--agent','a','--lane','narrow','--intent','test','--files','self-hosted/ir/**'],3,{state:'narrow',cli:true});
});
test('exact reviewed wide PR skips unavailable symbol diff; stale review does not',()=>{
 const p=pr(['other']);p.changedFiles=3728;
 result({prs:[p],patchFail:true},['--symbol','lower_let_stmt','--reviewed',`pr:2708@${p.headRefOid}`,'--review-reason','full external review',...files]);
 result({prs:[p],patchFail:true},['--symbol','lower_let_stmt','--reviewed','pr:2708@old','--review-reason','old review',...files],3);
});
