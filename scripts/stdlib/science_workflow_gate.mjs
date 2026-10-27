#!/usr/bin/env node
// Operational orchestration only. Numerical/artifact acceptance belongs to C++23.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
const options={};
for(let i=2;i<process.argv.length;i+=2){const key=process.argv[i];if(!['--out-dir','--producer','--checker','--reference'].includes(key)||!process.argv[i+1])throw Error('invalid arguments');options[key]=process.argv[i+1];}
for(const key of ['--out-dir','--producer','--checker','--reference'])assert(path.isAbsolute(options[key]||''),'absolute paths required');
const out=options['--out-dir'];
fs.mkdirSync(out);
function run(name,command,args=[]){
 const result=spawnSync(command,args,{cwd:out,env:process.env,timeout:180000,maxBuffer:16*1024*1024});
 fs.writeFileSync(path.join(out,name+'.stdout'),result.stdout||'');
 fs.writeFileSync(path.join(out,name+'.stderr'),result.stderr||'');
 assert.equal(result.status,0,name+' failed: '+(result.error?.message||result.signal||result.status));
 return new TextDecoder('utf-8',{fatal:true}).decode(result.stdout);
}
function section(text,name){
 const begin=name+'_BEGIN\n',end=name+'_END\n',start=text.indexOf(begin),stop=text.indexOf(end);
 assert(start>=0&&stop>start&&text.lastIndexOf(begin)===start&&text.lastIndexOf(end)===stop,'missing/duplicate '+name);
 return text.slice(start+begin.length,stop).trim()+'\n';
}
try{
 run('reference-fixture',options['--reference'],['--fixture']);
 const stdout=run('producer',options['--producer']);
 assert(stdout.includes('SCIENCE_WORKFLOW_OK\n'));
 const report=section(stdout,'SCIENCE_REPORT');
 assert.equal(JSON.parse(report).schema,'sounio.stdlib.science-workflow.v1');
 fs.writeFileSync(path.join(out,'workflow-report.json'),report);
 fs.writeFileSync(path.join(out,'workflow.svg'),section(stdout,'SCIENCE_SVG'));
 run('json-checker',options['--checker']);
 assert.equal(fs.readFileSync(path.join(out,'workflow-report-replayed.json'),'utf8').trim(),report.trim());
 run('cpp-verifier',options['--reference'],['workflow-report.json','workflow.svg','workflow.png']);
 const files=['workflow-input.csv','workflow-report.json','workflow-report-replayed.json','workflow.svg','workflow.png','producer.stdout'];
 fs.writeFileSync(path.join(out,'ARTIFACTS.sha256'),files.map(name=>crypto.createHash('sha256').update(fs.readFileSync(path.join(out,name))).digest('hex')+'  '+name).join('\n')+'\n');
 fs.writeFileSync(path.join(out,'STATUS.json'),JSON.stringify({schema:'sounio.stdlib.science-workflow.gate.v1',status:'pass',files},null,2)+'\n');
 console.log('SCIENCE_WORKFLOW_GATE_OK');
}catch(error){
 fs.writeFileSync(path.join(out,'STATUS.json'),JSON.stringify({schema:'sounio.stdlib.science-workflow.gate.v1',status:'fail',error:error.message},null,2)+'\n');
 throw error;
}
