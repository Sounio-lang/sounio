import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const oracle=process.env.STDLIB_HYPER_ORACLE;
if(!oracle)throw Error('STDLIB_HYPER_ORACLE must identify the compiled C++23 oracle');
const goldenText=fs.readFileSync(new URL('../../tests/fixtures/hyper/pipeline_golden.v1.json',import.meta.url),'utf8');
const golden=JSON.parse(goldenText);
const directory=fs.mkdtempSync(path.join(os.tmpdir(),'hyper-oracle-controls-'));
const baseline=Object.fromEntries(Object.entries(golden).filter(([,v])=>v.metrics).map(([lane,v])=>[lane,Object.entries(v.metrics).map(([key,value])=>'HYPER_METRIC '+lane+' '+key+' '+String(value)).join('\n')+'\n']));
function run({change=()=>{},golden=goldenText,omit=[]}={}){
 const logs={...baseline};change(logs);
 fs.writeFileSync(path.join(directory,'golden.json'),golden);
 const args=[path.join(directory,'golden.json')];
 for(const[lane,text]of Object.entries(logs)){if(omit.includes(lane))continue;const p=path.join(directory,lane+'.stdout');fs.writeFileSync(p,text);args.push(lane,p);}
 return spawnSync(oracle,args,{encoding:'utf8'});
}
test('exact golden metrics pass',()=>assert.equal(run().status,0));
test('missing metric fails',()=>assert.equal(run({change:x=>x.nn=''}).status,1));
test('duplicate metric fails',()=>assert.equal(run({change:x=>x.nn+='HYPER_METRIC nn sum_w 2\n'}).status,1));
test('outside tolerance fails',()=>assert.equal(run({change:x=>x.nn=x.nn.replace('sum_w 2','sum_w 3')}).status,1));
test('unknown metric fails',()=>assert.equal(run({change:x=>x.nn+='HYPER_METRIC nn unknown 0\n'}).status,1));
test('cross lane metric fails',()=>assert.equal(run({change:x=>x.nn+='HYPER_METRIC onn relu_positive 1\n'}).status,1));
test('NaN token fails',()=>assert.equal(run({change:x=>x.nn=x.nn.replace('sum_w 2','sum_w NaN')}).status,1));
test('numeric suffix fails',()=>assert.equal(run({change:x=>x.nn=x.nn.replace('sum_w 2','sum_w 2oops')}).status,1));
test('missing supplied lane fails',()=>assert.equal(run({omit:['nn']}).status,1));
test('negative tolerance rejected',()=>assert.equal(run({golden:goldenText.replace('"abs": 1e-06','"abs": -1')}).status,2));
test('duplicate golden key rejected',()=>assert.equal(run({golden:goldenText.replace('"sum_w": 2.0','"sum_w": 2.0, "sum_w": 2.0')}).status,2));
test('trailing golden content rejected',()=>assert.equal(run({golden:goldenText+'false'}).status,2));
process.on('exit',()=>fs.rmSync(directory,{recursive:true,force:true}));
