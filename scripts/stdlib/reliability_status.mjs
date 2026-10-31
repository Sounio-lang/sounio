#!/usr/bin/env node
// Aggregates independent operational receipts. No numerical science is done here.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
function contractAdjustments(root) {
 const rows=[];
 function visit(directory) {
  if(!fs.existsSync(directory))return;
  for(const entry of fs.readdirSync(directory,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name))){
   const full=path.join(directory,entry.name);
   if(entry.isDirectory())visit(full);
   else if(entry.isFile()&&entry.name.endsWith('.sio')){
    const match=new RegExp("^\\s*//@\\s*contract-adjustment:\\s*(.+?)\\s*$",'m').exec(fs.readFileSync(full,'utf8'));
    if(match)rows.push({test_path:path.relative(root,full).split(path.sep).join('/'),adjustment:match[1]});
   }
  }
 }
 visit(path.join(root,'tests/stdlib'));
 return rows;
}
export function aggregate(root, paths, codes, strict=false) {
  const notes=[], read=(p,schema)=>{
    try{const j=JSON.parse(fs.readFileSync(p,'utf8'));if(j.schema!==schema)throw Error('schema');return j;}
    catch(e){notes.push('invalid_or_missing:'+p+':'+e.message);return null;}
  };
  const inv=read(paths.inventory,'sounio.stdlib.inventory.v1');
  const hyper=read(paths.hyper,'sounio.stdlib.hyper_execution_status.v1');
  const science=read(paths.science,'sounio.stdlib.science_pipeline_status.v1');
  const e2e=read(paths.e2e,'sounio.stdlib.e2e.result.v1');
  const validTotals=t=>t&&['pass','fail','skip','total'].every(k=>Number.isSafeInteger(t[k])&&t[k]>=0)&&t.pass+t.fail+t.skip===t.total;
  const empty={pass:0,fail:0,skip:0,total:0};
  const totals=validTotals(e2e?.totals)?e2e.totals:empty;
  const hyperTotals=validTotals(hyper?.totals)?hyper.totals:empty;
  let complete=!!inv&&!!hyper&&!!science&&!!e2e;
  if(!validTotals(e2e?.totals)||totals.total===0){complete=false;notes.push('e2e_empty_or_invalid_selection');}
  if(!validTotals(hyper?.totals)){complete=false;notes.push('hyper_invalid_totals');}
  if(!Array.isArray(e2e?.results)||e2e.results.length!==totals.total){complete=false;notes.push('e2e_row_count_mismatch');}
  if(Array.isArray(e2e?.results)) {
    const seen=new Set(), actual={pass:0,fail:0,skip:0};
    for(const r of e2e.results) {
      if(!r||!Object.hasOwn(actual,r.status)||typeof r.test_path!=='string'||seen.has(r.test_path)){complete=false;notes.push('invalid_or_duplicate_e2e_row');continue;}
      seen.add(r.test_path);actual[r.status]++;
    }
    if(['pass','fail','skip'].some(k=>actual[k]!==totals[k])){complete=false;notes.push('e2e_status_count_mismatch');}
  }
  const ic=inv?.counts||{};
  if(!['sio_files','disabled_files','stub_mod_files','active_module_entrypoints'].every(k=>Number.isSafeInteger(ic[k])&&ic[k]>=0)||!(ic.sio_files>0)){complete=false;notes.push('inventory_invalid_or_empty');}
  const scienceStatus=['pass','fail','not_run'].includes(science?.status_summary)?science.status_summary:'not_run';
  const hyperStatus=['pass','fail','not_run'].includes(hyper?.status_summary)?hyper.status_summary:'not_run';
  const runtime=science?.runtime_regression_summary||{}, provenance=science?.runtime_provenance||{};
  if(!['pass','fail','not_run'].includes(runtime.status)||!Number.isSafeInteger(runtime.fail)||runtime.fail<0||
    !['soft','strict'].includes(science?.runtime_regression_enforcement)||
    !['souc_bin','souc_version','pinned_version_expected'].every(k=>typeof provenance[k]==='string')){complete=false;notes.push('science_runtime_receipt_invalid');}
  const exitsOK=['inventory','hyper','e2e'].every(k=>codes[k]===0);
  const passed=complete&&exitsOK&&totals.fail===0&&hyperTotals.fail===0&&scienceStatus==='pass'&&hyperStatus==='pass'&&
    (!strict||(science.runtime_regression_enforcement==='strict'&&runtime.status==='pass'&&runtime.fail===0));
  for(const[k,v]of Object.entries(codes))notes.push(k+'_exit_code='+v);
  const rel=p=>path.relative(root,p).startsWith('..')?p:path.relative(root,p);
  return {schema:'sounio.stdlib.reliability_status.v1',generated_at_utc:new Date().toISOString(),
    command:'bash scripts/stdlib_reliability_gate.sh',totals,status_summary:passed?'pass':complete?'fail':'not_run',
    failures:(e2e?.results||[]).filter(r=>r.status==='fail').map(r=>({test_path:r.test_path,category:r.reason||'other',error_excerpt:(r.detail_excerpt||'').slice(0,400)})),
    ignored:e2e?.ignored||[],contract_adjustments:contractAdjustments(root),inventory:{sio_files:ic.sio_files||0,disabled_files:ic.disabled_files||0,
      stub_mod_files:ic.stub_mod_files||0,active_module_entrypoints:ic.active_module_entrypoints||0},
    science_pipeline:{status:scienceStatus,status_artifact:rel(paths.science),runtime_regression_enforcement:science?.runtime_regression_enforcement||'soft',
      runtime_regression_summary_status:runtime.status||'not_run',runtime_regression_fail_count:runtime.fail||0,
      runtime_souc_bin:provenance.souc_bin||'',runtime_souc_version:provenance.souc_version||'',runtime_pinned_version_expected:provenance.pinned_version_expected||''},
    hyper_pipeline:{status:hyperStatus,status_artifact:rel(paths.hyper),totals:hyperTotals},
    notes};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  try {
    const [root,e2e,inventory,out,hyper,science,scanRC,hyperRC,e2eRC]=process.argv.slice(2);
    if(!root||!e2eRC)throw Error('missing aggregate arguments');
    const doc=aggregate(root,{e2e,inventory,hyper,science},{inventory:Number(scanRC),hyper:Number(hyperRC),e2e:Number(e2eRC)},process.env.STDLIB_RUNTIME_REGRESSION_STRICT==='1');
    fs.mkdirSync(path.dirname(path.resolve(out)),{recursive:true});fs.writeFileSync(out,JSON.stringify(doc,null,2)+'\n');
    console.log('STDLIB_RELIABILITY_GATE_'+doc.status_summary.toUpperCase());
    process.exitCode=doc.status_summary==='pass'?0:1;
  }catch(e){console.error(e.message);process.exitCode=1;}
}
