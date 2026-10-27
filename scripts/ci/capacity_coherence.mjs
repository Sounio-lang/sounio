#!/usr/bin/env node
// Source metadata audit only; runtime readback is a separate gate.
// Nonoptional opt_cleanup register peels require native before/after controls;
// skipping peels has produced wrong answers and is not certified here.
import fs from 'node:fs';
import path from 'node:path';
const [mode,source,expectedRaw,root]=process.argv.slice(2);
const prefix=mode==='ir'?'IRFUNCTION_INSTR_CAPACITY':'MIR_INSTR_CAPACITY';
const fail=reason=>{throw Error(reason);};
function unique(pattern,subject,reason) {
 const matches=[...subject.matchAll(new RegExp(pattern.source,pattern.flags.includes('g')?pattern.flags:pattern.flags+'g'))];
 if(matches.length!==1)fail(reason+'_count_'+matches.length);
 return matches[0];
}
try {
 if(!['ir','mir'].includes(mode)||!/^[0-9]+$/.test(expectedRaw||''))fail('invalid_arguments');
 const expected=Number(expectedRaw);if(!Number.isSafeInteger(expected))fail('invalid_expected_capacity');
 const text=fs.readFileSync(source,'utf8');
 if(mode==='ir') {
  const cap=Number(unique(/^pub let IR_MAX_INSTRS: i64 = ([0-9]+)\s*$/m,text,'ir_max_instrs')[1]);
  if(cap!==expected)fail('ir_max_instrs_expected_'+expected+'_got_'+cap);
  const max_funcs=Number(unique(/^pub let IR_MAX_FUNCS: i64 = ([0-9]+)\s*$/m,text,'ir_max_funcs')[1]);
  const max_strings=Number(unique(/^pub let IR_MAX_STRINGS: i64 = ([0-9]+)\s*$/m,text,'ir_max_strings')[1]);
  const struct_body=unique(/^pub struct IrFunction\s*\{(.*?)^\}\s*$/ms,text,'irfunction_struct')[1];
  if(/^\s*pub instrs:\s*\[IrInstr;/m.test(struct_body))fail('inline_instrs_field_reintroduced');
  if(!/^\s*pub region:\s*IrInstrRegion,\s*$/m.test(struct_body))fail('irfunction_region_handle_missing');
  const receipts=[];
  for(const [rel,capname,fnname]of [['self-hosted/ir/dce.sio','DCE_MAX_INSTRS','dce_run_impl'],['self-hosted/ir/const_prop.sio','CP_MAX_INSTRS','cp_run_impl']]){
   const file=path.join(root,rel);if(!fs.existsSync(file))fail('guarded_source_missing_'+capname);
   const body=fs.readFileSync(file,'utf8');
   const match=body.match(new RegExp('^let '+capname+': i64 = ([0-9]+)\\s*$','m'));
   if(!match)fail(capname.toLowerCase()+'_declaration_missing');
   const pass_cap=Number(match[1]),fn=body.match(new RegExp('^fn '+fnname+'\\(.*?^\\}','ms'));
   if(!fn)fail(fnname+'_not_found');
   const guarded=new RegExp('instr_count\\s*>\\s*'+capname+'\\s*\\{').test(fn[0]);
   if(pass_cap<cap&&!guarded)fail(fnname+'_truncates_at_'+pass_cap+'_without_refusing_cap_'+cap);
   receipts.push(capname+'='+pass_cap+':'+(guarded?'refuses':'covers'));
  }
  unique(/pub functions:\s*\*mut IrFunctionTable,/,text,'irmodule_functions_handle');
  const chunk=unique(/^pub struct IrFunctionChunk\s*\{(.*?)^\}\s*$/ms,text,'irfunction_chunk')[1];
  const entries=Number(unique(/pub entries:\s*\[IrFunction;\s*([0-9]+)\]/,chunk,'function_chunk_entries')[1]);
  const table=unique(/^pub struct IrFunctionTable\s*\{(.*?)^\}\s*$/ms,text,'irfunction_table')[1];
  const chunks=Number(unique(/pub chunks:\s*\[i64;\s*([0-9]+)\]/,table,'function_table_chunks')[1]);
  if(BigInt(entries)*BigInt(chunks)!==BigInt(max_funcs))fail('function_pool_dimensions_ne_ir_max_funcs');
  for(const fnname of ['ir_fn_table_chunk_for','ir_fn_table_chunk_at']){
   const body=unique(new RegExp('^fn '+fnname+'\\(.*?^\\}','ms'),text,fnname)[0];
   if(!body.includes('gi < 0 || gi >= IR_MAX_FUNCS'))fail(fnname+'_index_guard_missing');
   if(!body.includes('ir_fn_table_latch(t, IR_FN_TABLE_ERR_INDEX, gi)'))fail(fnname+'_refusal_latch_missing');
   const divisor=Number(unique(/let chunk_no = gi \/ ([0-9]+)/,body,fnname+'_division')[1]);
   if(divisor!==entries)fail(fnname+'_division_ne_chunk_entries');
   if(fnname==='ir_fn_table_chunk_for'){
    const bound=Number(unique(/if chunk_no >= ([0-9]+) \{/,body,'function_table_chunk_bound')[1]);
    if(bound!==chunks)fail('function_table_chunk_bound_ne_table');
   }
  }
  const init=unique(/^fn ir_fn_chunk_alloc\(.*?^\}/ms,text,'function_chunk_allocator')[0];
  const init_count=Number(unique(/while i < ([0-9]+) \{/,init,'function_chunk_initializer')[1]);
  if(init_count!==entries)fail('function_chunk_initializer_ne_entries');
  const ser_path=path.join(root,'self-hosted/ir/serialize.sio');
  if(!fs.existsSync(ser_path))fail('serialize_source_missing');
  const ser=fs.readFileSync(ser_path,'utf8');
  if(/var functions:\s*\[IrFunction;/.test(ser))fail('serialize_inline_function_array_reintroduced');
  const deserialize=unique(/^pub fn deserialize_ir_module_ref\(.*?^\}/ms,ser,'deserialize_module_ref')[0];
  for(const required of ['var module = ir_empty_module()','while i < fn_count','ir_fn_pool_ensure(&module, i)','ir_fn_set(&! module, i, func_pair.0)']){
   if(!deserialize.includes(required))fail('serialize_function_pool_path_missing_'+required);
  }
  const ser_str=unique(/var string_table:\s*\[Name;\s*([0-9]+)\]/,ser,'serialize_string_table')[1];
  if(Number(ser_str)!==max_strings)fail('serialize_string_table_literal_'+ser_str+'_ne_ir_max_strings_'+max_strings);
  if(!/^pub var SOIR_DESER_REFUSAL_COUNT:\s*i64\s*=\s*0\s*$/m.test(ser))fail('soir_deser_refusal_count_missing');
  const count=ser.split('soir_note_deser_refusal()').length-1;
  if(count<3)fail('soir_note_deser_refusal_call_count_'+count);
  if(!ser.includes('fn_count < 0 || fn_count > IR_MAX_FUNCS'))fail('serialize_fn_count_not_checked_against_ir_max_funcs');
  console.log('IRFUNCTION_INSTR_CAPACITY_CHECK cap='+cap+' max_funcs='+max_funcs+' max_strings='+max_strings+' storage=arena_region inline_field=absent serialize_functions='+max_funcs+' serialize_strings='+max_strings+' soir_deser_refusal=detectable '+receipts.join(' ')+' coherent=pass');
 }else{
  const cap=Number(unique(/^pub let MIR_MAX_INSTRS: i64 = ([0-9]+)\s*$/m,text,'mir_max_instrs')[1]);
  const body=unique(/^pub struct MachineBlock\s*\{(.*?)^\}\s*$/ms,text,'machineblock_struct')[1];
  const field=Number(unique(/^\s*pub instrs:\s*\[MachineInstr;\s*([0-9]+)\],\s*$/m,body,'machineblock_instrs_field')[1]);
  const starts=[...text.matchAll(/^pub fn machine_block_new\(\) -> MachineBlock(?:\s+with[^{]+)?\s*\{/gm)];
  if(starts.length!==1)fail('machine_block_new_count_'+starts.length);
  const start=starts[0].index+starts[0][0].length,end=text.indexOf('\npub fn machine_function_new()',start);
  if(end<0)fail('machine_block_new_end_missing');
  const initializer=Number(unique(/^\s*instrs:\s*\[machine_empty_instr\(\);\s*([0-9]+)\],\s*$/m,text.slice(start,end),'machine_block_new_instrs_initializer')[1]);
  const sites=[...text.matchAll(/mir_instr_capacity/g)].length;
  if(sites<3)fail('fail_closed_sites_expected_ge_3_got_'+sites);
  if(cap!==expected)fail('mir_max_instrs_expected_'+expected+'_got_'+cap);
  if(field!==expected)fail('machineblock_field_expected_'+expected+'_got_'+field);
  if(initializer!==expected)fail('initializer_expected_'+expected+'_got_'+initializer);
  if(!(cap===field&&field===initializer))fail('divergent_cap_'+cap+'_field_'+field+'_initializer_'+initializer);
  console.log('MIR_INSTR_CAPACITY_CHECK cap='+cap+' field='+field+' initializer='+initializer+' fail_closed_sites='+sites+' coherent=pass');
 }
}catch(error){console.error(prefix+'_FAIL reason='+error.message);process.exitCode=1;}
