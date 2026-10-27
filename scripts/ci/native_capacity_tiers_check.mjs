#!/usr/bin/env node
// Source metadata only. Native execution is a separate qualification.
import fs from 'node:fs';
import path from 'node:path';
const fail=r=>{throw Error(r);};
const flags=(r,extra='g')=>[...new Set((r.flags||'m')+extra)].join('');
const regex=r=>r instanceof RegExp?r:new RegExp(r,'m');
const all=(r,s)=>[...s.matchAll(new RegExp(regex(r).source,flags(regex(r))))];
function only(r,s,why){const m=all(r,s);if(m.length!==1)fail(why+'_count_'+m.length);return m[0];}
const test=(r,s)=>new RegExp(regex(r).source,regex(r).flags.replace('g','')).test(s);
const need=(r,s,why)=>{if(!test(r,s))fail(why);};
const accessor=(n,v)=>new RegExp('^'+v+'fn '+n+'\\(\\) -> i64 \\{ ([0-9]+) \\}$','m');
const body=(n,s,why,v='')=>only(new RegExp('^'+v+'fn '+n+'\\((?:.|\\n)*?^\\}$','m'),s,why)[0];
const list=s=>'['+[...s].sort((a,b)=>a-b).join(', ')+']';
function tier(n,m,a,w){const s=new Set([...m.slice(1),a[1]].map(Number));if(s.size!==1||![...s].every(x=>x===w))fail(n+'_expected_'+w+'_got_'+list(s));}
function arrays(r,s,count,why,acc,accWhy,w,name){
 const a=all(r,s);if(a.length!==count)fail(why+'_count_'+a.length);
 const values=new Set(a.flatMap(m=>m.slice(1)).map(Number));values.add(Number(only(accessor(acc,acc==='drv_label_capacity'?'':'pub '),s,accWhy)[1]));
 if(values.size!==1||![...values].every(x=>x===w))fail(name+'_expected_'+w+'_got_'+list(values));
}
try{
 const [root,...args]=process.argv.slice(2);
 if(args.length!==10||args.some(x=>!/^[0-9]+$/.test(x)||!Number.isSafeInteger(Number(x))))fail('invalid_arguments');
 const [ec,er,ee,el,base,labels,rodata,returns,strtab,drv]=args.map(Number);
 const files=['native/encode.sio','native/frame.sio','native/codegen_x86_linux.sio','native/codegen.sio','ir/ir.sio','native/elf.sio','native/lower_ir.sio','compiler/native_compile_driver.sio'];
 const [encode,frame,cgx,cg,ir,elf,lower,driver]=files.map(rel=>{const p=path.join(root,'self-hosted',rel);if(!fs.existsSync(p))fail('source_missing_self-hosted_'+rel.replaceAll('/','_'));return new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(fs.readFileSync(p));});
 tier('code_tier',only(/^pub var NC_BIG_CODE: \[i8; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,encode,'nc_big_code_decl'),only(accessor('nc_code_capacity_bytes','pub '),encode,'nc_code_capacity_accessor'),ec);
 arrays(/^pub var NC_FLAT_RELOC_(?:OFFSETS|KIND_CODES|IS_FUNCTIONS|TARGET_INDICES): \[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,frame,4,'nc_flat_reloc_decl','nc_flat_reloc_capacity','nc_flat_reloc_accessor',er,'reloc_tier');
 tier('elf_tier',only(/^var NC_BIG_ELF: \[i8; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,cgx,'nc_big_elf_decl'),only(accessor('nc_elf_capacity_bytes',''),cgx,'nc_elf_capacity_accessor'),ee);
 tier('legacy_elf_tier',only(/^var NATIVE_ELF_BUF: \[i8; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,cgx,'native_elf_buf_decl'),only(accessor('native_elf_buf_capacity_bytes',''),cgx,'native_elf_buf_accessor'),el);
 arrays(/^pub var NC_V2_LABEL_(?:OFFSETS|PATCH_OFFSETS|PATCH_IDS): \[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,frame,3,'nc_v2_label_decl','nc_v2_label_capacity','nc_v2_label_accessor',labels,'label_tier');
 const irMax=Number(only(/^pub let IR_MAX_INSTRS: i64 = ([0-9]+)\s*$/m,ir,'ir_max_instrs_decl')[1]);
 if(labels<irMax)fail('label_tier_'+labels+'_below_ir_max_instrs_'+irMax);
 if(labels>65536&&!frame.includes('NC_V2_LABEL_DIRTY_LEN'))fail('label_tier_'+labels+'_needs_dirty_cursor');
 const pairs=[['encode',encode,ec],['codegen_x86_linux',cgx,ec],['codegen',cg,ec],['frame',frame,er]];
 for(const [n,s,lit]of pairs)for(const line of s.split(/\r?\n/))if(test(new RegExp('[<>]=?\\s*'+lit+'\\b'),line)&&!line.includes('var '))fail('stray_literal_bound_'+n+'_'+lit);
 for(const [n,s]of pairs)for(const line of s.split(/\r?\n/)){if(/[<>]=?\s*2097152\b/.test(line))fail('stale_literal_bound_'+n+'_2097152');if(/[<>]=?\s*65536\b/.test(line)&&line.toLowerCase().includes('flat_reloc'))fail('stale_literal_bound_'+n+'_65536');}
 need(/pub reloc_overflow: bool,/,frame,'reloc_overflow_field_missing');
 need(/out\.reloc_overflow = false/,frame,'reloc_overflow_not_initialised');
 const reloc=body('nc_add_flat_reloc',cgx,'nc_add_flat_reloc');
 if(!reloc.includes('nc_flat_reloc_capacity()'))fail('nc_add_flat_reloc_not_using_accessor');
 need(/\}\s*else\s*\{(?:.|\n)*?\(\*nc\)\.reloc_overflow = true/,reloc,'nc_add_flat_reloc_missing_sentinel');
 only(/^\s*while i < \(\*nc\)\.flat_reloc_count && i < nc_flat_reloc_capacity\(\) \{$/m,cgx,'apply_relocations_bound');
 need(/if nc\.code_overflow \{(?:.|\n)*?return 19\s*\n\s*\}/,cgx,'rc19_code_overflow_check_missing');
 need(/if nc\.reloc_overflow \{(?:.|\n)*?return 20\s*\n\s*\}/,cgx,'rc20_reloc_overflow_check_missing');
 need(/pub label_overflow: bool,/,frame,'label_overflow_field_missing');
 need(/out\.label_overflow = false/,frame,'label_overflow_not_initialised');
 for(const n of ['nc_add_label_patch','nc_define_label']){const b=body(n,cgx,n);if(!b.includes('nc_v2_label_capacity()'))fail(n+'_not_using_accessor');need(/\}\s*else\s*\{(?:.|\n)*?nc_note_label_overflow\(nc\)/,b,n+'_missing_sentinel');}
 only(/^\s*while i < nc_v2_label_capacity\(\) \{$/m,cgx,'label_reset_full_bound');
 only(/^\s*if patch_lid >= 0 && patch_lid < nc_v2_label_capacity\(\) \{$/m,frame,'label_patch_lid_bounded');
 need(/if nc\.label_overflow \{(?:.|\n)*?return 22\s*\n\s*\}/,cgx,'rc22_label_overflow_check_missing');
 const put=only(/^fn nc_elf_put_u8\(val: i64\) with Mut, Panic \{(?:.|\n)*?^\}$/m,cgx,'nc_elf_put_u8')[0];
 if(!put.includes('native_elf_buf_capacity_bytes()'))fail('nc_elf_put_u8_unbounded');
 if(!put.includes('NATIVE_ELF_OVERFLOW = 1'))fail('nc_elf_put_u8_missing_sentinel');
 const legacy=body('compile_native_finalize_and_write_ref',cgx,'compile_native_finalize_and_write_ref');
 for(const [r,w]of [[/\(\*nc\)\.code_overflow/,'legacy_ignores_code_overflow'],[/\(\*nc\)\.reloc_overflow/,'legacy_ignores_reloc_overflow'],[/NATIVE_ELF_OVERFLOW != 0/,'legacy_ignores_elf_overflow'],[/return 19/,'legacy_missing_rc19'],[/return 20/,'legacy_missing_rc20'],[/return 21/,'legacy_missing_rc21'],[/\(\*nc\)\.label_overflow/,'legacy_ignores_label_overflow'],[/return 22/,'legacy_missing_rc22']])need(r,legacy,w);
 tier('rodata_tier',only(/^pub var NC_BIG_RODATA: \[i8; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,frame,'nc_big_rodata_decl'),only(accessor('nc_flat_rodata_capacity','pub '),frame,'nc_flat_rodata_capacity_accessor'),rodata);
 need(/pub rodata_overflow_bytes_attempted: i64,/,frame,'rodata_overflow_bytes_attempted_field_missing');
 need(/out\.rodata_overflow_bytes_attempted = 0/,frame,'rodata_overflow_bytes_attempted_not_initialised');
 need(/if nc\.rodata_overflow_bytes_attempted > nc_flat_rodata_capacity\(\) \{(?:.|\n)*?return 23\s*\n\s*\}/,cgx,'rc23_rodata_overflow_check_missing');
 const occurrences=cgx.split(/\r?\n/).map(l=>l.trim()).filter(l=>!l.startsWith('//')&&test(new RegExp('\\b'+base+'\\b'),l));
 if(occurrences.length!==6)fail('elf_base_addr_occurrence_count_'+occurrences.length);
 for(const l of occurrences){if(test(new RegExp('[<>]=?\\s*'+base+'\\b'),l))fail('elf_base_addr_used_as_bound');need(new RegExp('(?:,\\s*'+base+'\\)|,\\s*'+base+',|let base_addr: i64 = '+base+')'),l,'elf_base_addr_unexpected_form');}
 const irMax2=Number(only(/^pub let IR_MAX_INSTRS: i64 = ([0-9]+)\s*$/m,ir,'ir_max_instrs_decl_1646')[1]);
 const vregs=Number(only(/^pub let IR_MAX_VREGS: i64 = ([0-9]+)\s*$/m,ir,'ir_max_vregs_decl')[1]);
 tier('return_jump_tier',only(/^pub var NC_RETURN_JUMPS: \[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,frame,'nc_return_jumps_decl'),only(accessor('nc_return_jump_capacity','pub '),frame,'nc_return_jump_capacity_accessor'),returns);
 if(returns<irMax2)fail('return_jump_tier_'+returns+'_below_ir_max_instrs_'+irMax2);
 for(const [n,s]of [['frame',frame],['codegen_x86_linux',cgx],['codegen',cg]])if(/return_jump_count\s*<\s*16\b/.test(s)||/rj_i\s*<\s*16\b/.test(s))fail('stray_return_jump_literal_bound_'+n);
 const rec=body('record_return_jump_mut',frame,'record_return_jump_mut','pub ');
 if(!rec.includes('nc_return_jump_capacity()'))fail('record_return_jump_mut_not_using_accessor');
 need(/\}\s*else\s*\{\s*\(\*nc\)\.label_overflow = true/,rec,'record_return_jump_mut_missing_sentinel');
 const arm=/if c\.return_jump_count < nc_return_jump_capacity\(\) \{(?:.|\n)*?\}\s*else\s*\{(?:.|\n)*?c\.label_overflow = true/;
 for(const [n,s]of [['codegen_x86_linux',cgx],['codegen',cg]])if(all(arm,s).length!==1)fail('mir_ret_arm_missing_sentinel_'+n);
 const fl=only(/^pub var NC_IS_FLOAT_REG: \[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,frame,'nc_is_float_reg_decl');
 const limit=Number(only(/^pub let NC_VREG_LIMIT: i64 = ([0-9]+)$/m,frame,'nc_vreg_limit_decl')[1]);

 const fv=new Set([...fl.slice(1).map(Number),limit,vregs]);if(fv.size!==1)fail('float_typing_tier_not_ir_max_vregs_'+list(fv));
 const floatClear=body('nc_is_float_reg_clear',frame,'nc_is_float_reg_clear','pub ');
 need(/while i < NC_VREG_LIMIT \{\s*NC_IS_FLOAT_REG\[i as usize\] = 0/,floatClear,'float_typing_reset_not_full_capacity');
 const regClass=only(/^pub var NC_REG_CLASS: \[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,frame,'nc_reg_class_decl');
 if(regClass.slice(1).some(n=>Number(n)!==limit))fail('reg_class_tier_not_nc_vreg_limit');
 const classGet=body('nc_reg_class_get',frame,'nc_reg_class_get','pub ');
 if(!classGet.includes('if reg < 0 || reg >= NC_VREG_LIMIT { return 0 }'))fail('reg_class_get_missing_bound');
 const classSet=body('nc_reg_class_set',frame,'nc_reg_class_set','pub ');
 need(/if reg < 0 \|\| reg >= NC_VREG_LIMIT \{\s*NC_REG_CLASS_DROPPED = NC_REG_CLASS_DROPPED \+ 1\s*return\s*\}/,classSet,'reg_class_set_missing_detectable_bound');
 if(!classSet.includes('NC_REG_CLASS[reg as usize] = cls')||!classSet.includes('if reg + 1 > NC_REG_CLASS_HIGH_WATER')||!classSet.includes('NC_REG_CLASS_HIGH_WATER = reg + 1'))fail('reg_class_set_missing_high_water');
 const classReset=body('nc_reg_class_reset',frame,'nc_reg_class_reset','pub ');
 need(/while i < NC_REG_CLASS_HIGH_WATER \{\s*NC_REG_CLASS\[i as usize\] = 0/,classReset,'reg_class_reset_not_high_water');
 if(!classReset.includes('NC_REG_CLASS_HIGH_WATER = 0')||!classReset.includes('NC_REG_CLASS_DROPPED = 0'))fail('reg_class_reset_missing_counters');
 const highWords=Number((BigInt(vregs)+31n)/32n);
 const highSeen=only(/^pub var NC_HIGH_SEEN: \[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,frame,'nc_high_seen_decl');
 if(highSeen.slice(1).some(n=>Number(n)!==highWords))fail('high_seen_tier_not_ir_max_vregs');
 const resetWords=Number(only(/while i < ([0-9]+) \{\s*NC_HIGH_SEEN\[i as usize\] = 0/,classReset,'nc_high_seen_reset_bound')[1]);
 if(resetWords!==highWords)fail('high_seen_reset_not_full_capacity');
 const highNote=body('nc_high_float_note',frame,'nc_high_float_note','pub ');
 if(!highNote.includes('if reg < NC_MAX_VREGS || reg >= NC_VREG_LIMIT { return }'))fail('high_float_missing_reg_bound');
 if(!highNote.includes('let word = reg / 32')||!highNote.includes('let bit = reg % 32'))fail('high_float_bit_decomposition_mismatch');
 const highBound=Number(only(/if word < 0 \|\| word >= ([0-9]+) \{ return \}/,highNote,'nc_high_float_word_bound')[1]);
 if(highBound!==highWords)fail('high_float_bound_not_full_capacity');

 for(const [n,s]of [['codegen_x86_linux',cgx],['codegen',cg],['lower_ir',lower],['frame',frame]]){const ls=s.split(/\r?\n/);for(let i=0;i<ls.length;i++){const l=ls[i],next=ls[i+1]||'';if(l.includes('NC_MAX_VREGS')&&(l.includes('NC_IS_FLOAT_REG[')||(next.includes('NC_IS_FLOAT_REG[')&&!l.trim().startsWith('while'))))fail('float_typing_bounded_by_nc_max_vregs_'+n+'_line_'+(i+1));if(l.trim().startsWith('while')&&l.includes('NC_MAX_VREGS')&&next.includes('NC_IS_FLOAT_REG['))fail('float_typing_partial_reset_'+n+'_line_'+(i+1));}}
 tier('strtab_tier',only(/^pub struct StringTable \{\n    pub bytes: \[i8; ([0-9]+)\],$/m,elf,'string_table_bytes_decl'),only(accessor('elf_strtab_capacity_bytes','pub '),elf,'elf_strtab_capacity_accessor'),strtab);
 for(const n of ['string_table_add','data_section_align','data_section_add_f64']){const b=body(n,elf,n,'pub ');if(!b.includes('elf_strtab_capacity_bytes()'))fail(n+'_not_using_accessor');if(!b.includes('ELF_STRTAB_OVERFLOW = 1'))fail(n+'_missing_sentinel');if(/<\s*8192\b/.test(b))fail(n+'_stray_literal_bound');}
 const f64=body('data_section_add_f64',elf,'data_section_add_f64_body','pub '),guard=f64.indexOf('if t.len + 8 > elf_strtab_capacity_bytes()'),bump=f64.indexOf('t.len = t.len + 8');
 if(guard<0||bump<0||bump<guard)fail('data_section_add_f64_len_advanced_before_guard');
 const rc28=/if ELF_STRTAB_OVERFLOW != 0 \{(?:.|\n)*?return 28\s*\n\s*\}/,rc28count=all(rc28,cgx).length;if(rc28count!==2)fail('rc28_strtab_overflow_check_count_'+rc28count);
 const fin=body('finalize_compiled_elf',cgx,'finalize_compiled_elf','pub ');
 if(!fin.includes('ELF_STRTAB_OVERFLOW != 0')||!fin.includes('nc.label_overflow'))fail('finalize_compiled_elf_ignores_overflow');
 if(!body('compiler_new',frame,'compiler_new_1646','pub ').includes('ELF_STRTAB_OVERFLOW = 0'))fail('strtab_overflow_not_reset_per_compile');
 arrays(/^var DRV_LABEL_(?:OFFSETS|PATCH_OFFSETS|PATCH_IDS): \[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$/m,driver,3,'drv_label_decl','drv_label_capacity','drv_label_capacity_accessor',drv,'drv_label_tier');
 for(const n of ['drv_define_label','drv_add_label_patch','drv_patch_labels']){const b=body(n,driver,n);if(!b.includes('drv_label_capacity()'))fail(n+'_not_using_accessor');need(/\}\s*else\s*\{\s*DRV_LABEL_OVERFLOW = 1/,b,n+'_missing_sentinel');if(/<\s*1024\b/.test(b))fail(n+'_stray_literal_bound');}
 if(!body('drv_patch_labels',driver,'drv_patch_labels_target').includes('DRV_LABEL_OFFSETS[lid as usize] >= 0'))fail('drv_patch_labels_patches_undefined_target');
 need(/if DRV_LABEL_OVERFLOW != 0 \{(?:.|\n)*?return 6\s*\n\s*\}/,driver,'drv_rc6_label_overflow_check_missing');
 if(!body('drv_reset_codegen',driver,'drv_reset_codegen').includes('DRV_LABEL_OVERFLOW = 0'))fail('drv_label_overflow_not_reset');
 console.log('NATIVE_CAPACITY_TIERS_CHECK code='+ec+' reloc='+er+' elf='+ee+' legacy_elf='+el+' label='+labels+' rodata='+rodata+' base_addr_literals='+occurrences.length+' return_jumps='+returns+' strtab='+strtab+' drv_label='+drv+' fail_closed=rc19/rc20/rc21/rc22/rc23/rc28/drv_rc6 coherent=pass');
}catch(error){console.error('NATIVE_CAPACITY_TIERS_FAIL reason='+error.message);process.exitCode=1;}
