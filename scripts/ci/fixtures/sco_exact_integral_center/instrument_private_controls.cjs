const fs=require('fs');const root=process.argv[2],helper=process.argv[3];if(!root||!helper)throw Error('usage source-copy helper');
let p=root+'/self-hosted/ir/opt_cleanup.sio',s=fs.readFileSync(p,'utf8');if(s.includes('pub fn ocp_private_real_owned_controls'))throw Error('already instrumented');fs.appendFileSync(p,'\n'+fs.readFileSync(helper,'utf8'));
p=root+'/self-hosted/compiler/module_frontend.sio';s=fs.readFileSync(p,'utf8');const importAnchor='OCP_RESEARCH_EXACT_CENTER_LAST_FUNCTION}';if(s.split(importAnchor).length!==2)throw Error('import anchor');s=s.replace(importAnchor,'OCP_RESEARCH_EXACT_CENTER_LAST_FUNCTION, ocp_private_real_owned_controls}');const callAnchor='            let cleanup_receipt = opt_cleanup_module_inplace(&! (*module_box))';const callAnchor2='        let cleanup_receipt = opt_cleanup_module_inplace(&! (*module_box))';if(s.split(callAnchor).length!==2)throw Error('single path');s=s.replace(callAnchor,'            module_frontend_private_real_controls(&! (*module_box))\n'+callAnchor);if(s.split(callAnchor2).length!==3)throw Error('merged path including single indentation');const last=s.lastIndexOf(callAnchor2);s=s.slice(0,last)+'        module_frontend_private_real_controls(&! (*module_box))\n'+s.slice(last);
s+=`
// PRIVATE test-copy hook only. Does not load or grant research permission.
fn module_frontend_private_real_controls(module: &! IrModule) with IO, Mut, Div, Panic {
    if !str_eq(read_env("SOUNIO_PRIVATE_MFI_GUARD_CONTROLS"), "1") { return }
    var fi: i64 = 0
    var found = false
    while fi < (*module).fn_count {
        if ir_name_eq(ir_function_name_at(&(*module), fi), make_name("exact_integral_center")) {
            found = true
            if ocp_private_real_owned_controls(module, fi) {
                print("PRIVATE_REAL_OWNED_CONTROLS baseline=1 refusals=6 restored=1 PASS\\n")
            } else {
                print("PRIVATE_REAL_OWNED_CONTROLS FAIL\\n")
            }
        }
        fi = fi + 1
    }
    if !found { print("PRIVATE_REAL_OWNED_CONTROLS MISSING_TARGET\\n") }
}
`;
fs.writeFileSync(p,s);
