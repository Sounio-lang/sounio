import fs from 'node:fs';
import crypto from 'node:crypto';
const root=process.argv[2];
function declaration(source,name){
    const re=new RegExp('^(?:pub )?(?:fn|struct) '+name+'[ (\\{]','m');const match=re.exec(source);
    if(!match){throw Error('missing declaration '+name);}
    const end=source.indexOf('\n}\n',match.index);if(end<0){throw Error('missing declaration end '+name);}
    return source.slice(match.index,end+3);
}
const manifest=[];
for(const phase of ['before','after']){
    const dir=root+'/source/'+phase;fs.mkdirSync(dir,{recursive:true});
    const dep=fs.readFileSync(root+'/input/dependent.'+phase+'.sio','utf8');
    const ep=fs.readFileSync(root+'/input/epistemic.'+phase+'.sio','utf8');
    const constants=ep.match(/^const (?:PROVENANCE|VALIDITY)_KIND_[^\n]+/gm);
    if(!constants||constants.length!==12){throw Error('unexpected constant census');}
    const names=['Validity','provenance_kind_name','provenance_kind_needs_verification','validity_is_expired','validity_unsatisfiable','validity_subsumes'];
    const pieces=names.map(name=>{const text=declaration(ep,name);manifest.push(crypto.createHash('sha256').update(text).digest('hex')+'  '+phase+'/'+name);return text;});
    fs.writeFileSync(dir+'/epistemic.sio',constants.join('\n')+'\n'+pieces.join('\n')+'\n'+fs.readFileSync(root+'/input/epistemic_probe.sio.in','utf8').replace('@AFTER@',phase==='after'?'1':'0'));
    fs.writeFileSync(dir+'/dependent.sio',dep+'\n'+fs.readFileSync(root+'/input/dependent_probe.sio.in','utf8'));
    manifest.push(crypto.createHash('sha256').update(declaration(dep,'dep_check_single_constraint')).digest('hex')+'  '+phase+'/dep_check_single_constraint');
}
fs.writeFileSync(root+'/EXTRACTED_DECLARATIONS.sha256',manifest.join('\n')+'\n');
