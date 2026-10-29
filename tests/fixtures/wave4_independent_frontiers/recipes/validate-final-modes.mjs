// Final custody supplement: all paths/modes/link targets; regular contents use the actual overlaid SOURCE.sha256.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const [inventory,root,treeFile]=process.argv.slice(2);
if(!inventory||!root||!treeFile){throw Error('inventory, root, tree file required');}
const expectedTree=fs.readFileSync(treeFile,'utf8').trim();
const raw=fs.readFileSync(inventory);
if(raw.length===0||raw.at(-1)!==0){throw Error('incomplete NUL-terminated Git census');}
const expected=new Map();const directories=new Set(['']);
const tree={children:new Map()};
function hashObject(type,data){return crypto.createHash('sha1').update(type+' '+data.length+'\0').update(data).digest('hex');}
for(const entry of raw.subarray(0,-1).toString('utf8').split('\0')){
    const m=/^(100644|100755|120000) blob ([0-9a-f]{40})\t(.+)$/.exec(entry);
    if(!m){throw Error('unsupported or malformed tree entry: '+entry);}
    const [,mode,oid,name]=m;
    if(name.startsWith('/')||name.split('/').some(x=>x===''||x==='.'||x==='..')||expected.has(name)){throw Error('unsafe or duplicate path '+name);}
    expected.set(name,{mode,oid});
    const parts=name.split('/');let cursor=tree;let prefix='';
    for(const part of parts.slice(0,-1)){
        prefix=prefix?prefix+'/'+part:part;directories.add(prefix);
        let next=cursor.children.get(part);if(!next){next={children:new Map()};cursor.children.set(part,next);}
        if(!next.children){throw Error('file/directory conflict '+name);}cursor=next;
    }
    const leaf=parts.at(-1);if(cursor.children.has(leaf)){throw Error('duplicate tree leaf '+name);}
    cursor.children.set(leaf,{mode,oid});
}
if(!Buffer.from(raw.subarray(0,-1).toString('utf8'),'utf8').equals(raw.subarray(0,-1))){throw Error('non UTF-8 census refused');}
function treeHash(node){
    const entries=[...node.children.entries()];
    entries.sort(([an,a],[bn,b])=>Buffer.compare(Buffer.from(an+(a.children?'/':'\0')),Buffer.from(bn+(b.children?'/':'\0'))));
    const chunks=[];
    for(const [name,value] of entries){
        const mode=value.children?'40000':value.mode;
        const oid=value.children?treeHash(value):value.oid;
        chunks.push(Buffer.from(mode+' '+name+'\0'),Buffer.from(oid,'hex'));
    }
    return hashObject('tree',Buffer.concat(chunks));
}
const reconstructed=treeHash(tree);
if(reconstructed!==expectedTree){throw Error('flat inventory does not bind expected Git tree: '+reconstructed);}
const observed=new Set();const observedDirs=new Set(['']);const lines=[];
function walk(relative){
    for(const name of fs.readdirSync(path.join(root,relative)).sort()){
        const rel=relative?relative+'/'+name:name;const absolute=path.join(root,rel);const st=fs.lstatSync(absolute);
        if(st.isDirectory()){if(!directories.has(rel)){throw Error('unexpected directory '+rel);}observedDirs.add(rel);walk(rel);continue;}
        const want=expected.get(rel);if(!want){throw Error('unexpected file '+rel);}
        let actualMode;let data;
        if(st.isSymbolicLink()){actualMode='120000';data=fs.readlinkSync(absolute,{encoding:'buffer'});}
        else if(st.isFile()){
            const permissions=st.mode&0o7777;if(permissions!==0o644&&permissions!==0o755){throw Error('noncanonical permissions '+rel+' '+permissions.toString(8));}
            actualMode=permissions===0o755?'100755':'100644';data=null;
        }else{throw Error('unsupported filesystem object '+rel);}
        if(actualMode!==want.mode){throw Error('mode mismatch '+rel);}
        if(data!==null&&hashObject('blob',data)!==want.oid){throw Error('blob or symlink-target mismatch '+rel);}
        observed.add(rel);lines.push(actualMode+'\t'+want.oid+'\t'+rel);
    }
}
walk('');
for(const name of expected.keys()){if(!observed.has(name)){throw Error('missing file '+name);}}
for(const name of directories){if(!observedDirs.has(name)){throw Error('missing directory '+name);}}
console.log('GIT_FINAL_PATH_MODE_LINK_CENSUS_OK regular_contents_bound_separately=true tree='+reconstructed+' files='+observed.size+' directories='+observedDirs.size);
console.log(lines.join('\n'));
