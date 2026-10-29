// Filesystem identities only; no numerical computation.
import fs from 'node:fs';
import path from 'node:path';
import {digest,treeManifest,verifyTree} from '../wave4_variance_shadow/admission.mjs';
const[action,root,file]=process.argv.slice(2);
if(action==='seal')fs.writeFileSync(file,JSON.stringify(treeManifest(root),null,2)+'\n',{flag:'wx'});
else if(action==='verify')verifyTree(root,{file:path.resolve(file),sha256:digest(file)});
else throw Error('guard action');
