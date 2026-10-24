// Source-only transformation. No sampler, fit or numerical calculation runs here.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const args=process.argv.slice(2);
if(args.length!==2)throw Error('usage: node TRANSFORM.mjs ORIGINAL.sio NEW.sio');
const input=path.resolve(args[0]),output=path.resolve(args[1]);
if(input===output || fs.existsSync(output))throw Error('output must be new and distinct');
const raw=fs.readFileSync(input),original=raw.toString('utf8');
if(!Buffer.from(original,'utf8').equals(raw))throw Error('invalid UTF8');
for(const required of ['// Generated qualification driver: full profile-refit bootstrap, no assertions of calibrated validity.',' let p0:i64=1\n',' let p1:i64=2\n'])if(!original.includes(required))throw Error('unsupported original layout');
const plan=' assert(plan.status==0)\n';
const append='&!factor_scales1,&!permutation1)';
const close=' numeric_buffer_close(a0)\n';
for(const item of [plan,append,close])if(original.split(item).length!==2)throw Error('nonunique anchor '+item);
const setup=` var strict_embedding=numeric_buffer_new(p1*p0,p1*p0*8)
 var strict_warm=numeric_buffer_new(p1,p1*8)
 var strict_j:i64=0
 while strict_j<p1*p0 {numeric_buffer_set(&!strict_embedding,strict_j,1.0);strict_j=strict_j+1}
`;
const cleanup=' numeric_buffer_close(strict_embedding);numeric_buffer_close(strict_warm)\n';
let candidate=original.replace(/\bdynamic_count_profile_bootstrap_workspace\b/g,'dynamic_count_profile_bootstrap_nested_workspace').replace(/\bdynamic_count_profile_lrt_bootstrap\b/g,'dynamic_count_profile_lrt_bootstrap_nested');
candidate=candidate.replace(plan,plan+setup).replace(append,'&!factor_scales1,&!permutation1,&strict_embedding,&!strict_warm)').replace(close,cleanup+close);
// Invert every authorized edit: all data, options, seeds, selector and unrelated
// statements must be byte-identical. This refuses broader transformations.
const reverted=candidate.replace(/\bdynamic_count_profile_bootstrap_nested_workspace\b/g,'dynamic_count_profile_bootstrap_workspace').replace(/\bdynamic_count_profile_lrt_bootstrap_nested\b/g,'dynamic_count_profile_lrt_bootstrap').replace(plan+setup,plan).replace('&!factor_scales1,&!permutation1,&strict_embedding,&!strict_warm)',append).replace(cleanup+close,close);
if(reverted!==original)throw Error('inverse transform did not restore exact original');
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
fs.mkdirSync(path.dirname(output),{recursive:true});
fs.writeFileSync(output,candidate,{flag:'wx'});
const receipt={scope:'source-only exact inverse-verified transformation; no numerical execution',input,output,original_sha256:digest(raw),candidate_sha256:digest(candidate),transformer_sha256:digest(fs.readFileSync(fileURLToPath(import.meta.url))),changes:['two imported/called API names','caller embedding ones2 and warm2 for literal p0=1,p1=2 layout; runtime witness still checked','two appended call arguments','two buffer closes'],preserved:'all data/options/B/seeds/environment selector and other statements byte-identical under inverse transform'};
fs.writeFileSync(output+'.TRANSFORM.json',JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify(receipt));
