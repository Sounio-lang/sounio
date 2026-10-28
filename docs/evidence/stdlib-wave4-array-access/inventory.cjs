const fs=require('fs'),cp=require('child_process'),path=require('path'),crypto=require('crypto');
const root=process.argv[2],out=process.argv[3],before=process.argv[4];
function clean(s){let r=s.split(''),st='n';for(let i=0;i<s.length;i++){let c=s[i],n=s[i+1];if(st==='line'){if(c==='\n')st='n';else r[i]=' ';continue;}if(st==='block'){if(c!=='\n')r[i]=' ';if(c==='*'&&n==='/'){r[++i]=' ';st='n';}continue;}if(st==='str'||st==='char'){if(c!=='\n')r[i]=' ';if(c==='\\'){if(i+1<s.length)r[++i]=' ';continue;}if(c===(st==='str'?'"':"'"))st='n';continue;}if(c==='/'&&n==='/'){r[i]=r[++i]=' ';st='line';}else if(c==='/'&&n==='*'){r[i]=r[++i]=' ';st='block';}else if(c==='"'||c==="'"){r[i]=' ';st=c==='"'?'str':'char';}}return r.join('');}
function fns(s){let a=[],re=/^[ \t]*(?:pub[ \t]+)?fn[ \t]+(\w+)[ \t]*\(/gm,m;while(m=re.exec(s)){let b=-1,p=0,k=0;for(let z=m.index;z<s.length;z++){const c=s[z];if(c==='(')p++;else if(c===')')p--;else if(c==='[')k++;else if(c===']')k--;else if(c===';'&&p===0&&k===0)break;else if(c==='{'&&p===0&&k===0){b=z;break;}}if(b<0)continue;let d=0,e=-1;for(let i=b;i<s.length;i++){if(s[i]==='{')d++;else if(s[i]==='}'&&--d===0){e=i+1;break;}}if(e<0)continue;a.push({name:m[1],start:m.index,body:b,end:e});re.lastIndex=e;}return a;}
function arr(s,i){let d=0;for(let j=i;j<s.length;j++){if(s[j]==='[')d++;else if(s[j]===']'&&--d===0)return s.slice(i,j+1);}return null;}
const files=cp.execFileSync('git',['-C',root,'ls-files','--','stdlib/**/*.sio','tests/**/*.sio','self-hosted/**/*.sio'],{encoding:'utf8',maxBuffer:30e6}).trim().split('\n').filter(Boolean);
const rows=[],selected=new Set(),summary={files:files.length,raw_casts:0,linked_arrays:0,bss_arrays:0,uncertain:0,by_area:{}};
for(const file of files){
 let text=fs.readFileSync(path.join(root,file),'utf8');const override=path.join(before,file);if(fs.existsSync(override))text=fs.readFileSync(override,'utf8');
 const s=clean(text),fun=fns(s),decls=[];
 const re=/\b(?:var|let|const)\s+(\w+)\s*:\s*(&!?\s*)?\[/g;let m;
 while(m=re.exec(s)){let begin=s.indexOf('[',m.index),type=arr(s,begin);if(!type)continue;const owner=fun.find(f=>m.index>=f.body&&m.index<f.end);decls.push({name:m[1],kind:m[2]?'reference':'value',type,at:m.index,line:text.slice(0,m.index).split('\n').length,owner:owner?.name??'<global>',bss:!owner});}
 for(const f of fun){let h=s.slice(f.start,f.body),p=/\b(\w+)\s*:\s*(&!?\s*)?\[/g;while(m=p.exec(h)){let at=f.start+m.index,begin=s.indexOf('[',at),type=arr(s,begin);if(type)decls.push({name:m[1],kind:m[2]?'reference':'value',type,at,line:text.slice(0,at).split('\n').length,owner:f.name,bss:false,param:true});}}
 const casts=/\bas\s+\*(mut|const)\s+/g;
 while(m=casts.exec(s)){
  summary.raw_casts++;const area=file.split('/')[0];summary.by_area[area]=(summary.by_area[area]||0)+1;
  const left=s.slice(Math.max(0,m.index-200),m.index);let match=/(\w+)\s*$/.exec(left),explicitRef=false;
  const r=/(?:\(\s*)?(&!?)\s*(\w+)\s*\)?\s*$/.exec(left);if(r){match=[r[0],r[2]];explicitRef=true;}
  const owner=fun.find(f=>m.index>=f.body&&m.index<f.end),name=match?.[1],targetStart=casts.lastIndex,targetArray=s[targetStart]==='['?arr(s,targetStart):null;
  const local=decls.filter(d=>d.name===name&&d.at<m.index&&d.owner===(owner?.name??'<global>'));
  const global=decls.filter(d=>d.name===name&&d.bss);const choices=local.length?local:global;
  let linked=choices.length===1?choices[0]:null;
  if(match&&left.slice(0,left.length-match[0].length).trimEnd().endsWith('.'))linked=null;
  const record={file,line:text.slice(0,m.index).split('\n').length,owner:owner?.name??'<global>',expression:text.split('\n')[text.slice(0,m.index).split('\n').length-1].trim(),name:name??null,explicit_reference:explicitRef,target_mutable:m[1]==='mut',target_array:targetArray,target_prefix:s.slice(targetStart,targetStart+80).split('\n')[0].trim(),linked_declaration:linked};
  if(linked){summary.linked_arrays++;if(linked.bss)summary.bss_arrays++;selected.add(file);record.syntactic_class=(explicitRef||linked.kind==='reference')?(targetArray?'reference-array-pointer':'reference-element-or-named-pointer'):'value-to-raw';}
  else{summary.uncertain++;record.syntactic_class='unresolved-not-a-typed-claim';}
  rows.push(record);
 }
}
fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'IMPACT.json'),JSON.stringify({scope:'Syntactic declaration-linked inventory only; imports, aliases, fields, and shadowing require actual checker/BFS. Before library42 source is overlaid for the three files.',summary,records:rows},null,2)+'\n');fs.writeFileSync(path.join(out,'LINKED_FILES.tsv'),[...selected].sort().join('\n')+'\n');console.log(JSON.stringify(summary));
