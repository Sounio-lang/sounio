// Byte-preserving extraction of the actual native report/HTML/SVG streams.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const base=process.argv[2];if(!base)throw Error('artifact directory required');
const names=['viz_nb2_deviance','viz_nb_dispersion_diagnostic','viz_logarithmic_primitives','viz_genomics_entropy','viz_sampled_uncertainty'];
const identities=[];
for(const name of names){const dir=path.join(base,name),buf=fs.readFileSync(path.join(dir,'native.stdout'));const text=buf.toString('latin1');const begin=text.search(/<!doctype\s+html/i),end=text.toLowerCase().lastIndexOf('</html>');const svg=text.search(/<svg\b/),svgEnd=text.toLowerCase().indexOf('</svg>',svg);if(begin<0||end<begin||svg<begin||svgEnd<svg)throw Error('incomplete output: '+name);const parts={'report.txt':buf.subarray(0,begin),'figure.html':buf.subarray(begin,end+7),'figure.svg':buf.subarray(svg,svgEnd+6)};for(const [file,data] of Object.entries(parts)){fs.writeFileSync(path.join(dir,file),data);identities.push(crypto.createHash('sha256').update(data).digest('hex')+'  '+name+'/'+file);}}
fs.writeFileSync(path.join(base,'EXTRACTED_IDENTITIES.sha256'),identities.join('\n')+'\n');
