#!/usr/bin/env node
// Exit 122 means output limit exceeded; 124 means deadline exceeded.
// A bounded prefix is evidence of a FAILED phase, never a passing truncation.
import {spawn} from 'node:child_process';
import os from 'node:os';
const [seconds, bytes, separator, ...command] = process.argv.slice(2);
if (!/^[1-9][0-9]*$/.test(seconds||'') || !/^[1-9][0-9]*$/.test(bytes||'') || separator!=='--' || !command.length) throw Error('expected SECONDS MAX_BYTES -- COMMAND');
const duration=Number(seconds)*1000, limit=Number(bytes);
if(!Number.isSafeInteger(duration)||duration>2147483647||!Number.isSafeInteger(limit))throw Error('invalid bounds');
const child=spawn(command[0],command.slice(1),{detached:true,stdio:['ignore','pipe','pipe']});
let reason='',count=0,killTimer;
const kill=signal=>{try{process.kill(-child.pid,signal);}catch(e){if(e.code!=='ESRCH')throw e;}};
function stop(why) {
 if(reason)return;
 reason=why; kill('SIGTERM'); killTimer=setTimeout(()=>kill('SIGKILL'),1000);
}
const timer=setTimeout(()=>stop('deadline'),duration);
for(const stream of [child.stdout,child.stderr])stream.on('data',data=>{
 const available=Math.max(0,limit-count);count+=data.length;
 if(available)process.stdout.write(data.subarray(0,available));
 if(count>limit)stop('output-limit');
});
for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>stop(signal));
child.on('error',error=>{process.stderr.write(error.message+'\n');});
child.on('close',(code,signal)=>{
 clearTimeout(timer);clearTimeout(killTimer);if(reason)kill('SIGKILL');
 if(reason)process.stderr.write('\nSTDLIB_PHASE_FAILURE '+reason+' observed_bytes='+count+' limit_bytes='+limit+'\n');
 process.exitCode=reason==='output-limit'?122:reason==='deadline'?124:reason?143:(code!==null&&code>=0?code:signal?128+(os.constants.signals[signal]||0):127);
});
