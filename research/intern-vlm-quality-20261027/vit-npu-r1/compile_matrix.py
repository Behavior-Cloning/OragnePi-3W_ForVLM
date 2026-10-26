from pathlib import Path
import subprocess,fcntl,json,time,os
P=Path(__file__).resolve().parent;R=P.parents[2];state=P/'compile-matrix-status.json';assert not state.exists()
s={'status':'starting','pid':os.getpid(),'cases':[]}
def save():state.write_text(json.dumps(s,indent=2)+'\n')
save()
with open('/tmp/opi-prefill-compile.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for side,dtype in [(112,'fp16'),(112,'int8'),(224,'fp16'),(224,'int16'),(224,'int8')]:
  out=P/('s%d-%s'%(side,dtype));assert not out.exists()
  cmd=['/opt/opi-lab/export-venv/bin/python',str(R/'npu/kv-cache/compile_probe.py'),'--source',str(P/('s'+str(side))),'--results-root',str(P),'--output',str(out),'--dtype',dtype,'--timeout','180']
  s.update(status='running',side=side,dtype=dtype);save()
  with (P/('compile-%d-%s.log'%(side,dtype))).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=800)
  s['cases'].append({'side':side,'dtype':dtype,'returncode':z.returncode});save();print(s['cases'][-1],flush=True)
 s['status']='complete';save()
