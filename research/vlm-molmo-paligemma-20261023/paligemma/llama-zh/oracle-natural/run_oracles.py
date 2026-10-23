from pathlib import Path
import os,sys,subprocess,time,json,signal,shutil,hashlib
D=Path(__file__).parent;P=D.parent;PY='/opt/opi-lab/export-venv/bin/python'
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
base=Path('/opt/opi-lab/llama32-full-npu-20261023')
def available():
 return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
runs=[]
for label,model,build in [('q8',P/'q8-source/FuseChat-Llama-3.2-1B-Instruct-Q8_0.gguf',base/'q8-source'),('q4',P/'FuseChat-Llama-3.2-1B-Instruct-Q4_K_M.gguf',base/'oracle-natural/q4-source')]:
 if available()<7*1024**3:raise RuntimeError('Insufficient initial available RAM for oracle')
 env.update(LLAMA_GGUF_MODEL=str(model),LLAMA_BUILD_ROOT=str(build))
 if label=='q8':subprocess.run([PY,str(D/'verify_tokenizer.py')],env=env,check=True,timeout=90)
 out=D/label;out.mkdir(exist_ok=True);tele=[];reason=None;t0=time.monotonic()
 with (out/'run.log').open('w') as log:
  p=subprocess.Popen([PY,'-u',str(D/'source-snapshot/generate_reference.py'),'--execute','--max-new','96'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while p.poll() is None:
    status={}
    try:
     for line in Path(f'/proc/{p.pid}/status').read_text().splitlines():
      if line.startswith(('VmRSS:','VmHWM:','VmSwap:')):status[line.split(':')[0]]=int(line.split()[1])*1024
    except FileNotFoundError:pass
    av=available();tele.append(dict(monotonic=time.monotonic(),available_bytes=av,**status))
    if av<768*1024**2 or status.get('VmRSS',0)>9*1024**3 or time.monotonic()-t0>1200:
     reason='memory-or-time-guard';os.killpg(p.pid,signal.SIGTERM);break
    time.sleep(1)
   try:rc=p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait(timeout=10)
  finally:
   if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
 (out/'telemetry.json').write_text(json.dumps(tele,indent=2))
 record=dict(label=label,pid=p.pid,returncode=rc,guard=reason,seconds=time.monotonic()-t0,model=str(model),build=str(build),max_new=96,threads=2)
 (out/'run-status.json').write_text(json.dumps(record,indent=2));runs.append(record);print(json.dumps(record),flush=True)
 if rc!=0:raise RuntimeError('Reference failed; see isolated log')
 src=build/'oracle-natural';dest=out/'results';dest.mkdir(exist_ok=True)
 for f in src.iterdir():
  if f.is_file():shutil.copy2(f,dest/f.name)
 print((dest/'reference.json').read_text(),flush=True)
(D/'runs.json').write_text(json.dumps(runs,indent=2));print('NATURAL_ORACLES_COMPLETE',flush=True)
