"""Board-local all-NPU functional benchmark. Parent controller owns both locks."""
from pathlib import Path
import argparse,json,os,time,threading,traceback,signal
from resident_worker import ResidentWorker
p=argparse.ArgumentParser();p.add_argument('--natural-max-new',type=int,default=96);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=Path(__file__).resolve().parent;a.output.mkdir(parents=True,exist_ok=False);f=(a.output/'events.jsonl').open('x',buffering=1);spec=json.loads((a.assets/'inputs.json').read_text());oracle=json.loads((a.assets/'reference.json').read_text());refs={x['id']:x for x in oracle['cases']};worker=None;stop=threading.Event();hot=threading.Event();samples=[];guard_reason=[];hz=os.sysconf('SC_CLK_TCK')
def emit(event,**x):
 line=json.dumps(dict(event=event,at=time.monotonic(),**x),ensure_ascii=False);f.write(line+'\n');print(line,flush=True)
def interrupted(signum,frame):
 raise RuntimeError('Controlled stop signal '+str(signum)+' guard='+str(guard_reason))
for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,interrupted)
def info(pid):
 p=Path('/proc')/str(pid);stat=(p/'stat').read_text().rsplit(')',1)[1].split();mem={}
 for line in (p/'smaps_rollup').read_text().splitlines():
  k,_,v=line.partition(':')
  if k in ['Rss','Pss','Swap']:mem[k+'_kB']=int(v.split()[0])
 return dict(pid=pid,cpu_seconds=(int(stat[11])+int(stat[12]))/hz,**mem)
def snap():
 temps={p.parent.name:int(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')};mem={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  k,_,v=line.partition(':')
  if k in ['MemAvailable','SwapFree','SwapTotal']:mem[k+'_kB']=int(v.split()[0])
 processes=[]
 childfile=Path('/proc')/str(os.getpid())/'task'/str(os.getpid())/'children'
 for pid in [os.getpid()]+([int(x) for x in childfile.read_text().split()] if childfile.exists() else []):
  try:processes.append(info(pid))
  except (FileNotFoundError,ProcessLookupError):pass
 freq={p.parent.name:p.read_text().strip() for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')}
 return dict(at=time.monotonic(),temperature=temps,memory=mem,processes=processes,system_cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9])),frequency=freq,loadavg=os.getloadavg())
def monitor():
 while not stop.wait(.25):
  try:
   x=snap();samples.append(x)
   reason=None
   if max(x['temperature'].values(),default=0)>=78:reason='temperature>=78C'
   if x['memory']['MemAvailable_kB']<256*1024:reason='MemAvailable<256MiB'
   if reason and not hot.is_set():
    hot.set();guard_reason.append(dict(reason=reason,at=x['at']));os.kill(os.getpid(),signal.SIGUSR1)
  except Exception as e:samples.append(dict(error=repr(e)))
thread=threading.Thread(target=monitor,daemon=True);thread.start()
def call(case,kind,reset=False,feedback=None,diagnostic=False,max_new=32):
 rows=[];start=time.monotonic()
 for e in worker.generate(case['tokens'],max_new,feedback=feedback,reset=reset,diagnostic=diagnostic,timeout=600):
  if hot.is_set():raise RuntimeError('78C continuous thermal guard')
  rows.append(e)
 done=rows[-1];expected=0 if reset else len(spec['prefix']);assert done['reused_tokens']==expected,(kind,done['reused_tokens'],expected)
 emit('request',kind=kind,case=case['id'],prompt=case['prompt'],prompt_tokens=len(case['tokens']),dynamic_tokens=case['dynamic_tokens'],prefix_limit=len(spec['prefix']),start=start,end=time.monotonic(),result=done,tokens=rows[:-1],prefill_backend='all16 NPU scalar',decode_backend='all16 NPU plus four full-vocabulary FP16 NPU head shards')
 return done
try:
 emit('scope',source=spec['source'],source_sha256=spec['source_sha256'],source_precision=spec['source_precision'],cache_policy='only frozen system prefix; full-question cache forbidden',cpu_ops=['embedding lookup','RoPE cos/sin construction; rotary multiplication is NPU','KV copies','dispatch','greedy selection'],npu_ops=['all16 layer RMSNorm,QKV,rotary,attention,MLP','final RMSNorm','all128256 vocabulary logits'],not_measured=['vision','speech concurrency','long-duration stability'])
 cmd=['env','OPI_PREFIX_LIMIT='+str(len(spec['prefix'])),'taskset','-c','6,7',str(r/'resident'),str(a.graphs),'16',str(spec['capacity']),'128256',str(a.assets/'embeddings.f32'),'0',str(spec['eos'][0]),str(a.assets/'rope_inv.f32')]
 dumps=a.output/'logits';dumps.mkdir();worker=ResidentWorker(cmd,str(r),str(a.output/'resident.log'),dump_dir=dumps,timeout=180);emit('loaded',ready=worker.ready,resources=snap())
 for case in spec['cases']:
  reference=refs[case['id']]['tokens'];n=min(8,len(reference));call(case,'fixed_feedback_numeric',True,reference[:n],True,n)
 for case in spec['cases']:call(case,'natural_no_cache',True,max_new=a.natural_max_new)
 prepare=list(worker.generate(spec['prefix'],1,reset=True,timeout=300))[-1];emit('system_prefix_preparation',result=prepare,excluded_from_following_ttft=True)
 for case in spec['cases']:call(case,'natural_system_cache',False,max_new=a.natural_max_new)
 for case in spec['cases']:
  fb=refs[case['id']]['tokens'];fb=(fb+[spec['eos'][0]]*32)[:32];call(case,'fixed32_execution',False,fb,False,32)
 closed=worker.close();worker=None;emit('closed',result=closed);assert closed['driver_bytes']==0;emit('complete')
except Exception as e:emit('error',error=repr(e),traceback=traceback.format_exc());raise
finally:
 for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,signal.SIG_IGN)
 if worker:worker.abort()
 profile=Path('/sys/kernel/debug/viplite/mem_profile')
 emit('cleanup',resident_stopped=True,guards=guard_reason,driver_memory_profile=profile.read_text() if profile.exists() else None)
 stop.set();thread.join(2);(a.output/'telemetry.json').write_text(json.dumps(dict(cpu_parser='after-final-rparen-v2',clock_ticks=hz,guards=guard_reason,samples=samples),indent=2));f.close()
