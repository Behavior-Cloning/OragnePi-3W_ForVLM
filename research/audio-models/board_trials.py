"""One externally locked coordinator: intentionally parallel CPU audio and NPU LLM."""
import argparse,hashlib,json,os,platform,shlex,shutil,statistics,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KINDS=['vad','denoise','asr','speaker']
NPU=Path('/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018')
PREFIX='board-fresh-peak155-h1024-n32-20261018'
ENV=os.environ.copy();ENV.update(PYTHONPATH=str(ROOT/'deps'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def monitor(done,pids,rows):
 while not done.is_set():
  row={'monotonic':time.monotonic(),'temperature_mC':int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()),'processes':{}}
  mem=dict((l.split(':')[0],int(l.split()[1])) for l in Path('/proc/meminfo').read_text().splitlines())
  row.update(mem_available_kib=mem['MemAvailable'],swap_used_kib=mem['SwapTotal']-mem['SwapFree'])
  row['cpu_ticks']=Path('/proc/stat').read_text().splitlines()[0]
  row['frequencies_khz']={str(p.parent.name):p.read_text().strip() for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')}
  row['devfreq_hz']={str(p.parent.name):p.read_text().strip() for p in Path('/sys/class/devfreq').glob('*/cur_freq')}
  for pid in list(pids):
   try:
    status={}
    for l in Path(f'/proc/{pid}/status').read_text().splitlines():
     if l.startswith(('VmRSS:','VmHWM:','Threads:')):status[l.split(':')[0]]=int(l.split()[1])
    for l in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
     if l.startswith('Pss:'):status['Pss_kib']=int(l.split()[1])
    row['processes'][str(pid)]=status
   except (OSError,ValueError):pass
  rows.append(row);done.wait(.2)
def worker(kind,out,extra=[],cpus='0,1'):
 return ['taskset','-c',cpus,sys.executable,str(ROOT/'audio_worker.py'),'--kind',kind,'--out',str(out),'--threads','1',*extra]
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['standalone','mixed'],required=True);p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
 out=ROOT/a.out;out.mkdir(exist_ok=False);summary={'status':'running','mode':a.mode,'runs':[],'monitor_interval_seconds':.2}
 summary['environment']={'uname':platform.uname()._asdict(),'disk_usage_bytes':shutil.disk_usage(ROOT)._asdict(),'cpu_online':Path('/sys/devices/system/cpu/online').read_text().strip(),'runner_sha256':hashlib.sha256((NPU/(PREFIX+'-full_model')).read_bytes()).hexdigest()}
 try:
  if a.mode=='standalone':
   for kind in KINDS:
    folder=out/kind;folder.mkdir();print('START',kind,flush=True);resources=[];done=threading.Event()
    with (folder/'run.log').open('w') as log:
     child=subprocess.Popen(worker(kind,folder/'result.json',['--repeats','5']),env=ENV,stdout=log,stderr=subprocess.STDOUT);pids=[child.pid]
     thread=threading.Thread(target=monitor,args=(done,pids,resources));thread.start()
     try:code=child.wait(timeout=600)
     finally:
      if child.poll() is None:child.terminate();child.wait(timeout=10)
      done.set();thread.join();save(folder/'system-resources.json',resources)
    if code:raise RuntimeError(f'{kind} failed {code}; see {folder}')
    d=json.loads((folder/'result.json').read_text());summary['runs'].append({'kind':kind,'status':'complete','p50_seconds':d['p50_seconds'],'rtf_p50':d['rtf_p50'],'load_seconds':d['load_seconds'],'resources':d['final_resources']});save(out/'summary.json',summary)
    print('DONE',kind,summary['runs'][-1],flush=True)
  else:
   for repeat in range(a.repeats):
    conditions=['baseline',*KINDS,'all','asr-shared']
    if repeat%2:conditions=conditions[::-1]
    for dynamic in [64,256]:
     for condition in conditions:
      name=f'd{dynamic}-{condition}-{repeat}';folder=out/name;folder.mkdir();print('START',name,flush=True)
      kinds=KINDS if condition=='all' else [] if condition=='baseline' else ['asr'] if condition=='asr-shared' else [condition]
      children=[];logs=[];resources=[];done=threading.Event();pids=[]
      monitor_thread=threading.Thread(target=monitor,args=(done,pids,resources));monitor_thread.start()
      try:
       for kind in kinds:
        log=(folder/(kind+'.log')).open('w');logs.append(log)
        ready=folder/(kind+'.ready');go=folder/'go';stop=folder/'stop'
        c=subprocess.Popen(worker(kind,folder/(kind+'.json'),['--seconds','600','--ready',str(ready),'--go',str(go),'--stop',str(stop)],cpus='6,7' if condition=='asr-shared' else '0,1'),env=ENV,stdout=log,stderr=subprocess.STDOUT);children.append(c);pids.append(c.pid)
       deadline=time.monotonic()+120
       while not all((folder/(k+'.ready')).exists() for k in kinds):
        assert all(c.poll() is None for c in children),'audio exited before ready'
        if time.monotonic()>deadline:raise TimeoutError('audio model loading')
        time.sleep(.05)
       # Precondition while audio models are loaded but idle. No heater runs
       # during either the CPU audio stress or measured NPU response.
       with (folder/'thermal.log').open('w') as log:subprocess.run([sys.executable,'thermal_condition.py','--min','55000','--max','64999'],cwd=NPU,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=330)
       env={k:v for k,v in ENV.items() if not k.startswith('OPI_')}
       env.update(OPI_PERFORMANCE='1',OPI_DECODER_GROUP='12',OPI_FAST_GREEDY='1',OPI_GROUP_HEAD='1',OPI_PREFILL_TOKENS='16',OPI_CACHED_PREFIX='64',OPI_MAX_START_TEMP_MC='70000',OPI_FEEDBACK_FILE=str(NPU/(PREFIX+'-feedback.i32')))
       prompt=Path('/opt/opi-vlm/short-prefix-d64-20261018/p64-d64.i32') if dynamic==64 else Path('/opt/opi-vlm/short-prefix-20261018/p64-d256.i32')
       command=['taskset','-c','6,7','./'+PREFIX+'-full_model','.','24','1280','151674','embeddings.f32',str(prompt),'128',str(folder/'llm.json'),str(folder/'unused.f32'),'151645']
       (folder/'go').write_text(str(time.monotonic()))
       with (folder/'llm.log').open('w') as log:
        llm=subprocess.Popen(command,cwd=NPU,env=env,stdout=log,stderr=subprocess.STDOUT);pids.append(llm.pid)
        try:code=llm.wait(timeout=120)
        finally:
         if llm.poll() is None:llm.terminate();llm.wait(timeout=10)
       (folder/'stop').write_text(str(time.monotonic()))
       for c in children:
        c.wait(timeout=90);assert c.returncode==0,'audio inference failed'
       assert code==0,'NPU failed'
       d=json.loads((folder/'llm.json').read_text());assert d['driver_after_release_bytes']==0 and d['generated_tokens']==128
       assert d['head_on_npu'] and d['decoder_layers']==24 and d['cached_prefix_tokens']==64 and d['new_prompt_tokens']==dynamic
       d.update(condition=condition,dynamic=dynamic,repeat=repeat,case=name)
       start=d['response_start_monotonic_seconds'];end=start+d['total_seconds'];d['audio_overlap']={}
       for kind in kinds:
        audio=json.loads((folder/(kind+'.json')).read_text())
        rs=[r for r in audio['runs'] if r['end_monotonic']>start and r['start_monotonic']<end]
        covered=sum(max(0,min(r['end_monotonic'],end)-max(r['start_monotonic'],start)) for r in rs)
        assert covered>0
        d['audio_overlap'][kind]={'inference_overlap_seconds':covered,'response_seconds':end-start,'calls':len(rs),'samples':[r['sample'] for r in rs],'rtf_p50':statistics.median(r['rtf'] for r in rs),'cpu_seconds':sum(r['cpu_seconds'] for r in rs)}
       prior=[r for r in summary['runs'] if r['dynamic']==dynamic]
       assert not prior or d['tokens']==prior[0]['tokens'],'predictions changed under contention'
       summary['runs'].append(d);save(out/'summary.json',summary);print('DONE',name,'ttft',d['ttft_seconds'],'tps',d['decode_tokens_per_second'],flush=True)
      finally:
       for c in children:
        if c.poll() is None:c.terminate()
       for c in children:
        try:c.wait(timeout=10)
        except subprocess.TimeoutExpired:c.kill();c.wait()
       done.set();monitor_thread.join();save(folder/'system-resources.json',resources)
       for log in logs:log.close()
  summary['status']='complete'
 except BaseException as e:summary.update(status='failed',error=repr(e));raise
 finally:save(out/'summary.json',summary)
if __name__=='__main__':main()
