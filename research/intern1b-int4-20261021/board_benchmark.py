"""Paired NPU W8/W4 benchmark; caller owns both experiment locks."""
from pathlib import Path
import json,time,threading,psutil,os,hashlib,traceback,statistics
from resident_worker import ResidentWorker
r=Path(__file__).resolve().parent;run=r/'results'/time.strftime('%Y%m%dT%H%M%S');run.mkdir(parents=True,exist_ok=False)
f=(run/'events.jsonl').open('x',buffering=1);inputs=json.loads((r/'inputs.json').read_text());feedback={};prefix=inputs['prefix'];stop=threading.Event();samples=[]
def emit(**x):f.write(json.dumps(x,ensure_ascii=False)+'\n');print(x.get('event'),x.get('variant'),x.get('case'),flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def snap():
 procs=[psutil.Process(),*psutil.Process().children(recursive=True)];rows=[]
 for p in procs:
  try:m=p.memory_full_info();rows.append({'pid':p.pid,'rss':m.rss,'pss':m.pss,'cpu_seconds':sum(p.cpu_times()[:2])})
  except psutil.Error:pass
 sw=psutil.swap_memory()
 return {'at':time.monotonic(),'processes':rows,'available':psutil.virtual_memory().available,'swap_used':sw.used,'swap_in':sw.sin,'swap_out':sw.sout,'temperature':{x.parent.name:float(x.read_text())/1000 for x in Path('/sys/class/thermal').glob('thermal_zone*/temp')},'cpu_percent':psutil.cpu_percent(),'loadavg':os.getloadavg(),'cpu_freq':{x.parent.parent.name:int(x.read_text()) for x in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq')},'npu_frequency':Path('/sys/class/devfreq/3600000.npu/cur_freq').read_text().strip()}
def monitor():
 while not stop.wait(.5):
  try:samples.append(snap())
  except Exception as e:samples.append({'error':repr(e)})
mt=threading.Thread(target=monitor,daemon=True);mt.start();worker=None
try:
 # Hash once before measurements, not during model requests.
 graphs={'w8':'/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018','w4':str(r/'w4')};binary=Path('/opt/opi-vlm/resident-agent-20261019/resident')
 manifest={v:{p.name:sha(p) for p in sorted(Path(d).glob('*.nb'))} for v,d in graphs.items()}
 emit(event='provenance',graphs=graphs,graph_sha256=manifest,binary_sha256=sha(binary),source_sha256={n:sha(r/n) for n in ['board_benchmark.py','resident_worker.py','inputs.json']},scope='NPU text decoder only; W4 values exported via INT8 constants; no image encoder; existing CPU Qwen service left resident')
 for session,(variant,repeats) in enumerate([('w8',3),('w4',3),('w4',2),('w8',2)]):
  for wait in range(60):
   if max(snap()['temperature'].values())<60:break
   time.sleep(5)
  else:raise RuntimeError('cooldown below60 timed out')
  directory=Path(graphs[variant]);dumps=run/(str(session)+'-'+variant+'-diagnostic');dumps.mkdir()
  worker=ResidentWorker(['taskset','-c','6,7',str(binary),str(directory),'24','1280','151674',str(directory/'embeddings.f32'),'16','151645'],str(r),str(run/(str(session)+'-'+variant+'-worker.log')),dump_dir=dumps)
  emit(event='loaded',session=session,variant=variant,ready=worker.ready,resources=snap())
  def prepare():return list(worker.generate(prefix,1,reset=True))[-1]
  def call(case,max_new=64,fb=None,diagnostic=False):
   if max(snap()['temperature'].values())>=78:raise RuntimeError('78C thermal gate before request')
   prepare();started=time.monotonic();t=time.perf_counter();rows=list(worker.generate(case['tokens'],max_new,feedback=fb,diagnostic=diagnostic));result=rows[-1]
   if result['reused_tokens']!=len(prefix)//16*16:raise RuntimeError('unexpected system-prefix reuse')
   return dict(result=result,events=rows[:-1],wall_seconds=time.perf_counter()-t,start=started,end=time.monotonic())
  warm=call(inputs['quality'][0],16);emit(event='warmup',session=session,variant=variant,seconds=warm['wall_seconds'])
  if session in [0,1]:
   for case in inputs['quality']:
    z=call(case);emit(event='quality',session=session,variant=variant,case=case['id'],**z)
   if variant=='w8':
    for case in inputs['performance']:
     z=call(case);tokens=z['result']['tokens'];feedback[case['id']]=tokens+[tokens[-1]]*(64-len(tokens));emit(event='reference_feedback',variant=variant,case=case['id'],natural_tokens=len(tokens),feedback=feedback[case['id']],**z)
    (run/'feedback.json').write_text(json.dumps(feedback,indent=2))
   diag=call(inputs['performance'][0],64,feedback[inputs['performance'][0]['id']],True);emit(event='diagnostic',session=session,variant=variant,case=inputs['performance'][0]['id'],dump=str(dumps/(str(diag['result']['id'])+'-logits.f32')),**diag)
  for repeat in range(repeats):
   cases=inputs['performance'] if repeat%2==0 else list(reversed(inputs['performance']))
   for case in cases:
    before=snap();z=call(case,64,feedback[case['id']]);emit(event='measure',session=session,variant=variant,repeat=repeat,case=case['id'],dynamic_tokens=case['dynamic_actual'],before=before,after=snap(),**z)
  emit(event='closed',session=session,variant=variant,result=worker.close());worker=None
 emit(event='complete',feedback=feedback)
except Exception as e:emit(event='error',error=repr(e),trace=traceback.format_exc());raise
finally:
 if worker:worker.abort()
 stop.set();mt.join(timeout=3);emit(event='telemetry',samples=samples);f.close()
