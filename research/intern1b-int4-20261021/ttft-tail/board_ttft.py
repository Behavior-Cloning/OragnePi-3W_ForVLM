from pathlib import Path
import json,time,traceback,hashlib,psutil,threading
from resident_worker import ResidentWorker
r=Path(__file__).resolve().parent;root=r.parent
run=r/'results'/time.strftime('%Y%m%dT%H%M%S');run.mkdir(parents=True,exist_ok=False)
f=(run/'events.jsonl').open('x',buffering=1);inputs=json.loads((root/'inputs.json').read_text());feedback=json.loads((r/'feedback.json').read_text());prefix=inputs['prefix'];w=None
stop=threading.Event();samples=[]
def emit(**x):f.write(json.dumps(x,ensure_ascii=False)+'\n');print(x.get('event'),x.get('variant'),x.get('mode'),x.get('case'),flush=True)
def snap():
 sw=psutil.swap_memory()
 return {'time':time.monotonic(),'temperature':max(float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')),'available':psutil.virtual_memory().available,'swap_in':sw.sin,'swap_out':sw.sout,'cpu':psutil.cpu_percent()}
def monitor():
 while not stop.wait(.5):
  try:samples.append(snap())
  except Exception as e:samples.append({'error':repr(e)})
mt=threading.Thread(target=monitor,daemon=True);mt.start()
try:
 graphs={'w8':'/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018','w4':str(root/'w4')}
 order=[('w8',0,3),('w8',8,3),('w4',0,3),('w4',8,3),('w4',8,2),('w4',0,2),('w8',8,2),('w8',0,2)]
 emit(event='protocol',order=order,scope='engine TTFT, one generated token; same resident weights, same system KV32; diagnostic16 output tokens outside timings',binary_sha=hashlib.sha256((r/'resident').read_bytes()).hexdigest())
 for session,(variant,mode,repeats) in enumerate(order):
  if snap()['temperature']>=65:raise RuntimeError('65C start limit')
  g=graphs[variant];dump=run/(str(session)+'-'+variant+'-'+str(mode));dump.mkdir()
  w=ResidentWorker(['env','OPI_TTFT_TAIL_MIN='+str(mode),'taskset','-c','6,7',str(r/'resident'),g,'24','1280','151674',g+'/embeddings.f32','16','151645'],str(r),str(run/(str(session)+'-worker.log')),dump)
  emit(event='loaded',session=session,variant=variant,mode=mode,ready=w.ready)
  def call(case,count=1,diagnostic=False,fb=None):
   if snap()['temperature']>=75:raise RuntimeError('75C thermal guard')
   prep=list(w.generate(prefix,1,reset=True))[-1]
   a=time.monotonic();out=list(w.generate(case['tokens'],count,feedback=fb,diagnostic=diagnostic))[-1];b=time.monotonic()
   assert out['reused_tokens']==32 and out['generated_tokens']==count if fb else out['reused_tokens']==32
   return {'result':out,'start':a,'end':b,'dump':str(dump/(str(out['id'])+'-logits.f32')) if diagnostic else None}
  call(inputs['performance'][0]);emit(event='warmed',session=session,variant=variant,mode=mode)
  if session<4:
   for case in inputs['performance']:
    z=call(case,16,True,feedback[case['id']][:16]);emit(event='diagnostic',session=session,variant=variant,mode=mode,case=case['id'],**z)
   if variant=='w8':
    for case in inputs['quality']:
     z=call(case,64);emit(event='quality',session=session,variant=variant,mode=mode,case=case['id'],**z)
  for repeat in range(repeats):
   cases=inputs['performance'] if repeat%2==0 else list(reversed(inputs['performance']))
   for case in cases:
    z=call(case);emit(event='measure',session=session,variant=variant,mode=mode,case=case['id'],repeat=repeat,**z)
  emit(event='closed',session=session,variant=variant,mode=mode,result=w.close());w=None
 emit(event='complete')
except BaseException as e:emit(event='error',error=repr(e),trace=traceback.format_exc());raise
finally:
 if w:w.abort()
 stop.set();mt.join(timeout=3);emit(event='telemetry',samples=samples);f.close()
