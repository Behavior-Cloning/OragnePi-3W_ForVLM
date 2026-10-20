import json,time,os,threading,traceback,psutil,concurrent.futures,urllib.request,sys
from pathlib import Path
from board_adapters import Adapter
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r));from resident_worker import ResidentWorker
run=time.strftime('%Y%m%dT%H%M%S');f=(r/'results'/('co-resident-'+run+'.jsonl')).open('x',buffering=1)
def emit(**x):f.write(json.dumps(x,ensure_ascii=False)+'\n');print(x.get('event'),x.get('llm'),x.get('jev'),x.get('mode'),flush=True)
def snap():
 procs=[psutil.Process(),*psutil.Process().children(recursive=True)];procs += [q for q in psutil.process_iter(['name','cmdline']) if 'llama-server' in ' '.join(q.info.get('cmdline') or []) and q.pid not in [z.pid for z in procs]];v=psutil.virtual_memory();swap=psutil.swap_memory();proc=[]
 for p in procs:
  try:
   m=p.memory_full_info();proc.append(dict(pid=p.pid,rss=m.rss,pss=getattr(m,'pss',None),cpu_seconds=sum(p.cpu_times()[:2])))
  except psutil.Error:pass
 return dict(at=time.monotonic(),available=v.available,system_used=v.used,swap_used=swap.used,swap_in=swap.sin,swap_out=swap.sout,processes=proc,temperature={p.parent.name:float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')},frequency_khz={p.parents[1].name:int(p.read_text()) for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq')},loadavg=os.getloadavg())
def post(path,data):
 req=urllib.request.Request('http://127.0.0.1:8081'+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 return urllib.request.urlopen(req,timeout=180)
def cpu_request(tokens,n=32,cache=True):
 t=time.perf_counter();first=None;last=None
 with post('/completion',dict(prompt=tokens,n_predict=n,temperature=0,seed=42,cache_prompt=cache,stream=True)) as response:
  for line in response:
   if not line.startswith(b'data: '):continue
   data=line[6:].strip()
   if data==b'[DONE]':continue
   x=json.loads(data)
   if x.get('content') and first is None:first=time.perf_counter()-t
   if x.get('stop'):last=x
 if last is None or first is None:raise RuntimeError('CPU stream missing first content or final timings')
 return dict(ttft_seconds=first,wall_seconds=time.perf_counter()-t,final=last)
inputs=json.loads((r/'resident-inputs.json').read_text());cases=json.loads((r/'cases.json').read_text());acts=json.loads((r/'actions.json').read_text());opts={a['id']:a['description'] for a in acts[:10]}
for llm in ['smol500','intern1b','qwen15-cpu']:
 w=None;m=None;stop=threading.Event();mon=[]
 try:
  if llm=='qwen15-cpu':
   health=json.load(urllib.request.urlopen('http://127.0.0.1:8081/v1/models',timeout=10));emit(event='cpu_identity',llm=llm,models=health)
   system=inputs['intern1b']['system'];prefix=json.load(post('/apply-template',{'messages':[{'role':'system','content':system}],'add_generation_prompt':False}))['prompt'];prefix=json.load(post('/tokenize',{'content':prefix,'add_special':True,'parse_special':True}))['tokens']
   prompts=[]
   for q in inputs['intern1b']['queries']:
    t=json.load(post('/apply-template',{'messages':[{'role':'system','content':system},{'role':'user','content':'记忆：用户喜欢清淡饮食，明天需要上班。此轮对话：'+q}],'add_generation_prompt':True}))['prompt'];prompts.append(json.load(post('/tokenize',{'content':t,'add_special':True,'parse_special':True}))['tokens'])
   def prepare():cpu_request(prefix,1,False)
   def generate(i):return cpu_request(prompts[i%len(prompts)])
  else:
   cfg=inputs[llm];g='/opt/opi-vlm/npu-kv-probe/'+('full-smol500-c1280-dfpgateup' if llm=='smol500' else 'wsl-fresh-peak155-20261018')
   if not Path(g).exists():raise FileNotFoundError(g)
   binary=str(r/'bin/resident-smol') if llm=='smol500' else '/opt/opi-vlm/resident-agent-20261019/resident'
   cmd=['taskset','-c','4,5',binary,g,'32' if llm=='smol500' else '24','1280','49280' if llm=='smol500' else '151674',g+'/embeddings.f32','32' if llm=='smol500' else '16','49279' if llm=='smol500' else '151645']
   w=ResidentWorker(cmd,str(r),str(r/'results'/f'{run}-{llm}-worker.log'));emit(event='npu_ready',llm=llm,ready=w.ready)
   def prepare():list(w.generate(cfg['prefix'],1))
   def generate(i):
    z=list(w.generate(cfg['prompts'][i%len(cfg['prompts'])],32))[-1];z['expected_system_prefix_tokens']=len(cfg['prefix'])
    expected_reuse=len(cfg['prefix'])//(32 if llm=='smol500' else 16)*(32 if llm=='smol500' else 16);z['expected_aligned_reused_tokens']=expected_reuse
    if z.get('reused_tokens')!=expected_reuse:raise ValueError('Unexpected KV prefix reuse: '+str(z))
    return z
  emit(event='loaded',llm=llm,resource=snap())
  for i in range(3):
   prepare();emit(event='llm_baseline',llm=llm,i=i,result=generate(i),resource=snap())
  for jev in ['nano33','laya','jev08']:
   try:
    t=time.perf_counter();m=Adapter(jev,r,2);emit(event='pair_loaded',llm=llm,jev=jev,load_seconds=time.perf_counter()-t,resource=snap())
    state,n=m.state(cases[0],64);m.decide(state,opts)
    def decide(i):
     state,n=m.state(cases[i%10],64);t=time.perf_counter();z=m.decide(state,opts);return dict(seconds=time.perf_counter()-t,state_tokens=n,result=z)
    for mode in ['jev_only','llm_only','concurrent']:
     for i in range(5):
      if mode!='jev_only':prepare()
      before=snap();t=time.perf_counter();times={};measure_samples=[];sample_stop=threading.Event()
      def monitor():
       while not sample_stop.wait(.5):
        try:measure_samples.append(snap())
        except Exception as e:measure_samples.append({'error':repr(e)})
      mt=threading.Thread(target=monitor,daemon=True);mt.start()
      barrier=threading.Barrier(2)
      def timed(name,fn):
       barrier.wait();start=time.monotonic();v=fn(i);times[name]={'start':start,'end':time.monotonic()};return v
      with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
       if mode=='concurrent':
        a=pool.submit(timed,'llm',generate);b=pool.submit(timed,'jev',decide);result=dict(llm=a.result(),jev=b.result())
       elif mode=='llm_only':result=dict(llm=generate(i))
       else:result=dict(jev=decide(i))
      sample_stop.set();mt.join(timeout=3)
      emit(event='pair_measure',llm=llm,jev=jev,mode=mode,i=i,wall_seconds=time.perf_counter()-t,result=result,before=before,after=snap(),samples=measure_samples,intervals=times,overlap_seconds=max(0,min(times['llm']['end'],times['jev']['end'])-max(times['llm']['start'],times['jev']['start'])) if len(times)==2 else None)
   except Exception as e:emit(event='pair_error',llm=llm,jev=jev,error=repr(e),trace=traceback.format_exc())
   finally:
    if 'sample_stop' in locals():sample_stop.set();mt.join(timeout=3)
    if m:m.close();del m;m=None
    import gc;gc.collect()
 except Exception as e:emit(event='llm_error',llm=llm,error=repr(e),trace=traceback.format_exc())
 finally:
  if w:
   try:emit(event='npu_closed',llm=llm,result=w.close())
   except Exception as e:emit(event='close_error',llm=llm,error=repr(e))
f.close()
