from pathlib import Path
import os,sys,json,time,queue,threading,traceback,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
R=Path(__file__).resolve().parent;base=Path('/opt/opi-vlm/resident-agent-20261019')
sys.path[:0]=[str(R),str(base),str(base/'deps'),'/opt/opi-vlm/audio-lab-20261018/deps']
from agent_replay import Frontend,wav,snapshot,np
from resident_worker_v1 import ResidentWorker
from context_rag import Retriever
from tokenizers import Tokenizer
from fixture_validation import validate_fixture
cfg=json.loads((R/'config.json').read_text());doc=json.loads((R/'inputs.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(R/'inputs.json')==cfg['inputs_sha256']
assert sha(cfg['binary'])==cfg['binary_sha256']
tokpath=base/'tokenizer.json';assert sha(tokpath)==doc['tokenizer_sha256']
tok=Tokenizer.from_file(str(tokpath));validate_fixture(doc,lambda s:tok.encode(s,add_special_tokens=False).ids)
for p,h in cfg['graphs_sha256'].items():assert sha(p)==h,p
os.sched_setaffinity(0,{7})
f=(R/'events.jsonl').open('x',buffering=1);lock=threading.Lock();stop=threading.Event();errors=[]
state={'status':'starting','phase':'startup','worker':None};front=None;threads=[]
def emit(event,**kw):
 with lock:f.write(json.dumps({'event':event,'at':time.monotonic(),**kw},ensure_ascii=False)+'\n')
def ast():return {'frames':front.frames,'overruns':front.overruns,'max_lag_s':front.max_lag,'error':front.error,'queue':front.jobs.qsize()}
def feed():
 try:
  i=0
  while not stop.is_set():
   if front.error:raise RuntimeError(front.error)
   if front.jobs.empty():
    i+=1;x=wav(base/'speech'/('%02d.wav'%((i-1)%12+1)));front.jobs.put({'id':i,'kind':'command','mode':'stream','audio':np.concatenate([x,np.zeros(32000,np.float32)])})
   try:
    while True:
     e=front.events.get_nowait()
     if e.get('event')=='endpoint':emit('audio_endpoint',text=e.get('text'))
   except queue.Empty:pass
   stop.wait(.1)
 except BaseException:errors.append(traceback.format_exc());stop.set()
def monitor():
 try:
  with (R/'resources.jsonl').open('x',buffering=1) as log:
   while not stop.wait(.5):
    w=state['worker'];row=snapshot([os.getpid()]+([w.process.pid] if w else []));row.update(phase=state['phase'],variant=state.get('variant'),frontend=ast());log.write(json.dumps(row)+'\n')
    if row['temp_mC']>=78000:raise RuntimeError('78C guard')
 except BaseException:errors.append(traceback.format_exc());stop.set()
def req(w,tokens,n,**kw):
 assert not errors,errors
 ev=list(w.generate(tokens,max_new=n,**kw));return ev[-1],ev[:-1]
try:
 front=Frontend('stream',.5);front.start();assert front.ready.wait(120) and not front.error
 rag=Retriever(catalog_path=Path('/opt/opi-vlm/async-agent-20261025/atoms.json'))
 emit('resident_models',rag=rag.metadata,scope='synthetic streamed audio + resident RAG; text Intern NPU only; no motors/images')
 threads=[threading.Thread(target=fn,daemon=True) for fn in [feed,monitor]]
 for t in threads:t.start()
 for name,tail in [('original',True),('outlier8',True),('original-repeat',True)]:
  state.update(phase='load',variant=name)
  graph=cfg['outlier_graph'] if name=='outlier8' else cfg['original_graph'];env=['OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1','OPI_TTFT_TAIL_MIN=8','OPI_GREEDY_I16_SINGLE_PASS=1']
  if tail:env.append('OPI_PREFILL8_DIR='+(cfg['outlier_tail'] if name=='outlier8' else cfg['original_tail']))
  cmd=['env',*env,'taskset','-c','4,5',cfg['binary'],graph,'24','448','151674',graph+'/embeddings.f32','32','151645']
  dump=R/(name+'-dumps');dump.mkdir()
  w=ResidentWorker(cmd,str(base),str(R/(name+'.log')),dump_dir=str(dump),timeout=180);state['worker']=w
  emit('ready',variant=name,metrics=w.ready);assert w.ready['true_tail8_enabled']==tail
  try:
   warm=doc['performance'][0];req(w,warm['prefix_tokens'],1,reset=True);req(w,warm['tokens'],32,feedback=[100662]*32)
   state['phase']='performance'
   for c in doc['performance']:
    prep,_=req(w,c['prefix_tokens'],1,reset=True)
    m,ev=req(w,c['tokens'],32,feedback=[100662]*32)
    assert not m['diagnostic'] and m['fixed_feedback'] and m['new_tokens']==c['body_tokens']+8 and m['reused_tokens']==c['static_tokens']
    assert m['true_tail8_calls']==int(tail) and m['scalar_prompt_tokens']==0
    emit('measure',variant=name,case=c['id'],system_body=c['system_body_tokens'],system_template=5,system_total=c['static_tokens'],dynamic_body=c['body_tokens'],dynamic_template=8,metrics=m,token_events=ev,prefix=prep)
   state['phase']='quality'
   for c in doc['quality']:
    req(w,c['prefix_tokens'],1,reset=True);m,ev=req(w,c['tokens'],32)
    assert m['reused_tokens']==c['static_tokens'] and m['new_tokens']==c['body_tokens']+8
    emit('quality',variant=name,case=c['id'],system_total=c['static_tokens'],dynamic_body=c['body_tokens'],dynamic_template=8,metrics=m,text=tok.decode(m['tokens'],skip_special_tokens=True))
   state['phase']='diagnostic'
   for c in [x for x in doc['performance'] if x['id'] in ['s64-d64-0','s128-d256-0']]:
    req(w,c['prefix_tokens'],1,reset=True);m,ev=req(w,c['tokens'],4,feedback=[100662]*4,diagnostic=True)
    p=dump/(str(m['id'])+'-logits.f32');assert p.stat().st_size==2426784
    emit('diagnostic',variant=name,case=c['id'],metrics=m,path=str(p.relative_to(R)),sha256=sha(p))
  finally:
   emit('closed',variant=name,metrics=w.close());state['worker']=None
 state['status']='complete'
except BaseException:
 state['status']='failed';errors.append(traceback.format_exc());emit('error',error=errors[-1]);raise
finally:
 stop.set()
 if front:front.stop.set();front.join(15)
 for t in threads:t.join(5)
 meta={'status':state['status'] if not errors else 'failed','errors':errors,'audio':ast() if front else None,'ended':time.time()}
 (R/'status.json').write_text(json.dumps(meta,indent=2));f.close()
