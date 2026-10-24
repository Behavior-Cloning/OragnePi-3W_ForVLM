import os,sys,time,json,threading,traceback,hashlib,queue,ast
from pathlib import Path
ROOT=Path('/opt/opi-vlm/async-agent-20261025');BASE=Path('/opt/opi-vlm/resident-agent-20261019');OPT=Path('/opt/opi-vlm/intern1b-operator-opt-20261023')
sys.path[:0]=[str(BASE),str(BASE/'deps'),'/opt/opi-vlm/audio-lab-20261018/deps']
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from agent_replay import Frontend,wav,snapshot,np
from resident_worker import ResidentWorker
from tokenizers import Tokenizer
sys.path.insert(0,str(ROOT));from context_rag import Retriever
out=Path(sys.argv[1]);out.mkdir(exist_ok=False,parents=True);os.sched_setaffinity(0,{7});log=(out/'events.jsonl').open('w',buffering=1);lock=threading.Lock();stop=threading.Event();w=None

def emit(kind,**kw):
 with lock:log.write(json.dumps({'event':kind,'at':time.monotonic(),**kw},ensure_ascii=False)+'\n')
tok=Tokenizer.from_file(str(BASE/'tokenizer.json'));enc=lambda s:tok.encode(s,add_special_tokens=False).ids
# Extract exact v2 prefix literal, without executing the harness.
tree=ast.parse((ROOT/'harness.py').read_text());prefix=None
for n in ast.walk(tree):
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prefix' for t in n.targets) and isinstance(n.value,ast.Constant):prefix=n.value.value
assert prefix;prefix_ids=enc(prefix)
front=Frontend('stream',.5);front.start();assert front.ready.wait(120) and not front.error
rag=Retriever(catalog_path=ROOT/'atoms.json');emit('resident_models',rag=rag.metadata,system_tokens=len(prefix_ids))
def feed():
 i=0
 while not stop.is_set():
  if front.error:emit('audio_error',error=front.error);stop.set();return
  if front.jobs.empty():
   i+=1;x=wav(BASE/'speech'/('%02d.wav'%((i-1)%12+1)));front.jobs.put({'id':i,'kind':'command','mode':'stream','audio':np.concatenate([x,np.zeros(32000,np.float32)])})
  try:
   while True:
    e=front.events.get_nowait();e.pop('audio',None)
    if e.get('event')=='endpoint':emit('audio_endpoint',text=e.get('text'))
  except queue.Empty:pass
  stop.wait(.1)
def monitor():
 with (out/'resources.jsonl').open('w') as f:
  while not stop.wait(.5):
   ids=[os.getpid()]+([w.process.pid] if w else [])
   row=snapshot(ids);row['audio_frames']=front.frames;f.write(json.dumps(row)+'\n');f.flush()
threads=[threading.Thread(target=fn,daemon=True) for fn in [feed,monitor]]
for th in threads:th.start()
seedtexts=['用户刚回家有些疲惫，希望机器人点头回应，不要移动底盘。','用户正在阅读，希望保持安静，取消刚才向前移动的要求。','用户想打个招呼，希望摆臂回应。桌边空间有限，底盘保持静止。']
fixtures=[]
for budget in [32,48,64]:
 for i,base in enumerate(seedtexts):
  text=base
  assert len(enc(text))<=budget
  while len(enc(text))<budget:text+='。'
  assert len(enc(text))==budget
  ids=enc(prefix+'<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n')
  fixtures.append({'id':str(budget)+'-'+str(i),'body':text,'body_tokens':budget,'tokens':ids,'template_overhead':len(ids)-len(prefix_ids)-budget,'padding':'punctuation padding for exact-size performance only'})
quality=seedtexts+['请点头回应。','请弯腰，不要抬头。','取消刚才移动，保持静止。']
(out/'fixtures.json').write_text(json.dumps({'prefix':prefix,'prefix_tokens':prefix_ids,'performance':fixtures,'quality':quality},ensure_ascii=False,indent=2))
g=Path('/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018');configs=[('old',BASE/'resident',[]),('tail8',OPT/'resident',['OPI_TTFT_TAIL_MIN=8']),('group',OPT/'resident',['OPI_TTFT_TAIL_MIN=8','OPI_GROUP_PREFILL=1']),('group-exact',OPT/'resident',['OPI_TTFT_TAIL_MIN=8','OPI_GROUP_PREFILL=1','OPI_EXACT_PREFIX=1']),('old-repeat',BASE/'resident',[])]
try:
 for name,binary,env in configs:
  command=['env',*env,'taskset','-c','4,5',str(binary),str(g),'24','1280','151674',str(g/'embeddings.f32'),'16','151645']
  w=ResidentWorker(command,BASE,out/(name+'.log'));emit('loaded',variant=name,ready=w.ready,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
  list(w.generate(prefix_ids,1,reset=True));list(w.generate(fixtures[0]['tokens'],1))
  for repeat in range(3):
   for case in (fixtures if repeat%2==0 else list(reversed(fixtures))):
    if int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())>=78000:raise RuntimeError('78C cutoff')
    prep=list(w.generate(prefix_ids,1,reset=True))[-1];t=time.monotonic();z=list(w.generate(case['tokens'],1,timeout=60))[-1]
    emit('measure',variant=name,repeat=repeat,case=case['id'],body_tokens=case['body_tokens'],template_overhead=case['template_overhead'],wall_s=time.monotonic()-t,metrics=z,prefix_prepare_metrics=prep)
  for i,text in enumerate(quality):
   list(w.generate(prefix_ids,1,reset=True));ids=enc(prefix+'<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n');z=list(w.generate(ids,32,timeout=60))[-1];reply=tok.decode(z['tokens']);actions=rag.query(reply,k=3)
   emit('quality',variant=name,case=i,input=text,reply=reply,metrics=z,top3=actions)
  emit('closed',variant=name,metrics=w.close());w=None
 emit('complete',audio_frames=front.frames,max_audio_lag_s=front.max_lag)
except BaseException:emit('error',traceback=traceback.format_exc());raise
finally:
 if w:w.abort()
 stop.set();front.stop.set();front.join(10)
 for th in threads:th.join(5)
 log.close()
