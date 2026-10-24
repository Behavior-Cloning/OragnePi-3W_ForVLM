import os,sys,time,json,threading,traceback,hashlib,queue,ast,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/opt/opi-vlm/resident-agent-20261019');OPT=Path('/opt/opi-vlm/intern1b-operator-opt-20261023')
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
def token_hash(ids):return hashlib.sha256(struct.pack('<'+str(len(ids))+'I',*ids)).hexdigest()
fixture_document=json.loads((ROOT/'aligned-inputs.json').read_text())
assert hashlib.sha256((BASE/'tokenizer.json').read_bytes()).hexdigest()==fixture_document['tokenizer_sha256']
for system in fixture_document['systems']:
 assert enc(system['prefix'])==system['prefix_tokens'] and len(enc(system['system_body']))==system['system_body_tokens']
 assert len(system['prefix_tokens'])==system['static_tokens']==system['system_body_tokens']+system['system_template_tokens']
 assert token_hash(system['prefix_tokens'])==system['prefix_sha256']
for case in fixture_document['performance']+fixture_document['quality']:
 assert enc(case['prompt_text'])==case['tokens']
 assert len(case['prefix_tokens'])==case['static_tokens']
 assert token_hash(case['tokens'])==case['tokens_sha256'] and token_hash(case['prefix_tokens'])==case['prefix_sha256']
 assert len(case['tokens'])==case['static_tokens']+case['body_tokens']+case['template_overhead']
 assert case['template_overhead']==8 and case['static_tokens'] in [64,128]
front=Frontend('stream',.5);front.start();assert front.ready.wait(120) and not front.error
rag=Retriever(catalog_path=ROOT/'atoms.json');emit('resident_models',rag=rag.metadata,static_tokens=[64,128])
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
fixtures=fixture_document['performance'];quality=fixture_document['quality']
fixture_document['board_source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['ttft_aligned_matrix.py','context_rag.py','atoms.json','resident-mask','aligned-inputs.json']}
(out/'fixtures.json').write_text(json.dumps(fixture_document,ensure_ascii=False,indent=2))
g=Path('/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018');candidate=ROOT/'resident-mask'
configs=[('plain',candidate,[]),('tail8',candidate,['OPI_TTFT_TAIL_MIN=8']),('group',candidate,['OPI_TTFT_TAIL_MIN=8','OPI_GROUP_PREFILL=1']),('mask',candidate,['OPI_TTFT_TAIL_MIN=8','OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1']),('group-repeat',candidate,['OPI_TTFT_TAIL_MIN=8','OPI_GROUP_PREFILL=1']),('mask-validate',candidate,['OPI_TTFT_TAIL_MIN=8','OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1','OPI_MASK_TEMPLATE_VALIDATE=1'])]

try:
 for name,binary,env in configs:
  command=['env',*env,'taskset','-c','4,5',str(binary),str(g),'24','1280','151674',str(g/'embeddings.f32'),'16','151645']
  dump_dir=out/(name+'-diagnostic');dump_dir.mkdir();w=ResidentWorker(command,BASE,out/(name+'.log'),dump_dir=dump_dir);emit('loaded',variant=name,ready=w.ready,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
  list(w.generate(fixtures[0]['prefix_tokens'],1,reset=True));list(w.generate(fixtures[0]['tokens'],1))
  for repeat in range(0 if name=='mask-validate' else 3):
   for case in (fixtures if repeat%2==0 else list(reversed(fixtures))):
    if int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())>=78000:raise RuntimeError('78C cutoff')
    prep=list(w.generate(case['prefix_tokens'],1,reset=True))[-1];t=time.monotonic();z=list(w.generate(case['tokens'],1,timeout=60))[-1]
    assert z['reused_tokens']==case['expected_reused_tokens'] and z['new_tokens']==case['expected_new_tokens']
    emit('measure',variant=name,repeat=repeat,case=case['id'],static_tokens=case['static_tokens'],system_body_tokens=case['system_body_tokens'],body_tokens=case['body_tokens'],template_overhead=case['template_overhead'],wall_s=time.monotonic()-t,metrics=z,prefix_prepare_metrics=prep)
  for case in ([] if name=='mask-validate' else quality):
   list(w.generate(case['prefix_tokens'],1,reset=True));z=list(w.generate(case['tokens'],32,timeout=60))[-1];reply=tok.decode(z['tokens']);actions=rag.query(reply,k=3)
   assert z['reused_tokens']==case['expected_reused_tokens'] and z['new_tokens']==case['expected_new_tokens']
   emit('quality',variant=name,case=case['id'],static_tokens=case['static_tokens'],body_tokens=case['body_tokens'],input=case['input'],reply=reply,metrics=z,top3=actions)
  # Separate fixed-feedback full-logit checks, never included in speed stats.
  for case in [fixtures[0],fixtures[3],fixtures[6],fixtures[9]]:
   list(w.generate(case['prefix_tokens'],1,reset=True))
   z=list(w.generate(case['tokens'],4,feedback=[100662]*4,diagnostic=True,timeout=60))[-1]
   assert z['reused_tokens']==case['expected_reused_tokens'] and z['new_tokens']==case['expected_new_tokens']
   f=dump_dir/(str(z['id'])+'-logits.f32');raw=f.read_bytes();arr=np.frombuffer(raw,dtype='<f4')
   assert len(raw)==4*151674*4 and np.isfinite(arr).all()
   assert z['generated_tokens']==4 and z['diagnostic'] and z['fixed_feedback']
   emit('diagnostic',variant=name,case=case['id'],static_tokens=case['static_tokens'],body_tokens=case['body_tokens'],metrics=z,logits_file=str(f.relative_to(out)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  emit('closed',variant=name,metrics=w.close());w=None
 emit('complete',audio_frames=front.frames,max_audio_lag_s=front.max_lag)
except BaseException:emit('error',traceback=traceback.format_exc());raise
finally:
 if w:w.abort()
 stop.set();front.stop.set();front.join(10)
 for th in threads:th.join(5)
 log.close()
