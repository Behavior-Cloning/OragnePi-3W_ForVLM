import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,time,json,hashlib,platform,random,threading,resource
from pathlib import Path
import numpy as np,psutil,torch
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
ROOT=Path(__file__).resolve().parent
key=sys.argv[1];out=ROOT/'results'/key;out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
proc=psutil.Process();os.sched_setaffinity(0,{6,7})
cards=json.loads((ROOT/'cards.json').read_text());cases=json.loads((ROOT/'cases.json').read_text())
assert len(cards)==55 and len(cases)==60
docs=[a['name'].removeprefix('skill-')+'。用途：'+a['description'] for a in cards]
byid={a['id']:i for i,a in enumerate(cards)}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def snap():
 mem=proc.memory_full_info();s=psutil.swap_memory()
 return dict(time=time.time(),rss=mem.rss,pss=mem.pss,available=psutil.virtual_memory().available,temperature=max(float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')),cpu_ms=(proc.cpu_times().user+proc.cpu_times().system)*1000,loadavg=os.getloadavg(),swap_in=s.sin,swap_out=s.sout,cpu_freq={p.parents[1].name:int(p.read_text()) for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq')})
samples=[];stop=threading.Event();guard=[];phase='loading'
def monitor():
 with (out/'telemetry.jsonl').open('x') as f:
  while not stop.is_set():
   s=snap();s['phase']=phase;samples.append(s);f.write(json.dumps(s)+'\n');f.flush()
   if s['temperature']>=75:guard.append('temperature >=75C')
   if s['available']<350*1024**2:guard.append('available memory <350MiB')
   stop.wait(1)
def check():
 if guard:raise RuntimeError(guard[-1])
cooling=[]
def cool(force=False):
 global phase
 temp=max(float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'))
 if temp<(58 if force else 65):return
 old=phase;phase='cooling';t=time.perf_counter();before=temp
 print('COOLING',key,before,flush=True)
 while temp>58:
  check();time.sleep(1);temp=max(float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'))
  if time.perf_counter()-t>300:raise RuntimeError('cooling timeout')
 event=dict(after_phase=old,start_temp=before,end_temp=temp,seconds=time.perf_counter()-t);cooling.append(event)
 (out/'cooling.json').write_text(json.dumps(cooling,indent=2));phase=old
 print('COOLED',key,round(event['seconds'],2),temp,flush=True)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
def load(path,classification=False):
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left' if key=='qwen06' else 'right')
 cls=AutoModelForSequenceClassification if classification else AutoModel
 model,info=cls.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True,low_cpu_mem_usage=True)
 missing=[x for x in info.get('missing_keys',[]) if 'pooler' not in x]
 assert not missing,('missing parameters',missing)
 assert not info.get('mismatched_keys'),info
 print('LOADED',str(path),'parameters',sum(p.numel() for p in model.parameters()),'info',info,flush=True)
 return tok,model.eval(),info
modelpath=ROOT/'models'/key if key!='bge_small' else ROOT.parent/'model'
if key!='bge_small':
 manifest=json.loads((modelpath/'download-manifest.json').read_text())
 for f in manifest['files']:assert sha(modelpath/f['Path'])==f['Sha256'],f['Path']
load0=time.perf_counter()
if key=='reranker_base':
 tok,model,loadinfo=load(ROOT.parent/'model');rtok,reranker,rerankinfo=load(modelpath,True)
else:tok,model,loadinfo=load(modelpath)
load_seconds=time.perf_counter()-load0
params=sum(p.numel() for p in model.parameters())+(sum(p.numel() for p in reranker.parameters()) if key=='reranker_base' else 0)
def encode(texts,query=False):
 if query:
  if key=='qwen06':texts=['Instruct: Given a user high-level intent, retrieve robot skills that perform the requested action.\nQuery: '+x for x in texts]
  elif key!='gte_base':texts=['为这个句子生成表示以用于检索相关文章：'+x for x in texts]
 batch=tok(texts,padding=True,truncation=False,return_tensors='pt')
 assert batch['input_ids'].shape[1]<=512,('unexpected length',batch['input_ids'].shape)
 with torch.inference_mode():
  opts={'use_cache':False} if key=='qwen06' else {}
  states=model(**batch,**opts).last_hidden_state
  v=states[:,-1] if key=='qwen06' else states[:,0]
  return torch.nn.functional.normalize(v,p=2,dim=1).float().numpy(),int(batch['input_ids'].shape[1])
phase='indexing';ix0=time.perf_counter();parts=[];bs=1 if key=='qwen06' else 4
cool(force=True)
for i in range(0,len(cards),bs):
 check();cool();v,_=encode(docs[i:i+bs]);parts.append(v)
 if (i+bs)%32==0:print('INDEX',key,min(i+bs,len(cards)),len(cards),flush=True)
vectors=np.ascontiguousarray(np.concatenate(parts));np.save(out/'vectors.npy',vectors);index_seconds=time.perf_counter()-ix0
def feasible(a,c):
 tags=set(a['tags'])
 if 'allowed_actuators' in c and not set(a['actuators']).issubset(c['allowed_actuators']):return False
 if c.get('no_locomotion') and 'locomotion' in tags:return False
 if c.get('quiet') and 'noise' in tags:return False
 if c.get('low_energy') and a['energy']=='high':return False
 if tags.intersection(c.get('forbid_tags',[])):return False
 if c.get('target_available') is False and a['requires_target']:return False
 return True
def retrieve(c,pool,rerank_k=0):
 cpu0=time.process_time_ns();t0=time.perf_counter_ns();v,length=encode([c['query']],True);t1=time.perf_counter_ns()
 scores=vectors@v[0];order=[i for i in np.argsort(-scores,kind='stable').tolist() if cards[i]['type']==pool];t2=time.perf_counter_ns();stage1=order.copy();pairlength=0
 if rerank_k:
  selected=order[:rerank_k]
  # Batch four pairs to bound memory; timing includes tokenization and all batches.
  chunks=[]
  for i in range(0,len(selected),4):
   ids=selected[i:i+4];batch=rtok([c['query']]*len(ids),[docs[j] for j in ids],padding=True,truncation=False,return_tensors='pt')
   pairlength=max(pairlength,int(batch['input_ids'].shape[1]));assert pairlength<=512
   with torch.inference_mode():chunks.append(reranker(**batch).logits.view(-1).float().numpy())
  rs=np.concatenate(chunks);perm=np.argsort(-rs,kind='stable');order=[selected[i] for i in perm];scores=np.full(len(cards),-np.inf)
  for j,s in zip(selected,rs):scores[j]=s
 t3=time.perf_counter_ns()
 # Independent full query invocation; both raw and filtered rankings are captured.
 ctx=c['context']
 if ctx.get('emergency') or ctx.get('unsupported_request'):
  raise ValueError('Control/unsupported requests belong to the external execution layer, not the action retrieval benchmark')
 filtered=[i for i in order if feasible(cards[i],ctx)]
 top5=[{'id':cards[i]['id'],'score':float(scores[i]) if np.isfinite(scores[i]) else None} for i in filtered[:5]]
 t4=time.perf_counter_ns();cpu1=time.process_time_ns()
 return dict(total_ms=(t4-t0)/1e6,encode_ms=(t1-t0)/1e6,search_ms=(t2-t1)/1e6,rerank_ms=(t3-t2)/1e6,filter_result_ms=(t4-t3)/1e6,cpu_ms=(cpu1-cpu0)/1e6,query_tokens=length,pair_max_tokens=pairlength,top5=top5,order=[cards[i]['id'] for i in filtered],raw_order=[cards[i]['id'] for i in order],stage1_order=[cards[i]['id'] for i in stage1])

phase='warmup';pools=['atom','skill']
for c in cases[:3]:
 for pool in pools:check();cool();retrieve(c,pool)
for th in proc.threads():os.sched_setaffinity(th.id,{6,7})
phase='timing';cool(force=True);start=snap();rows=[]
schedule=[(j,rep,pool) for rep in range(2) for j in range(len(cases)) for pool in pools]
random.Random(20261022).shuffle(schedule);loop0=time.perf_counter()
with (out/'raw.jsonl').open('x') as f:
 for n,(j,rep,pool) in enumerate(schedule):
  check();cool();c=cases[j];row=dict(model=key,case_id=c['id'],repeat=rep,pool=pool,**retrieve(c,pool));rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
  if (n+1)%20==0:print('PROGRESS',key,n+1,len(schedule),'TEMP',snap()['temperature'],flush=True)
loop_seconds=time.perf_counter()-loop0;end=snap();phase='finished';stop.set();thread.join()
def stats(vals):return dict(mean=float(np.mean(vals)),p50=float(np.percentile(vals,50)),p95=float(np.percentile(vals,95)),max=float(max(vals)))
summary=[]
for pool in pools:
 group=[r for r in rows if r['pool']==pool];quality=[r for r in group if r['repeat']==0]
 assert len(group)==120 and len(quality)==60
 stable=sum(all(x['order']==q['order'] for x in group if x['case_id']==q['case_id']) for q in quality)
 summary.append(dict(model=key,pool=pool,candidates=sum(c['type']==pool for c in cards),timing_n=len(group),quality_n=60,stable_rankings=stable,average_cpu_cores=sum(x['cpu_ms'] for x in group)/sum(x['total_ms'] for x in group),**{metric:stats([x[metric] for x in group]) for metric in ['total_ms','encode_ms','search_ms','filter_result_ms','cpu_ms','query_tokens']}))
timing_samples=[s for s in samples if start['time']<=s['time']<=end['time']]+[start,end]
result=dict(complete=True,model=key,parameters=params,platform=platform.platform(),torch=torch.__version__,dtype='float32',threads=2,affinity=sorted(os.sched_getaffinity(0)),n=len(cards),quality_n=60,load_seconds=load_seconds,index_seconds=index_seconds,loop_seconds=loop_seconds,index_bytes=vectors.nbytes,summary=summary,load_info=loadinfo,pss_peak=max(s['pss'] for s in samples),rss_peak_os=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,timing_pss_peak=max(s['pss'] for s in timing_samples),temperature_start=start['temperature'],temperature_max=max(s['temperature'] for s in timing_samples),temperature_end=end['temperature'],available_min=min(s['available'] for s in samples),swap_in_delta=end['swap_in']-start['swap_in'],swap_out_delta=end['swap_out']-start['swap_out'],source_sha={n:sha(ROOT/n) for n in ['board_bench.py','cards.json','cases.json']},output_sha={'raw.jsonl':sha(out/'raw.jsonl'),'vectors.npy':sha(out/'vectors.npy')},note='Fresh query tokenization+FP32 encoding+55 dot scores+type selection+supplied-state filtering+Top5. Atom and skill calls independently encoded. No ASR, LLM, actual motors or language state parsing. Temperature cooldown outside single-call latency.')
result.update(cooling_events=cooling,cooling_seconds=sum(x['seconds'] for x in cooling),timing_cooling_seconds=sum(x['seconds'] for x in cooling if x['after_phase']=='timing'))
(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('COMPLETE',key,flush=True)
