from pathlib import Path
import os,sys,json,time,hashlib,subprocess,select,threading,re,traceback
P=Path(__file__).resolve().parent
sys.path[:0]=[str(P/'deps'),'/opt/opi-vlm/resident-agent-20261019/deps','/opt/opi-vlm/audio-lab-20261018/deps']
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from resident_worker_visual import ResidentWorker
O=P/'results';O.mkdir(exist_ok=False);events=(O/'events.jsonl').open('x',buffering=1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((P/'config.json').read_text());cases=json.loads((P/'vqa-cases.json').read_text());parent=json.loads((P/'parent-config.json').read_text())
for path,h in cfg['assets'].items():assert sha(path)==h,path
for path,h in parent['graphs_sha256'].items():assert sha(path)==h,path
for name,h in cfg['images'].items():assert sha(P/name)==h
os.sched_setaffinity(0,{4,5});stop=threading.Event();errors=[];pids=[];v=None;w=None
state={'phase':'loading'}
def emit(x):events.write(json.dumps({'at':time.monotonic(),**x},ensure_ascii=False)+'\n')
def monitor():
 with (O/'resources.jsonl').open('w',buffering=1) as f:
  while not stop.wait(.5):
   mem=Path('/proc/meminfo').read_text();temps={str(p):int(p.read_text()) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')};proc={}
   for pid in list(pids):
    try:proc[str(pid)]={'status':Path('/proc',str(pid),'status').read_text(),'smaps':Path('/proc',str(pid),'smaps_rollup').read_text()}
    except FileNotFoundError:pass
   f.write(json.dumps({'at':time.monotonic(),'phase':state['phase'],'meminfo':mem,'vmstat':Path('/proc/vmstat').read_text(),'cpu':Path('/proc/stat').read_text().splitlines()[0],'temperatures':temps,'processes':proc})+'\n')
   available=int(re.search(r'^MemAvailable:\s*(\d+)',mem,re.M).group(1))
   if available<96*1024 or max(temps.values())>=78000:errors.append('memory or thermal guard');return
thread=threading.Thread(target=monitor,daemon=True);thread.start()
def vevent(timeout=120):
 if not select.select([v.stdout],[],[],timeout)[0]:raise TimeoutError('vision event timeout')
 line=v.stdout.readline()
 if not line:raise RuntimeError('vision process closed')
 return json.loads(line)
def vreq(q):v.stdin.write(json.dumps(q)+'\n');v.stdin.flush();return vevent()
records=[];summary={'status':'failed'}
try:
 env={k:z for k,z in os.environ.items() if not k.startswith('OPI_')};env['OPI_VISION_Q8_REPACK']='1'
 err=(O/'vision.stderr').open('w');v=subprocess.Popen(['taskset','-c','6,7',cfg['vision_binary'],cfg['vision_model'],'native'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,env=env);pids.append(v.pid)
 emit({'kind':'vision_ready','metrics':vevent()})
 graph=parent['parent_graph'];envargs=['OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1','OPI_TTFT_TAIL_MIN=8','OPI_GREEDY_I16_SINGLE_PASS=1','OPI_PREFILL8_DIR='+parent['parent_tail'],'OPI_COMPACT64_DIR='+parent['compact_graph']]
 w=ResidentWorker(['env',*envargs,'taskset','-c','7',cfg['language_binary'],graph,'24','448','151674',graph+'/embeddings.f32','32','151645'],'/opt/opi-vlm/resident-agent-20261019',str(O/'language.log'),timeout=180);pids.append(w.process.pid);emit({'kind':'language_ready','metrics':w.ready})
 tok=Tokenizer.from_file('/opt/opi-vlm/resident-agent-20261019/tokenizer.json');enc=lambda x:tok.encode(x,add_special_tokens=False).ids
 sysbody=cases['system'];sysstr='<|im_start|>system\n'+sysbody+'<|im_end|>\n';st=enc(sysstr)
 list(w.generate(st,max_new=1,reset=True));emit({'kind':'system_cache_prepared','system_body':len(enc(sysbody)),'system_total':len(st),'reusable_aligned_tokens':32})
 sequence=[cases['cases'][0]]+cases['cases']*2
 for index,c in enumerate(sequence):
  assert not errors,errors;state['phase']='warmup' if index==0 else 'measure'
  begin=time.monotonic()
  im=Image.open(P/(c['image']+'.jpg')).convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
  arr=(np.asarray(im,dtype=np.float32)/255.0-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
  raw=O/'current-pixels.f32';arr.astype('<f4').tofile(raw);preprocess_end=time.monotonic()
  # Validation hash is recorded separately from production stage timing.
  pixelhash=sha(raw);assert pixelhash==cfg['pixels'][c['image']]
  hdone=time.monotonic();emb=O/'current-embedding.f32'
  ve=vreq({'id':str(index),'side':224,'threads':2,'input':str(raw),'output':str(emb)});assert ve.get('event')=='encoded',ve
  vision_end=time.monotonic();vectors=emb.read_bytes();assert len(vectors)==64*896*4
  pre=enc(sysstr+'<|im_start|>user\n<img>');tail=enc('</img>\n'+c['question']+'<|im_end|>\n<|im_start|>assistant\n');tokens=pre+[tok.token_to_id('<IMG_CONTEXT>')]*64+tail
  body=len(enc(c['question']));assert 64+body<=128
  first=None;ev=[];dispatch=time.monotonic()
  for e in w.generate(tokens,max_new=32,visual=(len(pre),len(st),vectors)):
   if first is None and e.get('event')=='token':first=time.monotonic()
   ev.append(e)
  end=time.monotonic();m=ev[-1];assert m['reused_tokens']==32
  row={'kind':'result','warmup':index==0,'index':index,'case':c['id'],'answer':tok.decode(m['tokens'],skip_special_tokens=True),'system_body':len(enc(sysbody)),'system_template':5,'system_total':len(st),'dynamic_image':64,'dynamic_text':body,'template_boundaries':len(tokens)-len(st)-64-body,'metrics':m,'vision_metrics':ve,'preprocess_seconds':preprocess_end-begin,'validation_hash_seconds':hdone-preprocess_end,'vision_wall_seconds':vision_end-hdone,'embedding_read_tokenize_seconds':dispatch-vision_end,'language_first_event_seconds':first-dispatch,'end_to_end_first_seconds':first-begin,'end_to_end_full_seconds':end-begin,'embedding_sha256':hashlib.sha256(vectors).hexdigest()}
  records.append(row);emit(row)
 summary={'status':'complete','scope':'resident CPU Q8 repacked vision + resident Intern NPU text; image decode/resize through first/full response; no audio/RAG/motors','records':records}
except BaseException:summary={'status':'failed','error':traceback.format_exc(),'records':records};raise
finally:
 if w:
  summary['language_closed']=w.close();emit(summary['language_closed'])
 if v and v.poll() is None:
  try:summary['vision_closed']=vreq({'op':'close'});v.wait(timeout=20)
  except BaseException:v.kill();v.wait();raise
 stop.set();thread.join(3);summary['monitor_errors']=errors;(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');events.close()
