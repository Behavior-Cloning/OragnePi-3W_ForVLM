from pathlib import Path
import os,sys,json,time,hashlib,threading,re,traceback,math
P=Path(__file__).resolve().parent
sys.path[:0]=['/tmp/opi-resident-vlm-r2/deps','/opt/opi-vlm/resident-agent-20261019/deps','/opt/opi-vlm/audio-lab-20261018/deps']
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from vision_worker import VisionWorker
from resident_worker_visual import ResidentWorker
cfg=json.loads((P/'config.json').read_text());parent=json.loads((P/'parent-config.json').read_text());fixture=json.loads((P/'fixtures.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n,h in cfg['assets'].items():assert sha(P/n)==h,n
for n,h in parent['graphs_sha256'].items():assert sha(n)==h,n
required=490415304+1334594552+151674*896*4+256*1024**2
available=int(re.search(r'MemAvailable:\s*(\d+)',Path('/proc/meminfo').read_text()).group(1))*1024
assert available>=required,('memory gate',available,required)
O=P/'results';O.mkdir(exist_ok=False);events=(O/'events.jsonl').open('x',buffering=1);elock=threading.Lock();stop=threading.Event();vstop=threading.Event();errors=[];pids=[];state={'phase':'load'};v=None;w=None;vt=None;rows=[];vr=[];latest={};guard=threading.Lock();summary={}
def emit(x):
 with elock:events.write(json.dumps({'at':time.monotonic(),**x},ensure_ascii=False)+'\n')
def monitor():
 with (O/'resources.jsonl').open('x',buffering=1) as f:
  while not stop.wait(.5):
   mem=Path('/proc/meminfo').read_text();temps={str(q):int(q.read_text()) for q in Path('/sys/class/thermal').glob('thermal_zone*/temp')};procs={}
   for pid in list(pids):
    try:procs[str(pid)]={'stat':Path('/proc',str(pid),'stat').read_text(),'status':Path('/proc',str(pid),'status').read_text(),'smaps':Path('/proc',str(pid),'smaps_rollup').read_text()}
    except FileNotFoundError:pass
   f.write(json.dumps({'at':time.monotonic(),'phase':state['phase'],'meminfo':mem,'vmstat':Path('/proc/vmstat').read_text(),'cpu':Path('/proc/stat').read_text().splitlines()[0],'temps':temps,'processes':procs})+'\n')
   if int(re.search(r'MemAvailable:\s*(\d+)',mem).group(1))<96*1024 or max(temps.values())>=78000:errors.append('memory/thermal guard');vstop.set();return
mt=threading.Thread(target=monitor,daemon=True);mt.start();os.sched_setaffinity(0,{4,5})
tok=Tokenizer.from_file('/opt/opi-vlm/resident-agent-20261019/tokenizer.json');enc=lambda s:tok.encode(s,add_special_tokens=False).ids
sysbody=fixture['system'];ss='<|im_start|>system\n'+sysbody+'<|im_end|>\n';st=enc(ss);pre=enc(ss+'<|im_start|>user\n<img>');imageid=tok.token_to_id('<IMG_CONTEXT>');assert len(st)==35
cases=[]
for c in fixture['cases']:
 tail=enc('</img>\n'+c['question']+'<|im_end|>\n<|im_start|>assistant\n');c={**c,'tokens':pre+[imageid]*64+tail};assert 64+len(enc(c['question']))<=128;cases.append(c)
feedback=[enc('好')[0]]*32

def vision_once(c,phase,scheduled=None):
 assert not errors,errors;t0=time.monotonic()
 with Image.open(P/'images'/(c['id']+'.jpg')) as im:a=np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC),dtype=np.float32)
 a=(a/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32);pixels=a.transpose(2,0,1).copy().astype('<f4').tobytes();t1=time.monotonic();m,data=v.encode(pixels);t2=time.monotonic();h=hashlib.sha256(data).hexdigest();assert h==cfg['vectors'][c['id']],('visual numerical parity',c['id'])
 row={'kind':'vision','phase':phase,'case':c['id'],'start':t0,'dispatch':t1,'end':t2,'scheduled':scheduled,'metrics':m,'vector_sha256':h};vr.append(row);emit(row)
 with guard:latest.update(case=c['id'],at=t2,capture_at=t0,data=data)
 return data

def vision_loop(period,phase):
 try:
  deadline=time.monotonic();index=0
  while not vstop.is_set():
   if vstop.wait(max(0,deadline-time.monotonic())):break
   vision_once(cases[index%len(cases)],phase,deadline);index+=1;deadline+=period
   if deadline<time.monotonic():
    missed=math.floor((time.monotonic()-deadline)/period)+1;emit({'kind':'skip_stale_vision_ticks','phase':phase,'count':missed});deadline+=missed*period
 except BaseException:errors.append(traceback.format_exc());vstop.set()

def language_requests(phase,natural=False,count=12,use_latest=False):
 state['phase']=phase
 for index in range(count):
  assert not errors,errors;c=cases[index%len(cases)];list(w.generate(st,max_new=1,reset=True));vector=(P/'vectors'/(c['id']+'.f32')).read_bytes();age=None;vector_case=c['id']
  if use_latest:
   waitstart=time.monotonic();rejected=False
   while True:
    with guard:snapshot=dict(latest)
    age=time.monotonic()-snapshot.get('capture_at',-1e9)
    if snapshot and age<=4:break
    if not rejected:emit({'kind':'wait_fresh_image','phase':phase,'old_capture_age_s':age if snapshot else None});rejected=True
    assert not errors,errors
    if time.monotonic()-waitstart>6:raise TimeoutError('no fresh image within6s; stale input rejected')
    time.sleep(.02)
   c=next(x for x in cases if x['id']==snapshot['case']);vector=snapshot['data'];vector_case=c['id'];assert age<=4
  begin=time.monotonic();first=None
  for e in w.generate(c['tokens'],max_new=32,feedback=None if natural else feedback,visual=(len(pre),len(st),vector)):
   if first is None and e.get('event')=='token':first=time.monotonic()
   m=e
  end=time.monotonic();assert m['reused_tokens']==32
  if not natural:assert m['generated_tokens']==32 and m['fixed_feedback']
  row={'kind':'language','phase':phase,'case':c['id'],'index':index,'start':begin,'first':first,'end':end,'system_body':len(enc(sysbody)),'system_template':5,'system_total':35,'dynamic_image':64,'dynamic_text':len(enc(c['question'])),'template_boundaries':len(c['tokens'])-35-64-len(enc(c['question'])),'metrics':m,'answer':tok.decode(m['tokens'],skip_special_tokens=True),'fixed_feedback':not natural,'vector_case':vector_case,'fresh_wait_s':begin-waitstart if use_latest else 0,'vector_age_s':age,'frame_age_at_complete_s':end-snapshot['capture_at'] if use_latest else None}
  rows.append(row);emit(row);(O/'progress.json').write_text(json.dumps({'phase':phase,'language_rows':len(rows),'vision_rows':len(vr),'at':time.time()}))
try:
 graph=parent['parent_graph'];env=['OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1','OPI_TTFT_TAIL_MIN=8','OPI_GREEDY_I16_SINGLE_PASS=1','OPI_PREFILL8_DIR='+parent['parent_tail'],'OPI_COMPACT64_DIR='+parent['compact_graph']]
 w=ResidentWorker(['env',*env,'taskset','-c','7',str(P/'resident-visual'),graph,'24','448','151674',graph+'/embeddings.f32','32','151645'],'/opt/opi-vlm/resident-agent-20261019',str(O/'language.log'),timeout=180);pids.append(w.process.pid);emit({'kind':'language_ready','metrics':w.ready})
 # No repeat of r2 fixed-feedback matrix; this run validates freshness handling.
 assert not errors,errors
 v=VisionWorker(str(P/'vision-profile'),str(P/'graphs'),str(O/'scratch'),str(O/'vision.log'));pids.append(v.p.pid);emit({'kind':'vision_ready','metrics':v.ready})
 for period in [5,2]:
  phase='fresh-period'+str(period);state['phase']=phase
  with guard:latest.clear();latest.update(case=cases[0]['id'],at=time.monotonic()-30,capture_at=time.monotonic()-31,data=(P/'vectors'/(cases[0]['id']+'.f32')).read_bytes())
  emit({'kind':'inject_stale_cache_age','phase':phase,'seconds':31,'synthetic_timestamp':True})
  vstop.clear();vt=threading.Thread(target=vision_loop,args=(period,phase),daemon=True);vt.start()
  try:language_requests(phase,natural=True,count=6,use_latest=True)
  finally:vstop.set();vt.join(180);assert not vt.is_alive();vt=None
 summary={'status':'complete','records':rows,'vision_records':vr,'scope':'both NPU models resident; 12 natural requests; capture-time TTL4s; 31s stale cache injection before each5s/2s phase; no audio/RAG'}
except BaseException:summary={'status':'failed','error':traceback.format_exc(),'records':rows,'vision_records':vr};raise
finally:
 vstop.set()
 if vt:vt.join(180)
 ce=[]
 for name,worker in [('language',w),('vision',v)]:
  if worker:
   try:summary[name+'_closed']=worker.close();emit({'kind':name+'_closed','metrics':summary[name+'_closed']})
   except BaseException as exc:ce.append(name+': '+repr(exc))
 stop.set();mt.join(3);summary.update(monitor_errors=errors,cleanup_errors=ce)
 if errors or ce:summary['status']='failed'
 (O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');events.close()
