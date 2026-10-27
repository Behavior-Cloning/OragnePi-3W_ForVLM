from pathlib import Path
import os,sys,json,time,hashlib,threading,re,traceback,importlib
P=Path(__file__).resolve().parent;side=int(sys.argv[1]);assert side in [112,224]
sys.path[:0]=['/tmp/opi-resident-vlm-r2/deps','/opt/opi-vlm/resident-agent-20261019/deps','/opt/opi-vlm/audio-lab-20261018/deps']
import numpy as np
from PIL import Image
VisionWorker=importlib.import_module('vision_worker'+str(side)).VisionWorker
cfg=json.loads((P/'config.json').read_text());cases=json.loads((P/'vqa-cases.json').read_text())['cases'];O=P/f'vision-{side}';O.mkdir(exist_ok=False);(O/'vectors').mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n,h in cfg['assets'].items():assert sha(P/n)==h,n
os.sched_setaffinity(0,{4,5});stop=threading.Event();errors=[]
def monitor():
 with (O/'resources.jsonl').open('w',buffering=1) as f:
  while not stop.wait(.5):
   mem=Path('/proc/meminfo').read_text();temps={str(p):int(p.read_text()) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')};f.write(json.dumps({'at':time.monotonic(),'meminfo':mem,'temps':temps,'cpu':Path('/proc/stat').read_text().splitlines()[0]})+'\n')
   if int(re.search(r'MemAvailable:\s*(\d+)',mem).group(1))<96*1024 or max(temps.values())>=78000:errors.append('memory/thermal guard');return
th=threading.Thread(target=monitor,daemon=True);th.start();v=None;rows=[];summary={'status':'failed'}
try:
 v=VisionWorker(str(P/'vision-resident'),str(P/'graphs'),str(O/'scratch'),str(O/'worker.log'))
 for index,c in enumerate(cases[:3]+cases):
  assert not errors,errors;start=time.monotonic()
  with Image.open(P/'images'/(c['id']+'.jpg')) as im:a=np.asarray(im.convert('RGB').resize((side,side),Image.Resampling.BICUBIC),dtype=np.float32)
  a=(a/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32);pixels=a.transpose(2,0,1).copy().astype('<f4').tobytes();pre=time.monotonic()
  m,data=v.encode(pixels);end=time.monotonic();h=hashlib.sha256(data).hexdigest()
  if index>=3:(O/'vectors'/(c['id']+'.f32')).write_bytes(data)
  row={'id':c['id'],'warmup':index<3,'side':side,'preprocess_s':pre-start,'vision_wall_s':end-pre,'jpeg_to_vectors_s':end-start,'metrics':m,'pixels_sha256':hashlib.sha256(pixels).hexdigest(),'vector_sha256':h};rows.append(row)
  (O/'progress.json').write_text(json.dumps({'completed':max(0,index-2),'total':len(cases)}))
 summary={'status':'complete','ready':v.ready,'records':rows}
except BaseException:summary={'status':'failed','error':traceback.format_exc(),'records':rows};raise
finally:
 if v:summary['closed']=v.close()
 stop.set();th.join(3);summary['monitor_errors']=errors;(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
