from pathlib import Path
import sys,os,json,time,hashlib,threading,re,traceback
P=Path(__file__).resolve().parent
sys.path[:0]=['/tmp/opi-resident-vlm-r2/deps','/opt/opi-vlm/resident-agent-20261019/deps','/opt/opi-vlm/audio-lab-20261018/deps']
import numpy as np
from PIL import Image
from vision_worker import VisionWorker
O=P/'results';O.mkdir(exist_ok=False);cfg=json.loads((P/'config.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n,h in cfg['assets'].items():assert sha(n)==h,n
os.sched_setaffinity(0,{4,5});stop=threading.Event();errors=[]
def monitor():
 with (O/'resources.jsonl').open('w',buffering=1) as f:
  while not stop.wait(.5):
   mem=Path('/proc/meminfo').read_text();temps={str(p):int(p.read_text()) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
   f.write(json.dumps({'at':time.monotonic(),'meminfo':mem,'temps':temps,'vmstat':Path('/proc/vmstat').read_text()})+'\n')
   if int(re.search(r'MemAvailable:\s+(\d+)',mem).group(1))<96*1024 or max(temps.values())>=78000:errors.append('memory or thermal guard');return
thread=threading.Thread(target=monitor,daemon=True);thread.start();v=None;rows=[];summary={'status':'failed'}
try:
 v=VisionWorker(str(P/'vision-resident'),'/tmp/opi-vit-tower-r2/graphs',str(O/'embeddings'),str(O/'worker.log'))
 for index,image in enumerate([1,1,2,1,2,1,2]):
  assert not errors,errors
  start=time.monotonic()
  with Image.open(P/f'image{image}.jpg') as im:a=np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC),dtype=np.float32)
  a=(a/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32);pixels=a.transpose(2,0,1).copy().astype('<f4').tobytes();pre=time.monotonic()
  assert hashlib.sha256(pixels).hexdigest()==cfg['pixels'][str(image)]
  t=time.monotonic();m,data=v.encode(pixels);end=time.monotonic()
  h=hashlib.sha256(data).hexdigest();assert h==cfg['vectors'][str(image)]
  rows.append({'warmup':index==0,'image':image,'preprocess_s':pre-start,'vision_wall_s':end-t,'jpeg_to_vectors_s':end-start,'metrics':m,'sha256':h})
 summary={'status':'complete','ready':v.ready,'records':rows}
except BaseException:summary={'status':'failed','error':traceback.format_exc(),'records':rows};raise
finally:
 if v:summary['closed']=v.close()
 stop.set();thread.join(3);summary['monitor_errors']=errors;(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
