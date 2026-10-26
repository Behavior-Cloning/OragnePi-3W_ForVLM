from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[2]/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
remote='/tmp/opi-resident-npu-vlm-r1'
files={n:P/n for n in ['board_pipeline.py','vision_worker.py','resident_worker_visual.py','preflight.py','config.json','parent-config.json','vqa-cases.json']}
for i in [1,2]:files[f'image{i}.jpg']=P.parent/f'fixtures/image{i}.jpg'
(P/'prepared-files.json').write_text(json.dumps({n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()},indent=2)+'\n')
with b.hostlock():
 locked('mkdir '+remote,check=True)
 for n,f in files.items():upload(f,remote+'/'+n)
 z=locked('cd '+remote+' && python3 preflight.py',capture_output=True,text=True,check=True,timeout=30)
(P/'board-preflight.json').write_text(z.stdout);print(z.stdout)
