from pathlib import Path
import sys,json,hashlib,tarfile,time,subprocess
P=Path(__file__).resolve().parent;D=P.parent;R=P.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
remote='/tmp/opi-vit224-concurrent-r1';status=P/'controller-status.json';assert not status.exists()
state={'status':'prepare','at':time.time(),'remote':remote}
def save():state['updated']=time.time();status.write_text(json.dumps(state,indent=2)+'\n')
files={n:P/n for n in ['board_concurrent.py','vision-profile','vision_worker.py','resident_worker_visual.py','fixtures.json']}
files.update({'resident-visual':D/'npu-vision-input-r1/runtime/resident-visual','parent-config.json':D/'npu-vision-input-r1/parent-config.json'})
old=D/'vqa-expanded-w8-r1';refs=json.loads((old/'vision-224/summary.json').read_text())['records'];refhash={r['id']:r['vector_sha256'] for r in refs if not r['warmup']}
for c in json.loads((P/'fixtures.json').read_text())['cases']:
 for ext,folder,base in [('jpg','images',old/'images'),('f32','vectors',old/'vision-224/vectors')]:files[folder+'/'+c['id']+'.'+ext]=base/(c['id']+'.'+ext)
graphs={c['name']:Path(c['path']) for c in json.loads((D/'vision224-w8-npu-r1/build-status.json').read_text())['cases']};graphs.update(stem=D/'vit-full-r1/endpoints-nbg/stem.nb',projector=D/'vit-full-r1/selective-projector-norm-gelu.nb')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();assets={n:sha(f) for n,f in files.items()};assets.update({'graphs/'+n+'.nb':sha(f) for n,f in graphs.items()})
(P/'config.json').write_text(json.dumps({'assets':assets,'vectors':refhash,'cpu_affinity':{'vision':6,'language':7,'python':[4,5]}},indent=2)+'\n');files['config.json']=P/'config.json'
with tarfile.open(P/'payload.tar.gz','w:gz') as t:
 for n,f in files.items():t.add(f,arcname=n)
save()
try:
 with b.hostlock():
  z=locked('cat /sys/kernel/debug/viplite/mem_profile; systemctl show opi-vlm-model.service -p ActiveState --value',capture_output=True,text=True,check=True);assert '0x00000000' in z.stdout and z.stdout.strip().endswith('inactive')
  locked('test ! -e '+remote+' && mkdir -p '+remote+'/graphs',check=True)
  upload(P/'payload.tar.gz',remote+'/payload.tar.gz');locked('cd '+remote+' && tar xzf payload.tar.gz && rm payload.tar.gz && chmod +x vision-profile resident-visual',check=True)
  state['status']='uploading';save()
  for n,f in graphs.items():upload(f,remote+'/graphs/'+n+'.nb')
  state['status']='running';save()
  z=locked('cd '+remote+' && python3 board_concurrent.py > controller.log 2>&1',capture_output=True,text=True,timeout=1000)
  state.update(status='collecting',returncode=z.returncode);save()
  data=locked('cd '+remote+' && tar czf - results controller.log',capture_output=True,timeout=120,check=True).stdout;(P/'raw.tar.gz').write_bytes(data)
  with tarfile.open(P/'raw.tar.gz') as t:
   assert all(not m.issym() and not m.islnk() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(P)
  x=json.loads((P/'results/summary.json').read_text());assert z.returncode==0 and x['status']=='complete'
  assert x['vision_closed']['driver_bytes']==0
  expected=''.join(sha(f)+'  '+remote+'/graphs/'+n+'.nb\n' for n,f in graphs.items());locked('sha256sum -c -',input=expected,text=True,capture_output=True,check=True)
  locked('rm -- '+' '.join(remote+'/graphs/'+n+'.nb' for n in graphs),check=True)
  health=locked('cat /proc/meminfo; cat /sys/kernel/debug/viplite/mem_profile',capture_output=True,text=True,check=True);(P/'post-health.txt').write_text(health.stdout)
  state.update(status='complete',ended=time.time());save();print(json.dumps(state))
except BaseException as e:state.update(status='failed-or-remote-unknown',error=str(e));save();raise
