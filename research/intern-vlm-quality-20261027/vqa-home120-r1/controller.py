from pathlib import Path
import sys,json,hashlib,tarfile,time,shutil,re
P=Path(__file__).resolve().parent;D=P.parent;R=P.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
remote='/tmp/opi-vqa-home120-r1';status=P/'controller-status.json';assert not status.exists()
state={'status':'preparing','at':time.time(),'remote':remote}
def save():status.write_text(json.dumps(state,indent=2)+'\n')
def collect(names,label):
 data=locked('cd '+remote+' && tar czf - '+' '.join(names),capture_output=True,check=True,timeout=120).stdout
 f=P/(label+'.tar.gz');f.write_bytes(data)
 with tarfile.open(f) as t:
  assert all(not m.issym() and not m.islnk() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(P)
files={n:P/n for n in ['board_vision.py','board_vqa.py','vision_worker224.py','vqa-cases.json']}
files.update({'vision-resident224':D/'vit-full-r1/vision-resident','resident-visual':D/'npu-vision-input-r1/runtime/resident-visual','resident_worker_visual.py':D/'npu-vision-input-r1/resident_worker_visual.py','parent-config.json':D/'npu-vision-input-r1/parent-config.json'})
for c in json.loads((P/'vqa-cases.json').read_text())['cases']:files['images/'+c['id']+'.jpg']=P/'images'/(c['id']+'.jpg')
hashes={n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()};(P/'upload-sha256.json').write_text(json.dumps(hashes,indent=2))
with tarfile.open(P/'payload.tar.gz','w:gz') as t:
 for n,f in files.items():t.add(f,arcname=n)
save()
try:
 with b.hostlock():
  active=locked('systemctl show opi-vlm-model.service -p ActiveState --value',capture_output=True,text=True,check=True).stdout.strip();assert active=='inactive'
  locked('test ! -e '+remote+' && mkdir '+remote+' && ln -s /tmp/opi-vit224-concurrent-r1/graphs '+remote+'/graphs',check=True)
  upload(P/'payload.tar.gz',remote+'/payload.tar.gz');locked('cd '+remote+' && tar xzf payload.tar.gz && rm payload.tar.gz && chmod +x vision-resident224 resident-visual',check=True)
  expected=''.join(h+'  '+remote+'/'+n+'\n' for n,h in hashes.items());locked('sha256sum -c -',input=expected,text=True,capture_output=True,check=True,timeout=90)
  for side in [224]:
   state.update(status='uploading-graphs',side=side);save()
   build=json.loads((P/'build-status.json' if side==112 else D/'vision224-w8-npu-r1/build-status.json').read_text());assert build['state']=='compiled-not-board-validated' and len(build['cases'])==24
   graphs={c['name']:Path(c['path']) for c in build['cases']}
   if side==112:
    endpoints=json.loads((D/'vision112-npu16-r1/build-status.json').read_text())['cases'];graphs.update({c['name']:Path(c['path']) for c in endpoints if c['name'] in ['stem','projector']})
   else:graphs.update(stem=D/'vit-full-r1/endpoints-nbg/stem.nb',projector=D/'vit-full-r1/selective-projector-norm-gelu.nb')
   total=sum(f.stat().st_size for f in graphs.values());mem=locked('cat /proc/meminfo; df -B1 --output=avail /tmp | tail -1',capture_output=True,text=True,check=True).stdout
   assert int(re.search(r'MemAvailable:\s*(\d+)',mem).group(1))*1024>total+1100*1024**2
   assert int(mem.splitlines()[-1])>total+96*1024**2
   gh={n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in graphs.items()}
   # Reuse frozen W8 graph files from completed concurrent run; SHA verified in board script.
   assets={**hashes,**{'graphs/'+n+'.nb':h for n,h in gh.items()}}
   cfg={'assets':assets,'side':side,'precision':'144 W8 per-channel FC weights; INT16 boundaries and FP16 sensitive operations; original stem/projector'}
   f=P/f'config-{side}.json';f.write_text(json.dumps(cfg,indent=2));upload(f,remote+'/config.json')
   locked('cp '+remote+'/vision-resident'+str(side)+' '+remote+'/vision-resident',check=True)
   state['status']='vision-running';save()
   z=locked('cd '+remote+' && python3 board_vision.py '+str(side)+' > vision-'+str(side)+'.log 2>&1',capture_output=True,text=True,timeout=600)
   collect(['vision-'+str(side),'vision-'+str(side)+'.log'],'vision-'+str(side)+'-raw');assert z.returncode==0,z.stderr
   assert json.loads((P/f'vision-{side}/summary.json').read_text())['status']=='complete'
   # Keep only archived vectors; remove this phase's verified graph copies before NPU language load.
   expected=''.join(h+'  '+remote+'/graphs/'+n+'.nb\n' for n,h in gh.items());locked('sha256sum -c -',input=expected,text=True,capture_output=True,check=True,timeout=90)
   # Shared graph directory retained for subsequent NPU optimization runs.
   state['status']='language-running';save()
   z=locked('cd '+remote+' && python3 board_vqa.py '+str(side)+' > vqa-'+str(side)+'.log 2>&1',capture_output=True,text=True,timeout=900)
   collect(['vqa-'+str(side),'vqa-'+str(side)+'.log','vqa-frozen-'+str(side)+'.json'],'vqa-'+str(side)+'-raw');assert z.returncode==0,z.stderr
   assert json.loads((P/f'vqa-{side}/summary.json').read_text())['status']=='complete'
  state.update(status='complete',ended=time.time());save();print(json.dumps(state))
except BaseException as e:state.update(status='failed-or-remote-unknown',error=str(e));save();raise
