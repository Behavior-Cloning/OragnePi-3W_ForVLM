from pathlib import Path
import sys,json,hashlib,tarfile,time
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[2];S=Path('/mnt/c/Users/pc/AppData/Local/Temp/opi-vitfull-20261027-r1');sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
remote='/tmp/opi-vit224-a8-screen-r1';statefile=P/'controller-status.json';assert not statefile.exists();state={'status':'preparing','at':time.time()}
def save():state['updated']=time.time();statefile.write_text(json.dumps(state,indent=2)+'\n')
files={n:P/n for n in ['probe','board.py']}
for layer in [0,12]:
 for mode in ['base','input8','input-output8']:
  f=S/(f'w8-native224-r1-l{layer:02d}' if mode=='base' else f'a8-screen-r1-l{layer:02d}-{mode}')/'export_nbg_unify/network_binary.nb';files[f'l{layer:02d}-{mode}.nb']=f
 for im in [1,2]:
  f=P/f'l{layer:02d}-input{im}.f32';np.load(S/f'l{layer:02d}/input{im}.npy').astype('<f4').tofile(f);files[f.name]=f
  np.load(S/f'l{layer+1:02d}/input{im}.npy').astype('<f4').tofile(P/f'l{layer:02d}-reference{im}.f32')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();(P/'config.json').write_text(json.dumps({'sha256':{n:sha(f) for n,f in files.items()}},indent=2)+'\n');files['config.json']=P/'config.json';save()
try:
 with b.hostlock():
  locked('test ! -e '+remote+' && mkdir '+remote,check=True)
  state['status']='uploading';save()
  for n,f in files.items():upload(f,remote+'/'+n)
  locked('chmod +x '+remote+'/probe',check=True);state['status']='running';save()
  z=locked('cd '+remote+' && python3 board.py > controller.log 2>&1',capture_output=True,text=True,timeout=400)
  data=locked('cd '+remote+' && tar czf - results controller.log',capture_output=True,timeout=120,check=True).stdout;(P/'raw.tar.gz').write_bytes(data)
  with tarfile.open(P/'raw.tar.gz') as t:
   assert all(not m.issym() and not m.islnk() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(P)
  assert z.returncode==0,z.stderr
  assert json.loads((P/'results/summary.json').read_text())['status']=='complete'
  expected=''.join(sha(f)+'  '+remote+'/'+n+'\n' for n,f in files.items() if n.endswith('.nb'));locked('sha256sum -c -',input=expected,text=True,capture_output=True,check=True)
  locked('rm -- '+' '.join(remote+'/'+n for n in files if n.endswith('.nb')),check=True)
 state.update(status='complete',returncode=z.returncode);save();print(json.dumps(state))
except BaseException as e:state.update(status='failed-or-remote-unknown',error=str(e));save();raise
