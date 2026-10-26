from pathlib import Path
import sys,tarfile,json,time,hashlib,numpy as np
P=Path(__file__).resolve().parent;R=P.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
state=P/'matrix-controller.json';assert not state.exists();remote='/tmp/opi-vit-matrix-r1'
def save(status,**kw):state.write_text(json.dumps({'status':status,'remote':remote,'at':time.time(),**kw},indent=2)+'\n')
save('preparing')
files={'vit-matrix-probe':P/'vit-matrix-probe','board_matrix.py':P/'board_matrix.py'}
for side in [112,224]:
 for dtype in ['fp16','int8','int16']:
  files['s%d-%s.nb'%(side,dtype)]=P/('s%d-%s'%(side,dtype))/'export_nbg_unify/network_binary.nb'
 for image in [1,2]:
  f=P/('s%d-input%d.f32'%(side,image));np.load(P/('s%d/input%d.npy'%(side,image))).astype('float32').tofile(f);files[f.name]=f
config={'sha256':{n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()}}
(P/'matrix-config.json').write_text(json.dumps(config,indent=2)+'\n');files['matrix-config.json']=P/'matrix-config.json'
with tarfile.open(P/'matrix-upload.tar','w') as tar:
 for name,path in files.items():tar.add(path,arcname=name)
with b.hostlock():
 locked('mkdir '+remote,check=True);upload(P/'matrix-upload.tar',remote+'/upload.tar')
 locked('cd '+remote+' && tar xf upload.tar && chmod +x vit-matrix-probe',check=True)
 save('running')
 z=locked('cd '+remote+' && python3 board_matrix.py > controller.log 2>&1',capture_output=True,text=True,timeout=600)
 save('collecting',returncode=z.returncode)
 data=locked('cd '+remote+' && tar czf - matrix-results controller.log',capture_output=True,timeout=90).stdout
 (P/'matrix-results.tar.gz').write_bytes(data);save('complete' if z.returncode==0 else 'failed',returncode=z.returncode)
with tarfile.open(P/'matrix-results.tar.gz') as tar:tar.extractall(P,filter='data')
print(state.read_text());print((P/'controller.log').read_text()[-1000:])
