from pathlib import Path
import sys,os,json,time,hashlib,shlex,numpy as np,tarfile
P=Path(__file__).resolve().parent;R=P.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
compiled=json.loads((P/'compile-full-status.json').read_text());assert compiled['status']=='compiled_not_board_validated' and len(compiled['cases'])==24
state=P/'board-stack-status.json';assert not state.exists();remote='/tmp/opi-vit-full-r1'
def save(s,**kw):state.write_text(json.dumps({'status':s,'at':time.time(),'pid':os.getpid(),'remote':remote,**kw},indent=2)+'\n')
save('preparing');files={'vit-stack-probe':P/'vit-stack-probe','board_stack.py':P/'board_stack.py'}
for x in compiled['cases']:
 n='layer%02d.nb'%x['layer'];f=P/'int16-nbg'/n;assert hashlib.sha256(f.read_bytes()).hexdigest()==x['nb_sha256'];files['graphs/'+n]=f
for i in [1,2]:
 f=P/('input%d.f32'%i);np.load(P/('s224/l00-input%d.npy'%i)).astype('float32').tofile(f);files[f.name]=f
cfg={'sha256':{n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()},'scope':'resident24 ViT blocks, CPU-precomputed stem; no projector/language/audio, not end-to-end VLM'}
(P/'board-config.json').write_text(json.dumps(cfg,indent=2)+'\n');files['config.json']=P/'board-config.json'
with b.hostlock():
 locked('mkdir '+remote+' && mkdir '+remote+'/graphs',check=True)
 for i,(name,f) in enumerate(files.items()):
  save('uploading',file=name,index=i,total=len(files));upload(f,remote+'/'+name)
 locked('chmod +x '+remote+'/vit-stack-probe',check=True)
 save('running');z=locked('cd '+remote+' && python3 board_stack.py > controller.log 2>&1',capture_output=True,text=True,timeout=420)
 save('collecting',returncode=z.returncode);data=locked('cd '+remote+' && tar czf - results controller.log',capture_output=True,timeout=120).stdout
 (P/'board-results.tar.gz').write_bytes(data)
with tarfile.open(P/'board-results.tar.gz') as tar:tar.extractall(P,filter='data')
save('complete' if z.returncode==0 else 'failed',returncode=z.returncode);print(state.read_text());print((P/'controller.log').read_text()[-1000:])
