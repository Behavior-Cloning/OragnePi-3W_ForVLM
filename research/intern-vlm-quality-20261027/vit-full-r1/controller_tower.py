from pathlib import Path
import sys,json,hashlib,tarfile,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[2]/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
remote='/tmp/opi-vit-tower-r1';out=P/'tower-r1';out.mkdir(exist_ok=False)
files={'tower-probe':P/'tower-probe','board_tower.py':P/'board_tower.py','input1.f32':P/'endpoint-assets/stem1.f32','input2.f32':P/'endpoint-assets/stem2.f32'}
hashes={n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()}
for i in range(24):hashes[f'graphs/layer{i:02d}.nb']=hashlib.sha256((P/f'selective-normgelu-full-l{i:02d}.nb').read_bytes()).hexdigest()
hashes['graphs/stem.nb']=hashlib.sha256((P/'endpoints-nbg/stem.nb').read_bytes()).hexdigest()
hashes['graphs/projector.nb']=hashlib.sha256((P/'selective-projector-norm-gelu.nb').read_bytes()).hexdigest()
(out/'config.json').write_text(json.dumps({'sha256':hashes,'scope':'NPU stem +24 blocks+projector; normalized224 NCHW ->64x896; not JPEG preprocessing or language'},indent=2)+'\n');files['config.json']=out/'config.json'
with b.hostlock():
 locked('mkdir '+remote+' && mkdir '+remote+'/graphs',check=True)
 for n,f in files.items():upload(f,remote+'/'+n)
 cmd='; '.join('ln -s /tmp/opi-vit-full-selective-r1/graphs/layer%02d.nb %s/graphs/layer%02d.nb'%(i,remote,i) for i in range(24))
 locked(cmd,check=True)
 locked('ln -s /tmp/opi-vit-endpoints-r1/stem.nb '+remote+'/graphs/stem.nb && ln -s /tmp/opi-vit-projector-selective-r1/projector.nb '+remote+'/graphs/projector.nb && chmod +x '+remote+'/tower-probe',check=True)
 z=locked('cd '+remote+' && python3 board_tower.py > controller.log 2>&1',capture_output=True,text=True,timeout=420)
 data=locked('cd '+remote+' && tar czf - results controller.log',capture_output=True,check=True,timeout=90).stdout
(out/'raw.tar.gz').write_bytes(data)
with tarfile.open(out/'raw.tar.gz') as t:t.extractall(out,filter='data')
print('returncode',z.returncode);print((out/'controller.log').read_text()[-1500:]);assert z.returncode==0
