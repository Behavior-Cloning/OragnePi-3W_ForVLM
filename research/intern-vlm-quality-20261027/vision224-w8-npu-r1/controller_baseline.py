from pathlib import Path
import sys,json,tarfile,hashlib
P=Path(__file__).resolve().parent;R=P.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
D=P.parent/'vit-full-r1';out=P/'baseline-after';out.mkdir(exist_ok=False);remote='/tmp/opi-vision224-w8-baseline-r1'
files={'tower-probe':D/'tower-fast16-probe','board_tower.py':P/'board_tower.py','input1.f32':D/'endpoint-assets/stem1.f32','input2.f32':D/'endpoint-assets/stem2.f32'}
hashes={n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()}
old=json.loads((D/'tower-r2/config.json').read_text())['sha256']
hashes.update({n:h for n,h in old.items() if n.startswith('graphs/')})
(out/'config.json').write_text(json.dumps({'sha256':hashes}));files['config.json']=out/'config.json'
with b.hostlock():
 locked('test ! -e '+remote+' && mkdir '+remote+' && ln -s /tmp/opi-vit-tower-r2/graphs '+remote+'/graphs',check=True)
 for n,f in files.items():upload(f,remote+'/'+n)
 locked('chmod +x '+remote+'/tower-probe',check=True)
 z=locked('cd '+remote+' && python3 board_tower.py > controller.log 2>&1',capture_output=True,text=True,timeout=420)
 data=locked('cd '+remote+' && tar czf - results controller.log',capture_output=True,check=True,timeout=90).stdout;(out/'raw.tar.gz').write_bytes(data)
 with tarfile.open(out/'raw.tar.gz') as t:
  assert all(not m.issym() and not m.islnk() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(out)
 assert z.returncode==0;print('baseline complete')
