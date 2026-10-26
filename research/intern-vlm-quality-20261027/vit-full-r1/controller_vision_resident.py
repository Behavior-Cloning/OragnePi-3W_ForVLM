from pathlib import Path
import sys,json,hashlib,tarfile
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[2]/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
remote='/tmp/opi-vision-resident-npu-r1';O=P/'vision-resident-r1';O.mkdir(exist_ok=False)
files={'vision-resident':P/'vision-resident','vision_worker.py':P/'vision_worker.py','board_vision_resident.py':P/'board_vision_resident.py'}
for i in [1,2]:files[f'image{i}.jpg']=P.parent/f'fixtures/image{i}.jpg'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assets={remote+'/'+n:sha(f) for n,f in files.items()}
for n,h in json.loads((P/'tower-r2/config.json').read_text())['sha256'].items():
 if n.startswith('graphs/'):assets['/tmp/opi-vit-tower-r2/'+n]=h
cfg={'assets':assets,'pixels':{str(i):sha(P/f'endpoint-assets/stem{i}.f32') for i in [1,2]},'vectors':{str(i):sha(P/f'tower-r2/results/run{i-1}-output.f32') for i in [1,2]}}
(O/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');files['config.json']=O/'config.json'
with b.hostlock():
 locked('mkdir '+remote,check=True)
 for n,f in files.items():upload(f,remote+'/'+n)
 locked('chmod +x '+remote+'/vision-resident',check=True)
 z=locked('cd '+remote+' && python3 board_vision_resident.py > controller.log 2>&1',capture_output=True,text=True,timeout=420)
 data=locked('cd '+remote+' && tar czf - results controller.log',capture_output=True,check=True,timeout=90).stdout
(O/'raw.tar.gz').write_bytes(data)
with tarfile.open(O/'raw.tar.gz') as t:t.extractall(O,filter='data')
print('returncode',z.returncode);print((O/'controller.log').read_text()[-1600:]);assert z.returncode==0
