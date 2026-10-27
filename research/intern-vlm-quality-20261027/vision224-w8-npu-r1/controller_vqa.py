from pathlib import Path
import sys,json,tarfile,hashlib
P=Path(__file__).resolve().parent;D=P.parent;R=P.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
base=D/'npu-vision-input-r1';remote='/tmp/opi-vision224-w8-vqa-r1';out=P/'npu-vqa';out.mkdir(exist_ok=False);assert not (out/'raw.tar.gz').exists()
f=P/'board_vqa.py'
files={'board_vqa.py':f,'resident_worker_visual.py':base/'resident_worker_visual.py','resident-visual':base/'runtime/resident-visual','parent-config.json':base/'parent-config.json','vqa-cases.json':D/'fixtures/vqa-cases.json'}
for i in [1,2]:files[f'image{i}.f32']=P/f'board/results/run{i-1}-output.f32'
(out/'upload-sha256.json').write_text(json.dumps({n:hashlib.sha256(f.read_bytes()).hexdigest() for n,f in files.items()},indent=2)+'\n')
with b.hostlock():
 locked('mkdir '+remote,check=True)
 for n,f in files.items():upload(f,remote+'/'+n)
 locked('chmod +x '+remote+'/resident-visual',check=True)
 z=locked('cd '+remote+' && python3 board_vqa.py > controller.log 2>&1',capture_output=True,text=True,timeout=600)
 data=locked('cd '+remote+' && tar czf - vqa-results vqa-frozen-inputs.json controller.log',capture_output=True,timeout=90).stdout
(out/'raw.tar.gz').write_bytes(data)
with tarfile.open(out/'raw.tar.gz') as t:t.extractall(out,filter='data')
print('returncode',z.returncode);print((out/'controller.log').read_text()[-1000:]);assert z.returncode==0
