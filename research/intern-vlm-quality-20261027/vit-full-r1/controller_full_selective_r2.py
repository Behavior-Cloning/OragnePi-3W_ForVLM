from pathlib import Path
import sys,tarfile,json,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[2]/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked
remote='/tmp/opi-vit-full-selective-r2';old='/tmp/opi-vit-full-selective-r1';assert not (P/'selective-stack-r2').exists()
with b.hostlock():
 locked('test ! -e '+remote+'/results && cp '+' '.join(old+'/'+n for n in ['vit-stack-probe','board_stack.py','config.json','input1.f32','input2.f32'])+' '+remote+'/',check=True)
 z=locked('cd '+remote+' && python3 board_stack.py > controller.log 2>&1',capture_output=True,text=True,timeout=420)
 data=locked('cd '+remote+' && tar czf - results controller.log',capture_output=True,check=True,timeout=120).stdout
(P/'selective-stack-r2.tar.gz').write_bytes(data)
with tarfile.open(P/'selective-stack-r2.tar.gz') as t:t.extractall(P/'selective-stack-r2',filter='data')
(P/'selective-stack-r2-status.json').write_text(json.dumps({'returncode':z.returncode,'remote':remote})+'\n');print('returncode',z.returncode);print((P/'selective-stack-r2/controller.log').read_text()[-1000:]);assert z.returncode==0
