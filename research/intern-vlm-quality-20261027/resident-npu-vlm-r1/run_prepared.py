from pathlib import Path
import sys,subprocess,json,time,tarfile
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[2]/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked
state=P/'controller-status.json';assert not state.exists(),'existing attempt; inspect rather than restart'
remote='/tmp/opi-resident-npu-vlm-r1'
with b.hostlock():
 z=locked('cd '+remote+' && python3 preflight.py',capture_output=True,text=True,check=True,timeout=30)
 gate=json.loads(z.stdout);(P/'latest-preflight.json').write_text(z.stdout)
 if not gate['eligible']:print(z.stdout);raise SystemExit('No models loaded; preflight blocked')
 state.write_text(json.dumps({'status':'running','at':time.time()})+'\n')
 try:z=locked('cd '+remote+' && python3 board_pipeline.py > controller.log 2>&1',capture_output=True,text=True,timeout=600)
 except subprocess.TimeoutExpired:
  state.write_text(json.dumps({'status':'controller_timeout_remote_unknown','at':time.time()})+'\n');raise
 data=locked('cd '+remote+' && tar czf - results controller.log',capture_output=True,timeout=120,check=True).stdout
(P/'raw.tar.gz').write_bytes(data)
with tarfile.open(P/'raw.tar.gz') as t:t.extractall(P,filter='data')
state.write_text(json.dumps({'status':'complete' if z.returncode==0 else 'failed','returncode':z.returncode})+'\n')
print(state.read_text())
