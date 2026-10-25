from pathlib import Path
import json,subprocess,time,traceback
T=Path(__file__).resolve().parent;S=T/'finish-down-row-scale-status.json';assert not S.exists()
s={'state':'waiting','controller_pid':46617,'started':time.time()}
def save():S.write_text(json.dumps(s,indent=2)+'\n')
save()
try:
 while True:
  f=T/'down-row-scale-original-input-replay-r1-controller.json'
  if f.exists():
   d=json.loads(f.read_text())
   if d['status'] in ['complete','failed']:
    assert d['status']=='complete' and d['returncode']==0,d
    break
  live=Path('/proc/46617/cmdline');assert live.exists(),'controller vanished without completed evidence'
  assert 'controller_down_row_scale_replay.py' in live.read_text(),'PID identity changed'
  time.sleep(10)
 s['state']='analyzing';save()
 with (T/'down-row-scale-replay-analysis.log').open('w') as log:subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(T/'runtime/profile-analysis/analyze_down_row_scale_replay.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
 s.update(state='complete',ended=time.time());save()
except BaseException:s.update(state='failed',error=traceback.format_exc(),ended=time.time());save();raise
