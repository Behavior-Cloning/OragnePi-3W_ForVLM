from pathlib import Path
import json,subprocess,time,os,traceback
T=Path(__file__).resolve().parent;P=T/'prefixsafe-i16';status=T/'continue-prefixsafe-a8-status.json';py='/opt/opi-lab/export-venv/bin/python'
assert not status.exists(),'single continuation only'
s={'state':'waiting_prefixsafe','wait_pid':43926,'started':time.time()}
def save():status.write_text(json.dumps(s,indent=2)+'\n')
save()
try:
 while True:
  c=json.loads((P/'controller.json').read_text())
  live=Path('/proc/43926/cmdline')
  if c['status'] in ['complete','failed']:
   assert c['status']=='complete' and c['returncode']==0,c
   if not live.exists():break
  else:assert live.exists(),'controller missing without terminal evidence'
  time.sleep(10)
 s['state']='analyzing_prefixsafe';save()
 with (P/'analyze.log').open('w') as log:subprocess.run([py,str(P/'analyze.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
 s['state']='running_a8';save()
 with (T/'compact-gateup-a8-controller.log').open('w') as log:subprocess.run([py,str(T/'controller_compact_gateup_a8.py'),'--execute'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1000)
 s['state']='analyzing_a8';save()
 with (T/'compact-gateup-a8-analysis.log').open('w') as log:subprocess.run([py,str(T/'runtime/profile-analysis/analyze_compact_gateup_a8.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 s.update(state='complete',ended=time.time());save()
except BaseException:s.update(state='failed',error=traceback.format_exc(),ended=time.time());save();raise
