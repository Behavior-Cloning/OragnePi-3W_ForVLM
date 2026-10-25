from pathlib import Path
import json,time,subprocess,traceback,os
P=Path(__file__).resolve().parent;run='down-row-scale-full-r1';state=P/'finish-status.json'
def save(s):
 t=state.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2));t.replace(state)
save({'status':'waiting','started':time.time()})
try:
 while True:
  c=json.loads((P/(run+'-controller.json')).read_text());pid=json.loads((P/'launch.json').read_text())['pid']
  try:live='/down-row-scale-full/controller.py' in Path('/proc/'+str(pid)+'/cmdline').read_text()
  except FileNotFoundError:live=False
  if c['status'] in ['complete','failed'] and not live:break
  if not live:raise RuntimeError('Controller vanished without terminal state; do not restart benchmark')
  time.sleep(15)
 assert c['status']=='complete' and c['returncode']==0,c
 save({'status':'analyzing','at':time.time()})
 with (P/'analysis.log').open('w') as f:z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(P/'analyze.py'),run],stdout=f,stderr=subprocess.STDOUT,timeout=180)
 assert z.returncode==0, 'Analyze failed; inspect analysis.log; benchmark will not restart'
 save({'status':'analysis_complete_pending_manual_audit','at':time.time(),'report':str(P/'results'/run/'REPORT.md')})
except BaseException:
 save({'status':'failed','at':time.time(),'error':traceback.format_exc()});raise
