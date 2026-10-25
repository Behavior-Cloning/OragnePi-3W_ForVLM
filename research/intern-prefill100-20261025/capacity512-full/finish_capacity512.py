"""Observe existing WSL controller identity then analyze recovered evidence. No board calls."""
from pathlib import Path
import json,os,sys,subprocess,time,traceback,hashlib
P=Path(__file__).resolve().parent
run='c512-m16-m32-aba-r1';launch=json.loads((P/'launch.json').read_text());pid=launch['pid']
status=P/'finish-status.json';state={'status':'waiting_existing_controller','run':run,'controller_pid':pid,'started':time.time()}
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(status)
def identity(pid):
 p=Path('/proc')/str(pid)/'stat'
 try:
  fields=p.read_text().rsplit(')',1)[1].split();return {'start_ticks':fields[19],'state':fields[0]}
 except FileNotFoundError:return None
initial=identity(pid);state['initial_identity']=initial;save()
try:
 deadline=time.monotonic()+6000
 while True:
  ctl=json.loads((P/(run+'-controller.json')).read_text());live=identity(pid)
  same=live is not None and initial is not None and live['start_ticks']==initial['start_ticks'] and live['state']!='Z'
  if ctl['status']=='complete':
   assert ctl.get('returncode')==0 and ctl['meta']['status']=='complete';break
  if ctl['status'] in ['failed','observation_failed']:
   state.update(status='controller_needs_review',controller_status=ctl['status'],controller_still_live=same,error=ctl.get('error'));save();sys.exit(1)
  if not same:raise RuntimeError('controller identity absent/changed without terminal completion; inspect artifacts, do not restart automatically')
  if time.monotonic()>deadline:raise TimeoutError('watcher observation budget exceeded; controller may remain live')
  time.sleep(15)
 state.update(status='analyzing',controller_status=ctl['status']);save()
 audit=json.loads((P/'INDEPENDENT_SOURCE_AUDIT.json').read_text())
 for n in ['analyze_fulli16.py','contract.py']:
  assert hashlib.sha256((P/n).read_bytes()).hexdigest()==audit['source_sha256'][n],n
 with (P/'analyze.log').open('w') as f:
  r=subprocess.run([sys.executable,str(P/'analyze_fulli16.py'),run],cwd=P,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 assert r.returncode==0,'analyzer failed; inspect analyze.log and source without altering original evidence'
 a=json.loads((P/'results'/run/'analysis.json').read_text())
 state.update(status='analysis_ready_manual_quality_pending',ended=time.time(),measured_grid_all_below100ms=a['measured_grid_all_below100ms'],whole_objective_verified=False,report=str(P/'results'/run/'REPORT.md'));save()
except BaseException:
 if state['status']!='controller_needs_review':state.update(status='needs_review',error=traceback.format_exc(),ended=time.time());save()
 raise
