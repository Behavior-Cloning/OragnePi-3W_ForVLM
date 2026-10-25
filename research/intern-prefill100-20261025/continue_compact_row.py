from pathlib import Path
import json,time,subprocess,hashlib,traceback
P=Path(__file__).resolve().parent
H={'controller_compact_row_down.py': '14c95354546d093c09aaa8728495864471dea186800a736fa41863092336c07d', 'runtime/profile-analysis/analyze_compact_row_down.py': '90f9eb27f891e9cbf5fa6149e6f0e593b59fd87db53a6040fff8212aa1a8efcf', 'graphs/compact-row-down-screen/INDEPENDENT_STATIC_AUDIT.json': 'd11195ddc894eae043462302803cd2b5f583a89e523a38df6e80d6d74d8596b2'}
status=P/'continue-compact-row-status.json'
def save(s):
 t=status.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2));t.replace(status)
save({'status':'waiting_tail','started':time.time()})
try:
 while True:
  c=json.loads((P/'down-row-scale-tail/down-row-scale-tail-r1-controller.json').read_text());f=json.loads((P/'down-row-scale-tail/finish-status.json').read_text())
  live=Path('/proc/51776/cmdline')
  active=live.exists() and 'down-row-scale-tail/controller.py' in live.read_text()
  if c['status']=='failed' or f['status']=='failed':raise RuntimeError('Tail controller or analysis failed; no automatic restart')
  if not active and c['status']=='complete' and f['status']=='analysis_complete_pending_manual_audit':break
  if not active and c['status'] not in ['complete','failed']:raise RuntimeError('Tail controller vanished without terminal state')
  time.sleep(15)
 assert c['returncode']==0
 a=json.loads((P/'down-row-scale-tail/results/down-row-scale-tail-r1/analysis.json').read_text());assert a['four_driver_releases_zero'] and a['archive_verified']
 for n,h in H.items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
 assert not (P/'compact-row-down-r1-controller.json').exists(),'No duplicate run'
 save({'status':'running_compact','started':time.time(),'scope':'Independent two-layer screen; does not promote previous model or imply semantic pass'})
 with (P/'compact-row-controller.log').open('w') as log:
  z=subprocess.run(['/opt/opi-lab/export-venv/bin/python','-u',str(P/'controller_compact_row_down.py'),'--execute'],cwd=P,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
 assert z.returncode==0
 c=json.loads((P/'compact-row-down-r1-controller.json').read_text());assert c['status']=='complete' and c['returncode']==0
 with (P/'compact-row-analysis.log').open('w') as log:
  z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(P/'runtime/profile-analysis/analyze_compact_row_down.py')],cwd=P,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 assert z.returncode==0
 save({'status':'complete_pending_manual_audit','ended':time.time()})
except BaseException:
 save({'status':'failed','error':traceback.format_exc(),'at':time.time()});raise
