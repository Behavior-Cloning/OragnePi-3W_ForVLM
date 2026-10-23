"""Wait for already running build/stager, then execute exactly one full board run."""
from pathlib import Path
import json,subprocess,time,sys
R=Path(__file__).resolve().parent;B=Path('/opt/opi-lab/llama32-full-npu-20261023/q4-source');end=time.monotonic()+3600;state=R/'finish-q4-status.json'
def update(**x):state.write_text(json.dumps(dict(at=time.time(),**x),indent=2))
update(state='waiting-existing-build-and-stager',build_pid=748,stage_pid=7854)
while time.monotonic()<end:
 status=json.loads((R/'latest-controller-q4-source.json').read_text())
 if status['state']=='error':raise RuntimeError('Existing stager failed: '+str(status))
 if (B/'attn-fp16-mlp-w8/manifest.json').exists() and status['state']=='staged-only':break
 time.sleep(5)
else:raise TimeoutError('Existing build/stager did not finish within one hour')
(R/'board_run_pre_guard.py').write_bytes((R/'board_run.py').read_bytes())
(R/'board_run.py').write_bytes((R/'board_run_guarded.py').read_bytes())
update(state='starting-full-controller',guard_revision='tested-three-faults')
p=subprocess.Popen([sys.executable,'-u',str(R/'deploy_run.py'),'--execute']);update(state='controller-running',pid=p.pid);rc=p.wait();update(state='controller-returned',returncode=rc)
if rc:raise SystemExit(rc)
s=json.loads((R/'latest-controller-q4-source.json').read_text())
with (R/'summarize-q4.log').open('w') as log:rc=subprocess.run([sys.executable,str(R/'summarize_board.py'),s['result']],stdout=log,stderr=subprocess.STDOUT).returncode
update(state='finished',summary_returncode=rc,result=s['result']);raise SystemExit(rc)
