#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess,time,traceback
R=Path(__file__).resolve().parent;REPO=R.parents[1]
SRC=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
MODEL=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf')
PY='/opt/opi-lab/export-venv/bin/python';state={'state':'running','start':time.time(),'cases':[]}
def save():(R/'fixture-status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 for name,variant in [('decode','w8-fused2-c1280-cap1-headmse'),('prefill16','w8-fused2-c1280-t16-cap1-productwide')]:
  out=R/'fixtures'/('cap1280-'+name);out.parent.mkdir(exist_ok=True)
  cmd=[PY,str(REPO/'npu/kv-cache/resized_fixture.py'),'--model',str(MODEL),'--source',str(SRC),'--folder',str(SRC/variant/'l0'),'--prompt-folder',str(R/'frozen-prompt'),'--output',str(out),'--history','128']
  with (R/(name+'-fixture.log')).open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
  state['cases'].append({'name':name,'returncode':rc,'command':cmd});save()
  if rc:raise RuntimeError(name+' fixture failed')
 state.update(state='complete',end=time.time());save()
except Exception as e:state.update(state='failed',error=repr(e),traceback=traceback.format_exc(),end=time.time());save();raise
