#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess,time,traceback
R=Path(__file__).resolve().parent;REPO=R.parents[1]
PY='/opt/opi-lab/export-venv/bin/python'
manifest=json.loads((R/'graphs/deployment-manifest.json').read_text())
assert len(manifest['cases'])==10 and not manifest['compile_failures']
s={'status':'running','started':time.time(),'cases':[]}
def save():(R/'fixture-status.json').write_text(json.dumps(s,indent=2)+'\n')
save()
try:
 for case in manifest['cases']:
  name=case['id'];folder=R/'graphs/cases'/name;out=R/'fixtures'/name;out.parent.mkdir(exist_ok=True)
  assert hashlib.sha256(Path(case['nb']).read_bytes()).hexdigest()==case['nb_sha256']
  cmd=[PY,str(REPO/'npu/kv-cache/resized_fixture.py'),'--model','/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf','--source','/opt/opi-lab/source-validation-20261017/gguf-fresh/export','--folder',str(folder),'--prompt-folder',str(R/'frozen-probe-prompt'),'--output',str(out),'--history','128']
  start=time.time()
  with (R/'fixtures'/(name+'.log')).open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
  row={'id':name,'returncode':rc,'seconds':time.time()-start,'command':cmd};s['cases'].append(row);save()
  if rc:raise RuntimeError(name+' fixture failed')
  info=json.loads((out/'probe-fixture.json').read_text())
  assert info['history']==128 and info['query_tokens']==case['query_tokens'] and info['capacity']==512
  row.update(input_sha256=hashlib.sha256((out/'probe-input.f32').read_bytes()).hexdigest(),reference_sha256=hashlib.sha256((out/'probe-reference.f32').read_bytes()).hexdigest());save()
 s.update(status='complete',ended=time.time());save()
except BaseException:
 s.update(status='failed',ended=time.time(),error=traceback.format_exc());save();raise
