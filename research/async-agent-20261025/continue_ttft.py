from pathlib import Path
import time,json,subprocess
p=Path(__file__).resolve().parent
while not (p/'pair32-status.json').exists():
 for name in ['pair300-v2-controller.json','pair300-32-controller.json']:
  f=p/name
  if f.exists() and json.loads(f.read_text()).get('status')=='failed':raise RuntimeError(f.read_text())
 time.sleep(10)
assert json.loads((p/'pair32-status.json').read_text())['status']=='complete'
run='ttft-32-64-matrix'
assert not (p/(run+'-controller.json')).exists(),'existing matrix; inspect'
subprocess.run(['python3','-u',str(p/'controller_ttft.py'),run,'900','matrix'],cwd=p,check=True)
