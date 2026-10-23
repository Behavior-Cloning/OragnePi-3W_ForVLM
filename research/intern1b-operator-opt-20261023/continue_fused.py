from pathlib import Path
import json,time,subprocess
r=Path(__file__).resolve().parent
for _ in range(180):
 s=json.loads((r/'status.json').read_text())
 if s['phase']=='returned':break
 if s['phase']=='failed':raise RuntimeError(s)
 time.sleep(10)
else:raise TimeoutError('main matrix did not return')
if s.get('exit')!=0:raise RuntimeError(s)
p=subprocess.run(['python3','-u',str(r/'fused/controller.py')]);raise SystemExit(p.returncode)
