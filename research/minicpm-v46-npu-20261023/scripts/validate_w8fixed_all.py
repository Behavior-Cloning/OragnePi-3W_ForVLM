from pathlib import Path
import subprocess,time,json
r=Path(__file__).resolve().parents[1];py='/opt/opi-lab/minicpm-v46-npu-20261023/ref311/bin/python'
for file,args in [('w8fixed-runtime.jsonl',['validate_w8_generation.py','w8fixed']),('w8fixed-holdouts.jsonl',['validate_w8_holdouts.py','w8fixed'])]:
 for _ in range(240):
  p=r/'results'/file
  if p.exists() and any(json.loads(x).get('event')=='closed' for x in p.read_text().splitlines()):break
  time.sleep(2)
 else:raise RuntimeError('missing results '+file)
 z=subprocess.run([py,'-u',str(r/'scripts'/args[0]),*args[1:]]);assert z.returncode==0
print('COMPLETE')
