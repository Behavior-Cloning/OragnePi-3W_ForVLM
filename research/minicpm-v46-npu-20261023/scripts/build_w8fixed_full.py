from pathlib import Path
import subprocess,concurrent.futures,json
r=Path(__file__).resolve().parents[1];rows=[]
def run(n):
 with (r/'results'/('w8fixed-full-'+n+'.log')).open('w') as f:z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_w8fixed.py'),n],stdout=f,stderr=subprocess.STDOUT,timeout=220)
 return {'name':n,'exit':z.returncode}
with concurrent.futures.ThreadPoolExecutor(2) as ex:
 for row in ex.map(run,['l'+str(i) for i in range(24) if i%4!=3 and i!=0]):
  rows.append(row);(r/'results/w8fixed-full-status.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
assert all(x['exit']==0 for x in rows)
