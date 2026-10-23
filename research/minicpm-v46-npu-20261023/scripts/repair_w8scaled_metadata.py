from pathlib import Path
import subprocess,concurrent.futures,json,time
r=Path(__file__).resolve().parents[1];rows=[]
def run(n):
 with (r/'results'/('w8scaled-repair-'+n+'.log')).open('w') as f:z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_w8_scaled.py'),n,'all'],stdout=f,stderr=subprocess.STDOUT)
 return {'name':n,'exit':z.returncode}
with concurrent.futures.ThreadPoolExecutor(2) as ex:
 for row in ex.map(run,['l'+str(i) for i in range(24) if i%4!=3 and i!=0]):
  rows.append(row);(r/'results/w8scaled-repair-status.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
