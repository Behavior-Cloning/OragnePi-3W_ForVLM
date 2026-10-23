from pathlib import Path
import subprocess,concurrent.futures,json
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');rows=[]
def run(n):
 with (r/'results'/('w8matint-full-'+n+'.log')).open('w') as f:
  z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/calibrate_only.py'),'--source',str(b/'w8retry-source'/n),'--results-root',str(b),'--output',str(b/'w8retry-i16'/n),'--dtype','int16'],stdout=f,stderr=subprocess.STDOUT,timeout=240)
  if z.returncode==0:z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_w8matint.py'),n,'f16'],stdout=f,stderr=subprocess.STDOUT,timeout=200)
 return {'name':n,'exit':z.returncode}
with concurrent.futures.ThreadPoolExecutor(2) as ex:
 for row in ex.map(run,['l7','l11','l15','l19','l23']):
  rows.append(row);(r/'results/w8matint-full-status.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
