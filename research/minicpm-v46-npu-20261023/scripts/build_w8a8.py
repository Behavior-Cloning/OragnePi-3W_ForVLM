from pathlib import Path
import subprocess,concurrent.futures
r=Path(__file__).resolve().parents[1]
def run(n):
 with (r/'results'/('w8a8-'+n+'.log')).open('w') as f:z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_w8a8.py'),n],stdout=f,stderr=subprocess.STDOUT)
 print(n,z.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(run,['l0','l3','head0']))
