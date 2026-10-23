from pathlib import Path
import subprocess,concurrent.futures
r=Path(__file__).resolve().parents[1]
def run(pair):
 n,v=pair
 with (r/'results'/('w8cal-'+v+'-'+n+'.log')).open('w') as f:z=subprocess.run(['/opt/opi-lab/export-venv/bin/python','-u',str(r/'scripts/compile_w8cal.py'),n,v],stdout=f,stderr=subprocess.STDOUT)
 print(n,v,z.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(run,[('head0','all'),('l0','all'),('l3','all'),('l0','protect'),('l0','mlponly')]))
