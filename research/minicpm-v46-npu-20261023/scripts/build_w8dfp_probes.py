from pathlib import Path
import subprocess,concurrent.futures,time
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');compiler=r.parents[1]/'npu/kv-cache/compile_weight_int8.py'
def run(item):
 n,count=item;src=b/('w8scaled-all' if n=='l0' else 'w8cal-all')/n
 for _ in range(180):
  if (src/'conversion.json').exists():break
  time.sleep(1)
 cmd=['/opt/opi-lab/export-venv/bin/python',str(compiler),'--base',str(src),'--results-root',str(b),'--output',str(b/'w8dfp'/n),'--expected-linears',str(count),'--weight-mode','dfp','--dfp-method','mse','--timeout','180']
 with (r/'results'/('w8dfp-'+n+'.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print(n,z.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(run,[('l0',8),('l3',7),('head0',1)]))
