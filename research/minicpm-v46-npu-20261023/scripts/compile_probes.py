import subprocess,json,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');results=[]
for k in ['reduce-b1','matmul-b1']:
 cmd=['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_probe.py'),'--source',str(b/k),'--results-root',str(b),'--output',str(b/(k+'-fp16')),'--dtype','fp16','--timeout','180']
 with (r/'results'/('compile-'+k+'.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 results.append({'kind':k,'exit':z.returncode});(r/'results/compile-status.json').write_text(json.dumps(results,indent=2));print(results[-1],flush=True)
