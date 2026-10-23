from pathlib import Path
import subprocess,json,time
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');rows=[]
compiler=r.parents[1]/'npu/kv-cache/compile_weight_int8.py'
for name,count in [('head0',1),('l0',8),('l3',7)]:
 p=b/'w8retry-i16'/name/'decoder.quantize'
 for _ in range(360):
  if p.exists() and (b/'w8retry-i16'/name/'conversion.json').exists():
   rec=json.loads((b/'w8retry-i16'/name/'conversion.json').read_text())
   if any(z['stage']=='quantize' and z['exit']==0 for z in rec['stages']):break
  time.sleep(2)
 else:raise RuntimeError('calibration missing '+name)
 for mode in ['perchannel','dfp']:
  out=b/('w8retry-'+mode)/name
  cmd=['/opt/opi-lab/export-venv/bin/python',str(compiler),'--base',str(p.parent),'--results-root',str(b),'--output',str(out),'--expected-linears',str(count),'--float-nonlinear','--weight-mode',mode,'--timeout','180']
  with (r/'results'/('w8retry-'+mode+'-'+name+'.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  row={'name':name,'mode':mode,'exit':z.returncode};rows.append(row);(r/'results/w8retry-compile-status.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
print('COMPLETE',flush=True)
