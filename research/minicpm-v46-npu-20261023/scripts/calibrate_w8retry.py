from pathlib import Path
import subprocess,json,time
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023')
x=subprocess.run([str(b/'ref311/bin/python'),str(r/'scripts/export_w8retry_subset.py')]);assert x.returncode==0
rows=[]
for name in ['head0','l0','l3']:
 with (r/'results'/('w8retry-calibrate-'+name+'.log')).open('w') as f:
  z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_probe.py'),'--source',str(b/'w8retry-source'/name),'--results-root',str(b),'--output',str(b/'w8retry-i16'/name),'--dtype','int16','--timeout','180'],stdout=f,stderr=subprocess.STDOUT)
 rows.append({'name':name,'exit':z.returncode});(r/'results/w8retry-calibration-status.json').write_text(json.dumps(rows,indent=2));print(rows[-1],flush=True)
