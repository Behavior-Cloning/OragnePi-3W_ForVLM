from pathlib import Path
import subprocess,json
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');py=b/'ref311/bin/python'
for stage,cmd in [('export',[str(py),str(r/'scripts/model_layers.py')]),('compile',['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_probe.py'),'--source',str(b/'linear0-epssafe'),'--results-root',str(b),'--output',str(b/'linear0-epssafe-fp16'),'--dtype','fp16','--timeout','240'])]:
 with (r/'results'/('linear0-'+stage+'.log')).open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print(stage,z.returncode,flush=True)
 if z.returncode:break
