from pathlib import Path
import subprocess
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023')
z=subprocess.run([str(b/'ref311/bin/python'),str(r/'scripts/export_w8element.py'),'l0']);assert z.returncode==0
z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_probe.py'),'--source',str(b/'w8element-source/l0'),'--results-root',str(b),'--output',str(b/'w8element-fp16/l0'),'--dtype','fp16','--timeout','180']);print('EXIT',z.returncode,flush=True)

assert z.returncode==0
z=subprocess.run(["/opt/opi-lab/export-venv/bin/python",str(r/"scripts/compile_w8element.py"),"l0","all"]);raise SystemExit(z.returncode)
