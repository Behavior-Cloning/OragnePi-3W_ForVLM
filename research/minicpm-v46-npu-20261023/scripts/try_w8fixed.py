from pathlib import Path
import subprocess
r=Path(__file__).resolve().parents[1];py='/opt/opi-lab/export-venv/bin/python'
for args in [('compile_w8fixed.py','l0'),('run_w8cal_probes.py','w8fixed')]:
 z=subprocess.run([py,'-u',str(r/'scripts'/args[0]),*args[1:]]);assert z.returncode==0
