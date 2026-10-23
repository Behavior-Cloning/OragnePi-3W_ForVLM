from pathlib import Path
import time,json,subprocess
r=Path(__file__).resolve().parents[1];p=Path('/opt/opi-lab/minicpm-v46-npu-20261023/w8matint-f16/l0/conversion.json')
for _ in range(180):
 if p.exists() and json.loads(p.read_text())['exit']==0:break
 time.sleep(1)
else:raise RuntimeError('compile failed or timed out')
while Path('/proc/60386').exists():time.sleep(1)
z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/run_w8cal_probes.py'),'w8matint-f16']);raise SystemExit(z.returncode)
