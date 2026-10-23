from pathlib import Path
import subprocess,time
r=Path(__file__).resolve().parents[1];p=Path('/opt/opi-lab/minicpm-v46-npu-20261023/w8mm-cal/l0/conversion.json')
for _ in range(120):
 if p.exists() and 'CALIBRATED_ONLY' in p.read_text():break
 time.sleep(1)
else:raise RuntimeError('missing quantization')
z=subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(r/'scripts/compile_w8matint_linear.py'),'l0','f16']);raise SystemExit(z.returncode)
