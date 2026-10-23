from pathlib import Path
import json,subprocess,time,shutil,hashlib
r=Path(__file__).resolve().parent;out=Path('/opt/opi-lab/llama32-compat-20261023');python='/opt/opi-lab/export-venv/bin/python'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for _ in range(60):
 if (r/'export-summary.json').exists():break
 time.sleep(2)
else:raise RuntimeError('export missing; inspect export.log')
assert shutil.disk_usage('/mnt/f').free>8*1024**3
for phase,cmd in [('calibrate',[python,str(r/'import_calibrate.py'),'--source',str(out),'--output',str(out/'compiled'),'--dtype','int16','--timeout','300']),('w8',[python,str(r/'compile_w8.py'),'--base',str(out/'compiled'),'--output',str(out/'compiled'),'--float-sensitive','--float-residual','--timeout','300'])]:
 print(phase,flush=True)
 z=subprocess.run(cmd,timeout=960)
 if z.returncode:raise SystemExit(z.returncode)
 if phase=='calibrate':
  # ONNX is reproducible and hash recorded; delete only after successful import.
  p=out/'layer0.onnx';record=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason='Imported graph retained; bounded artifact budget')
  p.unlink();(r/'onnx-cleanup.json').write_text(json.dumps(record,indent=2))
 print('size',sum(x.stat().st_size for x in out.rglob('*') if x.is_file()),flush=True)
print('complete',flush=True)
