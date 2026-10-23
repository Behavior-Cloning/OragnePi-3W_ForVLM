from pathlib import Path
import hashlib,json,time,shutil,subprocess
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');assert str(b)=='/opt/opi-lab/minicpm-v46-npu-20261023';before={'wsl':shutil.disk_usage('/').free,'f':shutil.disk_usage('/mnt/f').free};rows=[]
for p in b.rglob('*'):
 if p.is_symlink() or not p.is_file():continue
 remove=p.suffix=='.onnx' or p.name=='decoder.data' or p.name.endswith('.export.data') or p.suffix=='.o' or p.name=='gen_nbg' or (p.name=='network_binary.nb' and p.is_relative_to(b/'compiled'))
 if not remove:continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()});p.unlink()
# Keep the corrected embeddings; remove the obsolete baseline duplicate after exact hash comparison.
a=b/'full/embeddings.f16';c=b/'full-epssafe/embeddings.f16'
if a.exists() and c.exists():
 def sha(p):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  return h.hexdigest()
 digest=sha(a);assert digest==sha(c);rows.append({'path':str(a),'bytes':a.stat().st_size,'sha256':digest});a.unlink()
after={'wsl':shutil.disk_usage('/').free,'f':shutil.disk_usage('/mnt/f').free};result={'time':time.time(),'files':rows,'bytes':sum(x['bytes'] for x in rows),'before':before,'after_delete':after,'preserved':['original model.safetensors on F','compiled-epssafe NBGs','source scripts/config/quantization and logs','reference fixtures and benchmark output']};(r/'results/final-local-cleanup.json').write_text(json.dumps(result,indent=2));print('DELETED_BYTES',result['bytes'],flush=True)
try:
 z=subprocess.run(['fstrim','-v','/'],capture_output=True,text=True,timeout=90);result['trim']={'exit':z.returncode,'stdout':z.stdout,'stderr':z.stderr}
except Exception as e:result['trim']={'error':repr(e)}
result['after_trim']={'wsl':shutil.disk_usage('/').free,'f':shutil.disk_usage('/mnt/f').free};(r/'results/final-local-cleanup.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='files'},flush=True)
