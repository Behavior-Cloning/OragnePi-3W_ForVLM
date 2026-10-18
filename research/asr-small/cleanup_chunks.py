"""Remove only transfer chunks proven byte-identical to ranges of retained GGUFs."""
import hashlib,json,os,re,shutil,time
from pathlib import Path
root=Path('/opt/opi-vlm/models');rows=[];before=shutil.disk_usage(root)._asdict()
for d in sorted(root.rglob('*.gguf.chunks')):
 original=d.with_name(d.name[:-7])
 if not original.is_file() or d.is_symlink():continue
 # Chunk naming in previous uploader: 8 MiB numbered chunks plus retry .tail.
 with original.open('rb') as f:
  for p in sorted(d.iterdir()):
   match=re.fullmatch(r'(\d{4})(\.tail)?',p.name)
   if not match or not p.is_file() or p.is_symlink():continue
   offset=int(match[1])*8388608;size=p.stat().st_size
   if size>8388608 or offset+size>original.stat().st_size:continue
   data=p.read_bytes();f.seek(offset);ref=f.read(size)
   if data!=ref:
    rows.append({'path':str(p),'status':'retained_mismatch','bytes':size});continue
   rows.append({'path':str(p),'original':str(original),'offset':offset,'bytes':size,'sha256':hashlib.sha256(data).hexdigest(),'status':'deleted_verified_duplicate'})
   p.unlink()
 if not any(d.iterdir()):d.rmdir()
os.sync();after=shutil.disk_usage(root)._asdict()
result={'before':before,'after':after,'freed_bytes':after['free']-before['free'],'removed_file_bytes':sum(r['bytes'] for r in rows if r['status'].startswith('deleted')),'files':rows,'timestamp':time.time()}
Path('/opt/opi-vlm/cleanup-chunks-20261018.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='files'}));print('deleted',sum(r['status'].startswith('deleted') for r in rows),'retained_mismatch',sum(r['status']=='retained_mismatch' for r in rows))
