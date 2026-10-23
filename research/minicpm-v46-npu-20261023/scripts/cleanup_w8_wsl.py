from pathlib import Path
import json,os,hashlib,collections,time,shutil
r=Path(__file__).resolve().parents[1];b=Path('/opt/opi-lab/minicpm-v46-npu-20261023');records=[];groups={};before=shutil.disk_usage(b)
for p in b.rglob('*'):
 if p.is_symlink() or not p.is_file() or 'ref311' in p.relative_to(b).parts:continue
 if not (p.suffix in ['.onnx','.o'] or p.name in ['decoder.data','gen_nbg'] or p.name.endswith('_nbg_unify.data')):continue
 st=p.stat();k=(st.st_dev,st.st_ino);d=groups.setdefault(k,{'links':st.st_nlink,'paths':[],'blocks':st.st_blocks*512,'size':st.st_size});d['paths'].append(str(p))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 records.append({'path':str(p),'bytes':st.st_size,'inode':st.st_ino,'nlink':st.st_nlink,'sha256':h.hexdigest()})
manifest={'scope':'this side experiment only; regenerable ONNX, intermediate weight data and objects; NBGs, original weights and references retained','before_free':before.free,'files':records,'apparent_bytes':sum(x['bytes'] for x in records),'unique_bytes':sum(x['size'] for x in groups.values()),'fully_unlinked_allocated_bytes':sum(x['blocks'] for x in groups.values() if len(x['paths'])==x['links'])}
p=r/'results/w8-cleanup-wsl.json';p.write_text(json.dumps(manifest,indent=2))
for x in records:Path(x['path']).unlink()
manifest['after_free']=shutil.disk_usage(b).free;manifest['completed']=True;p.write_text(json.dumps(manifest,indent=2));print({k:v for k,v in manifest.items() if k!='files'})
