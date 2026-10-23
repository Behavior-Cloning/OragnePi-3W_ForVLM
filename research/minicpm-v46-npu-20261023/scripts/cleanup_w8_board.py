from pathlib import Path
import json,os,hashlib,collections,shutil
r=Path('/opt/opi-vlm/minicpm-v46-npu-20261023');expected=json.loads((r/'baseline-retained-wsl-hashes.json').read_text())
for name,sha in expected.items():
 p=r/'baseline-fp16'/name
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,str(p)
paths=list((r/'baseline-fp16').glob('*.nb'))+list((r/'w8cal-all').glob('*'))+list(r.glob('w8*.nb'))+list(r.glob('w8*-input.f32'))+list(r.glob('w8*-output.f32'));paths=[p for p in paths if p.is_file() and not p.is_symlink()];records=[];groups={};before=shutil.disk_usage(r)
for p in paths:
 st=p.stat();key=(st.st_dev,st.st_ino);g=groups.setdefault(key,{'links':st.st_nlink,'count':0,'blocks':st.st_blocks*512});g['count']+=1
 records.append({'path':str(p),'bytes':st.st_size,'nlink':st.st_nlink,'inode':st.st_ino,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
x={'scope':'obsolete FP16 NBGs verified retained in WSL, superseded unscaled W8, copied single-layer probe files; current comparison models preserved','files':records,'before_free':before.free,'fully_unlinked_allocated_bytes':sum(g['blocks'] for g in groups.values() if g['count']==g['links'])};out=r/'w8-cleanup-board.json';out.write_text(json.dumps(x,indent=2))
for p in paths:p.unlink()
x['after_free']=shutil.disk_usage(r).free;x['completed']=True;out.write_text(json.dumps(x,indent=2));print({k:v for k,v in x.items() if k!='files'})
