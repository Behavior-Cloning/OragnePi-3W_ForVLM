from pathlib import Path
import json,os,shutil,time
out=Path('/opt/opi-vlm/minicpm-v46-npu-20261023');before=shutil.disk_usage('/');rows=[]
roots=[Path('/root/.cache/pip/http'),Path('/root/.cache/pip/http-v2')]
files=[]
for root in roots:
 if root.exists():files.extend(p for p in root.rglob('*') if p.is_file() and not p.is_symlink())
files.extend(p for p in [Path('/var/cache/apt/pkgcache.bin'),Path('/var/cache/apt/srcpkgcache.bin')] if p.is_file())
files.extend(Path('/var/cache/apt/archives').glob('*.deb'))
for p in files:
 st=p.stat();rows.append({'path':str(p),'bytes':st.st_size,'allocated':st.st_blocks*512});p.unlink()
after=shutil.disk_usage('/');j={'timestamp':time.time(),'scope':'pip HTTP cache and apt download/metadata cache only; model artifacts preserved','files':rows,'logical_bytes':sum(x['bytes'] for x in rows),'free_before':before.free,'free_after':after.free};(out/'cleanup.json').write_text(json.dumps(j,indent=2));print(json.dumps({k:v for k,v in j.items() if k!='files'}),flush=True)
