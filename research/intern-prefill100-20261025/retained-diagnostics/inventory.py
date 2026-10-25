from pathlib import Path
import json,hashlib,time,os
P=Path(__file__).resolve().parent;R=P.parent
started=time.time();paths=set()
for base in [R/'results']+[p/'results' for p in R.iterdir() if p.is_dir()]:
 if base.is_dir():paths.update(p for p in base.rglob('*.f32') if p.is_file() and not p.is_symlink())
records=[]
for p in sorted(paths):
 h=hashlib.sha256();size=p.stat().st_size
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4194304),b''):h.update(chunk)
 assert p.stat().st_size==size
 records.append({'local_path':str(p),'bytes':size,'sha256':h.hexdigest()})
x={'status':'complete_local_only','started':started,'ended':time.time(),'pid':os.getpid(),'scope':'Inventory retained raw diagnostic evidence only; no board access or deletion. Any future board cleanup requires path allowlist plus actual remote SHA/size match and both locks.','files':records,'total_bytes':sum(p['bytes'] for p in records)}
(P/'inventory.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'files':len(records),'bytes':x['total_bytes']}))
