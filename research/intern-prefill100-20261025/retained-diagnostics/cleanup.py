"""Plan or remove only replicated *-logits.f32 in this experiment root, under both locks."""
from pathlib import Path
import argparse,json,hashlib,sys,shlex,time
P=Path(__file__).resolve().parent
ROOT='/opt/opi-vlm/intern-prefill100-20261025'
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as stream:
  for b in iter(lambda:stream.read(4194304),b''):h.update(b)
 return h.hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--execute-plan',type=Path);a=ap.parse_args()
inv=json.loads((P/'inventory.json').read_text());assert inv['status']=='complete_local_only'
known={}
for f in inv['files']:
 p=Path(f['local_path']);assert p.is_file() and not p.is_symlink() and 'results' in p.parts
 assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
 known[(f['sha256'],f['bytes'])]=f['local_path']
if a.execute_plan:
 plan=json.loads(a.execute_plan.read_text());assert plan['root']==ROOT and plan['status']=='plan_only'
 for f in plan['files']:assert (f['sha256'],f['bytes']) in known
 data={'mode':'delete','files':plan['files'],'root':ROOT}
else:data={'mode':'plan','root':ROOT,'known':[{'sha256':h,'bytes':size,'local_path':path} for (h,size),path in known.items()]}
remote=r"""
from pathlib import Path
import json,sys,hashlib,shutil,time
x=json.load(sys.stdin);root=Path(x['root']);assert str(root)=='/opt/opi-vlm/intern-prefill100-20261025'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def allowed(p):
 return root in p.parents and p.resolve()==p and not p.is_symlink() and p.is_file() and p.name.endswith('-logits.f32') and 'fulli16-suffixfix' not in p.relative_to(root).parts
free_before=shutil.disk_usage(root).free;files=[]
if x['mode']=='plan':
 known={(f['sha256'],f['bytes']):f['local_path'] for f in x['known']}
 for p in sorted(root.rglob('*-logits.f32')):
  if not allowed(p):continue
  size=p.stat().st_size
  if size==0:continue
  digest=sha(p)
  if (digest,size) in known:files.append({'remote_path':str(p),'bytes':size,'sha256':digest,'retained_local_path':known[(digest,size)]})
 status='plan_only'
else:
 assert x['mode']=='delete';assert len({f['remote_path'] for f in x['files']})==len(x['files'])
 # Validate every exact path and retained digest before deleting any file.
 for f in x['files']:
  p=Path(f['remote_path']);assert allowed(p),str(p)
  assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],str(p)
 for f in x['files']:Path(f['remote_path']).unlink();files.append(f)
 status='deleted_retained_duplicates'
print(json.dumps({'status':status,'root':str(root),'files':files,'bytes':sum(f['bytes'] for f in files),'free_before':free_before,'free_after':shutil.disk_usage(root).free,'unix':time.time()}))
"""
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
with b.hostlock():
 command='flock -w 5 /run/opi-npu-kv-benchmark.lock /opt/opi-vlm/jev-lab-20261020/venv/bin/python -c '+shlex.quote(remote)
 z=b.ssh(command,input=json.dumps(data).encode(),capture_output=True,timeout=180,check=True)
 out=json.loads(z.stdout);out['local_inventory_sha256']=sha(P/'inventory.json');out['source_sha256']=sha(Path(__file__))
 name='deleted-'+str(int(time.time()))+'.json' if a.execute_plan else 'plan-'+str(int(time.time()))+'.json'
 (P/name).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'manifest':str(P/name),'status':out['status'],'files':len(out['files']),'bytes':out['bytes'],'free_before':out['free_before'],'free_after':out['free_after']}))
