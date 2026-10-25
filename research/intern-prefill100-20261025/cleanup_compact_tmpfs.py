"""Remove only completed tmpfs probe copies whose bytes are retained in WSL."""
from pathlib import Path
import argparse,json,hashlib,shlex,sys,time,tarfile
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('run');a.add_argument('--execute-plan',type=Path);a=a.parse_args()
assert a.run.replace('-','').replace('_','').isalnum()
ctl=json.loads((R/(a.run+'-controller.json')).read_text());assert ctl['status']=='complete' and ctl['returncode']==0
root='/dev/shm/opi-prefill100-probe-'+a.run;assert ctl['remote']==root
d=R/'results'/a.run;cfg=json.loads((d/'config.json').read_text());assert cfg['staging_kind']=='tmpfs'
assert json.loads((d/'meta.json').read_text())['status']=='complete'
assert a.run=='compact-past128-m256-abcd-a-r1'
source={'original.nb':R/'graphs/cases/c512-m256-l0/export_nbg_unify/network_binary.nb','row224-c512.nb':R/'graphs/candidate-softmax-row224-m256/export_nbg_unify/network_binary.nb','plain-c128.nb':R/'graphs/compact-past128-m256/plain/export_nbg_unify/network_binary.nb','row224-c128.nb':R/'graphs/compact-past128-m256/row224/export_nbg_unify/network_binary.nb','input512.f32':R/'fixtures-canonical/c512-m256-l0/probe-input.f32','input128.f32':R/'fixtures-compact/c128-m256-l0/probe-input.f32','probe':R/'runtime/compact-probe/tensor-probe-compact','board_probe_compact.py':R/'board_probe_compact.py','config.json':R/(a.run+'-config.json')}
assert set(source)==set(ctl['source_sha256'])
known={}
for name,path in source.items():
 digest=sha(path);assert digest==ctl['source_sha256'][name];known[name]={'sha256':digest,'bytes':path.stat().st_size,'local_path':str(path)}
archive=R/(a.run+'.tar.gz');assert sha(archive)==ctl['archive_sha256']
with tarfile.open(archive) as tar:
 raw_names=[m.name.removeprefix('./') for m in tar.getmembers() if m.isfile()]
 assert all(not m.issym() and not m.islnk() for m in tar.getmembers())
 for name in raw_names:
  f=(d/name).resolve();assert d.resolve() in f.parents
  assert hashlib.sha256(tar.extractfile('./'+name).read()).hexdigest()==sha(f)
for name in raw_names:
 path=d/name;assert path.is_file() and not path.is_symlink();known['result/'+name]={'sha256':sha(path),'bytes':path.stat().st_size,'local_path':str(path)}
known['process.log']=known['result/process.log']
exit_copy=R/(a.run+'-retained-exit.txt');exit_copy.write_text(str(ctl['returncode']));known['exit']={'sha256':sha(exit_copy),'bytes':exit_copy.stat().st_size,'local_path':str(exit_copy)}
mode='plan'
if a.execute_plan:
 plan=json.loads(a.execute_plan.read_text());assert plan['root']==root and plan['status']=='plan_only'
 for q in plan['files']:assert q['relative'] in known and q['sha256']==known[q['relative']]['sha256'] and q['bytes']==known[q['relative']]['bytes']
 expected=plan['files'];mode='delete'
else:expected=[]
remote=r"""
from pathlib import Path
import json,hashlib,sys,shutil
x=json.load(sys.stdin);p=Path(x['root']);assert p.parent==Path('/dev/shm') and p.name.startswith('opi-prefill100-probe-') and p.is_dir() and p.resolve()==p
free=shutil.disk_usage(p).free;files=[];dirs=[]
for f in p.rglob('*'):
 assert not f.is_symlink() and f.resolve()==f
 if f.is_dir():dirs.append(f);continue
 assert f.is_file();n=str(f.relative_to(p));assert n in x['known'],n
 k=x['known'][n];h=hashlib.sha256(f.read_bytes()).hexdigest();size=f.stat().st_size
 assert h==k['sha256'] and size==k['bytes'],n
 files.append({'relative':n,'sha256':h,'bytes':size,'local_path':k['local_path']})
assert {q['relative'] for q in files}==set(x['known'])
if x['mode']=='delete':
 assert sorted(files,key=lambda q:q['relative'])==sorted(x['expected'],key=lambda q:q['relative'])
 for q in files:(p/q['relative']).unlink()
 for f in sorted(dirs,key=lambda z:len(z.parts),reverse=True):f.rmdir()
 p.rmdir()
print(json.dumps({'root':str(p),'status':'plan_only' if x['mode']=='plan' else 'removed_verified_tmpfs_copies','files':files,'bytes':sum(q['bytes'] for q in files),'free_before':free,'free_after':shutil.disk_usage('/dev/shm').free}))
"""
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
with b.hostlock():
 for k in known.values():assert sha(k['local_path'])==k['sha256']
 z=b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(remote),input=json.dumps({'root':root,'known':known,'mode':mode,'expected':expected}).encode(),capture_output=True,timeout=90,check=True)
out=json.loads(z.stdout);out['source_sha256']=sha(Path(__file__));out['unix']=time.time();dest=R/(a.run+'-tmpfs-'+mode+'.json');dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'}))
