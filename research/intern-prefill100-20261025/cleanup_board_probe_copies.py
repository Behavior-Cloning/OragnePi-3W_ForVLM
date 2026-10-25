#!/usr/bin/env python3
"""Remove only hash-verified board copies of this task's completed probes."""
from pathlib import Path
import hashlib,json,shlex,sys,time
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
P=Path(__file__).resolve().parent
OUT=P/'cleanup-board-probe-copies-20261025.json'
assert not OUT.exists(),'unique cleanup evidence required'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
rows={}
def add(remote,local,expected=None):
 local=Path(local);h=sha(local)
 if expected:assert h==expected
 assert remote.startswith('/opt/opi-vlm/intern-prefill100-20261025/') and '..' not in Path(remote).parts
 rows[remote]={'remote':remote,'windows_backup':str(local),'sha256':h,'bytes':local.stat().st_size}
for run in ['single-block-c512-r1','fp16conv-m16-r1','fp16conv-m32-r1']:
 status=json.loads((P/(run+'-controller.json')).read_text());assert status['status']=='complete' and status['returncode']==0
 cfg=json.loads((P/(run+'-config.json')).read_text())
 if run=='single-block-c512-r1':
  for row in cfg['sequence']:
   key='c512-m%d-l%d'%(row['query_tokens'],row['first_layer'])
   add(row['nb'],P/'graphs/cases'/key/'export_nbg_unify/network_binary.nb',row['nb_sha256'])
   add(row['input'],P/'fixtures-canonical'/key/'probe-input.f32',row['input_sha256'])
 else:
  row=cfg['sequence'][1];n='conv1x1-m16-l2' if 'm16' in run else 'conv1x1-c512-m32-l2'
  add(row['nb'],P/'graphs/candidates-fp16'/n/'export_nbg_unify/network_binary.nb',row['nb_sha256'])
 for local in (P/'results'/run).glob('*.f32'):
  add('/opt/opi-vlm/intern-prefill100-20261025/runs/'+run+'/result/'+local.name,local)
# Fullbaseline f32 has been independently rehashed; remove only board copies.
run='fullbaseline-c1280-r1';base=P/'fullbaseline/results'/run
st=json.loads((P/'fullbaseline'/(run+'-controller.json')).read_text());assert st['status']=='complete' and st['returncode']==0
assert (base/'analysis.json').exists()
for local in base.rglob('*.f32'):add(st['remote']+'/result/'+str(local.relative_to(base)),local)
manifest={'status':'prepared','scope':'only task-owned completed board probe upload copies and f32 result copies; Windows backups retained; original production graphs, source weights and logs retained','created':time.time(),'files':list(rows.values()),'bytes_planned':sum(x['bytes'] for x in rows.values())}
OUT.write_text(json.dumps(manifest,indent=2)+'\n')
script="""import hashlib,json,pathlib,stat,sys,shutil
rows=json.load(sys.stdin)
for r in rows:
 p=pathlib.Path(r['remote']);assert stat.S_ISREG(p.lstat().st_mode) and not p.is_symlink()
 assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
before=shutil.disk_usage('/opt/opi-vlm').free
for r in rows:pathlib.Path(r['remote']).unlink()
print(json.dumps({'verified_files':len(rows),'removed_bytes':sum(r['bytes'] for r in rows),'free_before':before,'free_after':shutil.disk_usage('/opt/opi-vlm').free}))
"""
with b.hostlock():
 command='python3 -c '+shlex.quote(script)
 x=b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(command),input=json.dumps(list(rows.values())).encode(),capture_output=True,timeout=90,check=True)
 manifest.update(status='complete',finished=time.time(),board=json.loads(x.stdout))
 OUT.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['board']))
