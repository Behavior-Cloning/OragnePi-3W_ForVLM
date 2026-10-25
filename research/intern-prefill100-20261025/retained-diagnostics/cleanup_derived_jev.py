from pathlib import Path
import hashlib,json,sys,shlex,time
P=Path(__file__).resolve().parent
LOCAL=P.parents[1]/'jev-validation/models'
REMOTE='/opt/opi-vlm/jev-lab-20261020/models'
NAMES=['laya/laya-int8.onnx','nano33/nano33-int8.onnx','nano33/nano33-fp32.onnx']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
state={'status':'preparing','started':time.time(),'files':[],'scope':'Only3 derived ONNX board copies, exact Windows retained bytes; no original weights, venv or active Intern/speech artifacts.'}
def save():(P/'derived-jev-cleanup.json').write_text(json.dumps(state,indent=2)+'\n')
try:
 save()
 with b.hostlock():
  def locked(cmd,**kw):return b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),**kw)
  for name in NAMES:
   remote=REMOTE+'/'+name;local=LOCAL/name
   z=locked('test -f '+shlex.quote(remote)+' && test ! -L '+shlex.quote(remote)+' && sha256sum '+shlex.quote(remote),capture_output=True,text=True,timeout=120,check=True)
   want=z.stdout.split()[0]
   if not local.exists():
    assert name=='nano33/nano33-fp32.onnx';tmp=local.with_suffix('.onnx.backup-part');assert not tmp.exists()
    z=locked('cat '+shlex.quote(remote),capture_output=True,timeout=180,check=True);tmp.write_bytes(z.stdout);del z
    assert sha(tmp)==want;tmp.replace(local)
   assert local.is_file() and not local.is_symlink() and sha(local)==want
   state['files'].append({'remote_path':remote,'retained_local_path':str(local),'sha256':want,'bytes':local.stat().st_size});save()
  script=r"""
from pathlib import Path
import json,sys,hashlib,shutil,os
x=json.load(sys.stdin);root=Path('/opt/opi-vlm/jev-lab-20261020/models');allowed={'laya/laya-int8.onnx','nano33/nano33-int8.onnx','nano33/nano33-fp32.onnx'}
assert {str(Path(f['remote_path']).relative_to(root)) for f in x}==allowed
for f in x:
 p=Path(f['remote_path']);assert p.is_file() and not p.is_symlink() and p.resolve()==p and p.stat().st_size==f['bytes']
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(4194304),b''):h.update(chunk)
 assert h.hexdigest()==f['sha256']
# Known benchmark ownership is covered by both locks; reject any live direct Jev script as well.
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 if proc.parent.name==str(os.getpid()):continue
 try:args=proc.read_bytes().split(b'\0')
 except OSError:continue
 assert not any(a.startswith(b'/opt/opi-vlm/jev-lab-20261020/') and a.endswith(b'.py') for a in args),'live Jev script'
before=shutil.disk_usage(root).free
for f in x:Path(f['remote_path']).unlink()
print(json.dumps({'deleted_paths':[f['remote_path'] for f in x],'deleted_bytes':sum(f['bytes'] for f in x),'free_before':before,'free_after':shutil.disk_usage(root).free}))
"""
  z=locked('python3 -c '+shlex.quote(script),input=json.dumps(state['files']).encode(),capture_output=True,timeout=180,check=True);state.update(status='complete',ended=time.time(),result=json.loads(z.stdout));save();print(json.dumps(state['result']))
except BaseException:
 import traceback
 state.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
