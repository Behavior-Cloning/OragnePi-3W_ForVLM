import hashlib,json,shlex,sys,tarfile
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access as b
repo=Path(__file__).resolve().parents[2];out=repo/'results/audio-models-20261018';remote='/opt/opi-vlm/audio-lab-20261018'
with b.hostlock():
 for name in ['paced_chain.py','board_trials.py','audio_worker.py']:
  b.scp(repo/'research/audio-models'/name,'root@192.168.10.11:'+remote+'/'+name,check=True,timeout=60)
 cmd=f'cd {remote} && PYTHONPATH=deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u paced_chain.py > paced-v1.log 2>&1'
 r=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=720)
 b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(f'cd {remote} && tar -czf paced-v1.tar.gz paced-v1 paced-v1.log'),check=True,timeout=60)
 dest=out/'paced-v1.tar.gz';b.scp('root@192.168.10.11:'+remote+'/'+dest.name,dest,check=True,timeout=120)
 with tarfile.open(dest) as t:
  for m in t.getmembers():assert not m.name.startswith('/') and '..' not in Path(m.name).parts and not m.issym() and not m.islnk()
  t.extractall(out)
 (out/'paced-v1-transfer.json').write_text(json.dumps({'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'exit_code':r.returncode},indent=2));r.check_returncode()
print((out/'paced-v1.log').read_text())
