import hashlib,json,shlex,sys,tarfile
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access as b
repo=Path(__file__).resolve().parents[2];assets=Path('/mnt/f/Files/OPi-Lab/audio-models-20261018');out=repo/'results/audio-models-20261018';remote='/opt/opi-vlm/audio-lab-20261018'
with b.hostlock():
 for file in [repo/'research/audio-models/quality_trials.py',assets/'speaker-samples.tar']:
  b.scp(file,'root@192.168.10.11:'+remote+'/'+file.name,check=True,timeout=60)
 h=hashlib.sha256((assets/'speaker-samples.tar').read_bytes()).hexdigest()
 cmd=f'cd {remote} && echo '+shlex.quote(h+'  speaker-samples.tar')+' | sha256sum -c - && tar -xf speaker-samples.tar && PYTHONPATH=deps OPENBLAS_NUM_THREADS=1 taskset -c 0,1 python3 -u quality_trials.py > quality-v1.log 2>&1'
 r=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=600)
 b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(f'cd {remote} && tar -czf quality-v1.tar.gz quality-v1 quality-v1.log speaker-samples/manifest.json'),check=True,timeout=60)
 dest=out/'quality-v1.tar.gz';b.scp('root@192.168.10.11:'+remote+'/'+dest.name,dest,check=True,timeout=120)
 with tarfile.open(dest) as t:
  for m in t.getmembers():assert not m.name.startswith('/') and '..' not in Path(m.name).parts and not m.issym() and not m.islnk()
  t.extractall(out)
 (out/'quality-v1-transfer.json').write_text(json.dumps({'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'exit_code':r.returncode},indent=2))
 r.check_returncode()
print((out/'quality-v1.log').read_text())
