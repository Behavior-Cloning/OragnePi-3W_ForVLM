"""Copy experiment scripts, run one locked board coordinator, retain raw evidence."""
import argparse,hashlib,json,shlex,sys,tarfile
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access as b
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['standalone','mixed'],required=True);p.add_argument('--name',required=True);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
repo=Path(__file__).resolve().parents[2];remote='/opt/opi-vlm/audio-lab-20261018';out=repo/'results/audio-models-20261018';out.mkdir(exist_ok=True)
assert a.name.replace('-','').isalnum()
with b.hostlock():
 for name in ['audio_worker.py','board_trials.py']:
  b.scp(repo/'research/audio-models'/name,'root@192.168.10.11:'+remote+'/'+name,check=True,timeout=60)
 (out/(a.name+'-scripts.json')).write_text(json.dumps({n:hashlib.sha256((repo/'research/audio-models'/n).read_bytes()).hexdigest() for n in ['audio_worker.py','board_trials.py']},indent=2))
 command=f'set -o pipefail; cd {remote} && PYTHONPATH=deps OPENBLAS_NUM_THREADS=1 python3 -u board_trials.py --mode {a.mode} --out {a.name} --repeats {a.repeats} 2>&1 | tee {a.name}.log'
 result=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock bash -c '+shlex.quote(command),timeout=7200)
 pack=f'cd {remote} && tar -czf {a.name}.tar.gz {a.name} {a.name}.log'
 b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(pack),check=True,timeout=120)
 dest=out/(a.name+'.tar.gz');b.scp('root@192.168.10.11:'+remote+'/'+dest.name,dest,check=True,timeout=300)
 with tarfile.open(dest) as t:
  for m in t.getmembers():
   assert not m.name.startswith('/') and '..' not in Path(m.name).parts and not m.issym() and not m.islnk()
  t.extractall(out)
 (out/(a.name+'-transfer.json')).write_text(json.dumps({'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'exit_code':result.returncode},indent=2))
 result.check_returncode()
print((out/(a.name+'.log')).read_text())
