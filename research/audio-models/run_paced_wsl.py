import argparse,hashlib,json,shlex,sys,tarfile
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access as b
p=argparse.ArgumentParser();p.add_argument('--name',default='paced-v1');p.add_argument('--vad-silence',type=float,default=.25);a=p.parse_args();assert a.name.replace('-','').isalnum() and .1<=a.vad_silence<=2
repo=Path(__file__).resolve().parents[2];out=repo/'results/audio-models-20261018';remote='/opt/opi-vlm/audio-lab-20261018'
with b.hostlock():
 for name in ['paced_chain.py','board_trials.py','audio_worker.py']:
  b.scp(repo/'research/audio-models'/name,'root@192.168.10.11:'+remote+'/'+name,check=True,timeout=60)
 cmd=f'cd {remote} && OPI_AUDIO_PACED_NAME={a.name} OPI_AUDIO_VAD_SILENCE={a.vad_silence} PYTHONPATH=deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u paced_chain.py > {a.name}.log 2>&1'
 r=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=720)
 b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(f'cd {remote} && tar -czf {a.name}.tar.gz {a.name} {a.name}.log'),check=True,timeout=60)
 dest=out/(a.name+'.tar.gz');b.scp('root@192.168.10.11:'+remote+'/'+dest.name,dest,check=True,timeout=120)
 with tarfile.open(dest) as t:
  for m in t.getmembers():assert not m.name.startswith('/') and '..' not in Path(m.name).parts and not m.issym() and not m.islnk()
  t.extractall(out)
 (out/(a.name+'-transfer.json')).write_text(json.dumps({'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'exit_code':r.returncode},indent=2));r.check_returncode()
 for n in ['paced_chain.py','audio_worker.py','run_paced_wsl.py']:(out/a.name/('executed-'+n)).write_bytes((repo/'research/audio-models'/n).read_bytes())
print((out/(a.name+'.log')).read_text())
