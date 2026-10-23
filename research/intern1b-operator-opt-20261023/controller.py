from pathlib import Path
import json,sys,subprocess,shlex,time,traceback
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
r=Path(__file__).resolve().parent;remote='/opt/opi-vlm/intern1b-operator-opt-20261023'
def status(phase,**kw):
 x=dict(phase=phase,time=time.time(),**kw);(r/'status.json').write_text(json.dumps(x,indent=2));print(x,flush=True)
try:
 with b.hostlock():
  status('deploying');b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock mkdir -p '+remote,check=True,timeout=30)
  for p in r.iterdir():
   if p.suffix in ['.c','.h'] or p.name in ['resident_worker.py','board_run.py','inputs.json']:b.scp(p,b.HOST+':'+remote+'/'+p.name,check=True,timeout=60)
  cmd='cd '+remote+' && cc -O2 -Wall -Wextra -Werror -DMODEL_DIM=896 -DMODEL_KV=2 -DMODEL_HD=64 -DROPE_BASE=1000000.0 -DFUSED_LAYERS=2 -DKEY_CACHE_TRANSPOSED=0 -DHEAD_SHARDS=4 -DHEAD_CHUNK=49152 resident_session.c -lNBGlinker -lVIPhal -lm -o resident'
  (r/'build-command.txt').write_text(cmd+'\n');status('building');b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),check=True,timeout=180)
  status('benchmarking');z=b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock timeout --kill-after=10s 1800s /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+remote+'/board_run.py',timeout=1830)
  status('collecting',exit=z.returncode)
  subprocess.run(['sshpass','-e','rsync','-rtz','-e','ssh '+' '.join(b.COMMON),b.HOST+':'+remote+'/results/',str(r/'results')+'/'],env=b.env(),check=True,timeout=180)
  status('returned',exit=z.returncode)
except BaseException as e:status('failed',error=repr(e),trace=traceback.format_exc());raise
