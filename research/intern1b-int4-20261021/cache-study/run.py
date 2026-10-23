from pathlib import Path
import fcntl,json,sys,shlex,subprocess,time,traceback
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
r=Path(__file__).resolve().parent;remote='/opt/opi-vlm/intern1b-int4-side-20261021/cache-study'
def status(phase,**kw):
 out={'phase':phase,'time':time.time(),**kw};(r/'status.json').write_text(json.dumps(out,indent=2));print(out,flush=True)
try:
 lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:status('board-busy');raise
 status('deploying')
 b.ssh('flock -n /run/opi-npu-kv-benchmark.lock mkdir -p '+remote,check=True,timeout=30)
 for p in r.iterdir():
  if p.name in ['resident_worker.py','board_cache.py','inputs.json']:
   b.scp(p,b.HOST+':'+remote+'/'+p.name,check=True,timeout=60)
 status('benchmarking')
 z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock timeout --kill-after=10s 1200s /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+remote+'/board_cache.py',timeout=1230)
 status('collecting',exit=z.returncode)
 subprocess.run(['sshpass','-e','rsync','-rtz','-e','ssh '+' '.join(b.COMMON),b.HOST+':'+remote+'/results/',str(r/'results')+'/'],env=b.env(),check=True,timeout=180)
 status('returned',exit=z.returncode)
except BaseException as e:status('failed',error=repr(e),trace=traceback.format_exc());raise
