import sys,time,json,subprocess,shlex,tarfile
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
p=Path(__file__).resolve().parent;run=sys.argv[1];secs=int(sys.argv[2]);variant=sys.argv[3];h='ttft_matrix.py';remote='/opt/opi-vlm/async-ttft-20261025';out=p/'results'/run
status={'run':run,'seconds':secs,'variant':variant,'status':'starting'}
def save(): (p/(run+'-controller.json')).write_text(json.dumps(status))
save()
try:
 with b.hostlock():
  b.ssh('mkdir -p '+remote,check=True,timeout=20)
  for name in [h,'context_rag.py','atoms.json']:
   src=p/name if name!='atoms.json' else p.parent/'home-companion-atoms128-v2-20261022/atoms.json';b.scp(str(src),b.HOST+':'+remote+'/'+name,check=True,timeout=30)
  exitfile=remote+'/'+run+'.exit';log=remote+'/'+run+'.log'
  command='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 15 '+str(secs+240)+' /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+remote+'/'+h+' '+remote+'/'+run
  shell=command+' >'+log+' 2>&1; rc=$?; printf "%s" "$rc" >'+exitfile
  b.ssh('nohup sh -c '+shlex.quote(shell)+' </dev/null >/dev/null 2>&1 &',check=True,timeout=20)
  status['status']='running';save();deadline=time.monotonic()+secs+360
  while True:
   time.sleep(15)
   r=b.ssh('cat '+exitfile,capture_output=True,text=True,timeout=20)
   if r.returncode==0:status['returncode']=int(r.stdout.strip());break
   if time.monotonic()>deadline:raise TimeoutError('detached board deadline exceeded')
  b.ssh('tar -czf '+remote+'/'+run+'.tar.gz -C '+remote+'/'+run+' fixtures.json events.jsonl resources.jsonl',check=True,timeout=60)
  b.scp(b.HOST+':'+remote+'/'+run+'.tar.gz',str(p/(run+'.tar.gz')),check=True,timeout=120)
  out.mkdir(parents=True,exist_ok=True)
  with tarfile.open(p/(run+'.tar.gz')) as t:t.extractall(out)
  status['status']='finished';save()

except BaseException:
 import traceback
 status.update(status='failed',error=traceback.format_exc());save();raise
