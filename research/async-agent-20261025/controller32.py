import sys,time,json,subprocess,traceback,shlex,threading
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access as b
ROOT=Path(__file__).resolve().parent
run=sys.argv[1];seconds=sys.argv[2];remote='/opt/opi-vlm/async-agent32-20261025'
status={'run':run,'seconds':seconds,'status':'starting'}
def save(): (ROOT/(run+'-controller.json')).write_text(json.dumps(status,indent=2))
save()
try:
 with b.hostlock():
  b.ssh('mkdir -p '+remote,check=True,timeout=20)
  for name in ['harness32.py','context_rag.py','atoms.json']:
   src=ROOT/name if name!='atoms.json' else ROOT.parent/'home-companion-atoms128-v2-20261022/atoms.json'
   b.scp(str(src),b.HOST+':'+remote+'/'+name,check=True,timeout=40)
  status['status']='running';save()
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout '+str(int(seconds)+200)+' /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+remote+'/harness32.py --seconds '+seconds+' --out '+remote+'/'+run
  poll_stop=threading.Event()
  def progress():
   while not poll_stop.wait(45):
    try:
     rr=b.ssh('tail -n 1 '+remote+'/'+run+'/resources.jsonl',capture_output=True,text=True,timeout=15)
     if rr.returncode==0 and rr.stdout.strip():
      row=json.loads(rr.stdout.strip().splitlines()[-1]);(ROOT/(run+'-progress.json')).write_text(json.dumps(row,ensure_ascii=False))
    except Exception as e:print('progress-only warning',type(e).__name__,flush=True)
  th=threading.Thread(target=progress,daemon=True);th.start()
  try:result=b.ssh(cmd,capture_output=True,text=True,timeout=int(seconds)+240)
  finally:poll_stop.set();th.join(20)
  (ROOT/(run+'-board.log')).write_text(result.stdout+'\n'+result.stderr)
  dest=ROOT/'results'/run;dest.mkdir(parents=True,exist_ok=True)
  for name in ['meta.json','events.jsonl','resources.jsonl','npu.log']:
   r=b.scp(b.HOST+':'+remote+'/'+run+'/'+name,str(dest/name),timeout=60,capture_output=True,text=True)
   if r.returncode: print(r.stderr,flush=True)
  status.update(status='finished',returncode=result.returncode);save()
except BaseException:
 status.update(status='failed',error=traceback.format_exc());save();raise
