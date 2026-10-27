from pathlib import Path
import sys,json,time,tarfile
P=Path(__file__).resolve().parent;R=P.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
remote='/tmp/opi-vit224-freshness-r3';old='/tmp/opi-vit224-concurrent-r1';status=P/'controller-status.json';assert not status.exists();state={'status':'preparing','at':time.time()}
def save():state['updated']=time.time();status.write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 with b.hostlock():
  locked('test ! -e '+remote+' && mkdir '+remote+' && cp '+old+'/*.py '+old+'/*.json '+old+'/vision-profile '+old+'/resident-visual '+remote+'/ && ln -s '+old+'/graphs '+remote+'/graphs && ln -s '+old+'/images '+remote+'/images && ln -s '+old+'/vectors '+remote+'/vectors',check=True)
  upload(P/'board_concurrent.py',remote+'/board_concurrent.py');upload(P/'config.json',remote+'/config.json')
  state['status']='running';save()
  z=locked('cd '+remote+' && python3 board_concurrent.py > controller.log 2>&1',capture_output=True,text=True,timeout=1000)
  # Always retrieve logs, even when preflight fails before results exists.
  data=locked('cd '+remote+' && tar czf - controller.log '+('results' if z.returncode==0 else '--ignore-failed-read results'),capture_output=True,timeout=120).stdout;(P/'raw.tar.gz').write_bytes(data)
  with tarfile.open(P/'raw.tar.gz') as t:
   assert all(not m.issym() and not m.islnk() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(P)
  assert z.returncode==0,z.stderr
  x=json.loads((P/'results/summary.json').read_text());assert x['status']=='complete' and x['vision_closed']['driver_bytes']==0
  health=locked('cat /proc/meminfo; cat /sys/kernel/debug/viplite/mem_profile',capture_output=True,text=True,check=True);(P/'post-health.txt').write_text(health.stdout)
  state.update(status='complete',returncode=z.returncode);save();print(json.dumps(state))
except BaseException as e:state.update(status='failed-or-remote-unknown',error=str(e));save();raise
