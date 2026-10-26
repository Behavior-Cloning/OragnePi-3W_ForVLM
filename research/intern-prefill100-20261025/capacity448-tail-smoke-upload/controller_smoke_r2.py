from pathlib import Path
import sys,json,hashlib,tarfile,shlex,subprocess,time,traceback
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent))
from controller_down_outputfp16_tmpfs import b,locked
remote='/tmp/candidate-c448-tail8-smoke-r2'
state=R/'smoke-r2-execution.json'
assert not state.exists()
s={'status':'verifying','pid':__import__('os').getpid(),'started':time.time()}
def save():state.write_text(json.dumps(s,indent=2))
save()
try:
 with b.hostlock():
  expected={}
  with tarfile.open(R/'upload.tar') as t:
   for m in t.getmembers():
    if m.isfile():expected[m.name]=hashlib.sha256(t.extractfile(m).read()).hexdigest()
  code="from pathlib import Path;import hashlib,json;p=Path("+repr(remote)+");print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in "+repr(list(expected))+"}))"
  z=locked('python3 -c '+shlex.quote(code),timeout=45,capture_output=True,text=True,check=True)
  actual=json.loads(z.stdout);assert actual==expected
  (R/'upload-r2-verified-sha.json').write_text(json.dumps(actual,indent=2))
  b.scp(str(R/'board_smoke_r2.py'),b.HOST+':'+remote+'/board_smoke_r2.py',check=True,timeout=30)
  s['status']='running';save()
  z=locked('timeout -k 10 600 python3 '+remote+'/board_smoke_r2.py',timeout=630,capture_output=True,text=True)
  (R/'smoke-r2-process.log').write_text(z.stdout+z.stderr)
  s['returncode']=z.returncode
  for n in ['events-r2.jsonl','base-r2.log','tail8-r2.log','base-repeat-r2.log']:
   b.scp(b.HOST+':'+remote+'/'+n,str(R/n),timeout=30)
  zhealth=locked('cat /sys/kernel/debug/viplite/mem_profile',capture_output=True,text=True)
  (R/'post-health-r2.txt').write_text(zhealth.stdout)
  s.update(status='complete' if z.returncode==0 else 'failed',ended=time.time());save()
except BaseException:
 s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
