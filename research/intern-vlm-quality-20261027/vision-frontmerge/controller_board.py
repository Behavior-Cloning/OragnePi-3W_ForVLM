from pathlib import Path
import sys,json,hashlib,time,tarfile,traceback
D=Path(__file__).resolve().parent;R=D.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'));from controller_down_outputfp16_tmpfs import b,locked
remote='/opt/opi-vlm/experiments/vision-frontmerge-r2';sf=D/'board-r2-status.json';assert not sf.exists();data=(D/'board-upload.tar.gz').read_bytes();s={'state':'starting','time':time.time(),'remote':remote}
def save():sf.write_text(json.dumps(s,indent=2))
save()
try:
 with b.hostlock():
  locked('test ! -e '+remote+' && mkdir -p '+remote,check=True,timeout=30)
  locked('cat > '+remote+'/upload.tar.gz',input=data,check=True,timeout=120)
  z=locked('sha256sum '+remote+'/upload.tar.gz',capture_output=True,text=True,check=True,timeout=30);assert z.stdout.split()[0]==hashlib.sha256(data).hexdigest()
  locked('tar -xzf '+remote+'/upload.tar.gz -C '+remote+' && chmod +x '+remote+'/vision-worker',check=True,timeout=30)
  s['state']='running';save();print('board screen running',flush=True)
  z=locked('python3 '+remote+'/board_bench.py',capture_output=True,text=True,timeout=1000);s['returncode']=z.returncode;s['stdout']=z.stdout;s['stderr']=z.stderr[-2000:];print(z.stdout,flush=True)
  raw=locked('tar -czf - -C '+remote+' results',capture_output=True,timeout=120,check=True).stdout;(D/'board-r2-results.tar.gz').write_bytes(raw)
  dest=D/'board-r2-results';dest.mkdir(exist_ok=False)
  with tarfile.open(D/'board-r2-results.tar.gz') as t:
   assert all(not x.issym() and not x.islnk() and '..' not in Path(x.name).parts and not x.name.startswith('/') for x in t.getmembers());t.extractall(dest)
  s['state']='collected';s['ended']=time.time();save()
except Exception:
 s['state']='failed';s['error']=traceback.format_exc();save();print(s['error'],flush=True)
