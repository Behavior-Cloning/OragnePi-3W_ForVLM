from pathlib import Path
import sys,json,hashlib,time,tarfile,traceback
D=Path(__file__).resolve().parent;R=D.parents[2];sys.path.insert(0,str(R/'research/intern-prefill100-20261025'));from controller_down_outputfp16_tmpfs import b,locked
m=json.loads((D/'vision-q8-manifest.json').read_text());model=Path(m['output']);remote='/opt/opi-vlm/experiments/vision-q8-r1';parent='/opt/opi-vlm/experiments/vision-frontmerge-r4';sf=D/'q8-board-status.json';assert not sf.exists();s={'state':'starting','time':time.time(),'remote':remote}
def save():sf.write_text(json.dumps(s,indent=2))
save()
try:
 with b.hostlock():
  z=locked("python3 -c 'import shutil;print(shutil.disk_usage(\"/opt\").free)'",capture_output=True,text=True,check=True,timeout=20);free=int(z.stdout);assert free>m['bytes']+96*1024**2,free;s['pre_disk_free']=free
  locked('test ! -e '+remote+' && mkdir -p '+remote,check=True,timeout=30)
  s['state']='uploading_q8';save();print('uploading Q8',flush=True)
  with model.open('rb') as f:locked('cat > '+remote+'/encoder-q8.gguf.part',stdin=f,check=True,timeout=300)
  z=locked('sha256sum '+remote+'/encoder-q8.gguf.part',capture_output=True,text=True,check=True,timeout=60);assert z.stdout.split()[0]==m['output_sha256']
  locked('mv '+remote+'/encoder-q8.gguf.part '+remote+'/encoder-q8.gguf',check=True,timeout=20)
  locked('cat > '+remote+'/board_bench.py',input=(D/'board_q8.py').read_bytes(),check=True,timeout=30)
  locked('ln -s '+parent+'/vision-worker '+remote+'/vision-worker',check=True,timeout=20)
  for side in [224,448]:
   for name in ['image1','image2']:
    file=name+'-'+str(side)+'.f32';locked('ln -s '+parent+'/'+file+' '+remote+'/'+file,check=True,timeout=20)
  s['state']='running';save();print('Q8 screen running',flush=True)
  z=locked('python3 '+remote+'/board_bench.py',capture_output=True,text=True,timeout=500);s['returncode']=z.returncode;s['stdout']=z.stdout;s['stderr']=z.stderr[-2000:];print(z.stdout,flush=True)
  raw=locked('tar -czf - -C '+remote+' results',capture_output=True,timeout=120,check=True).stdout;(D/'q8-board-results.tar.gz').write_bytes(raw)
  dest=D/'q8-board-results';dest.mkdir(exist_ok=False)
  with tarfile.open(D/'q8-board-results.tar.gz') as t:
   assert all(not x.issym() and not x.islnk() and '..' not in Path(x.name).parts and not x.name.startswith('/') for x in t.getmembers());t.extractall(dest)
  s['state']='collected';s['ended']=time.time();save()
except Exception:
 s['state']='failed';s['error']=traceback.format_exc();save();print(s['error'],flush=True)
