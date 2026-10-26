from pathlib import Path
import sys,json,shlex,hashlib,time,os,traceback,tarfile
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent.parent));from controller_down_outputfp16_tmpfs import b,locked
remote='/opt/opi-vlm/experiments/main-m64-r11';state=R/'controller.json';assert not state.exists()
cfg=json.loads((R/'config.json').read_text())
files={n:R/n for n in ['board_matrix.py','resident_worker_v1.py','inputs.json','context_rag.py','fixture_validation.py','schedule.py','config.json']}
files['resident-main64']=R.parent/'runtime/resident-main64'
files.update({n:Path(p) for n,p in json.loads((R/'upload-sources.json').read_text()).items()})
s={'status':'preparing','pid':os.getpid(),'remote':remote,'started':time.time(),'upload_bytes':sum(p.stat().st_size for p in files.values())}
def save():state.write_text(json.dumps(s,indent=2))
save()
try:
 with b.hostlock():
  code="import shutil;assert shutil.disk_usage('/opt').free>"+str(s['upload_bytes']+250*1024**2)
  locked('python3 -c '+shlex.quote(code),check=True)
  locked('mkdir '+remote+'; mkdir '+remote+'/main64',check=True)
  expected={}
  for n,p in files.items():
   s.update(status='uploading',upload_file=n);save()
   expected[n]=hashlib.sha256(p.read_bytes()).hexdigest();b.scp(str(p),b.HOST+':'+remote+'/'+n,check=True,timeout=120)
  code='from pathlib import Path;import hashlib,json;p=Path('+repr(remote)+');print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in '+repr(list(files))+'}))'
  z=locked('python3 -c '+shlex.quote(code),capture_output=True,text=True,check=True);assert json.loads(z.stdout)==expected
  (R/'deployment-sha.json').write_text(json.dumps(expected,indent=2))
  code='from pathlib import Path;p=Path('+repr(remote+'/main64')+');src=Path('+repr(cfg['parent_graph'])+');[(p/f.name).symlink_to(f) for f in src.iterdir() if f.name.startswith("layer") and f.suffix==".nb" or f.name=="embeddings.f32"]'
  locked('python3 -c '+shlex.quote(code)+'; chmod 755 '+remote+'/resident-main64',check=True)
  command='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 1600 /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+remote+'/board_matrix.py'
  wrapper=command+' >'+remote+'/process.log 2>&1; result=$?; echo "$result" >'+remote+'/exit'
  z=b.ssh('nohup sh -c '+shlex.quote(wrapper)+' </dev/null >/dev/null 2>&1 & echo $!',capture_output=True,text=True,check=True,timeout=20)
  s.update(status='running',board_wrapper_pid=int(z.stdout.strip()));save()
  for i in range(170):
   time.sleep(10)
   z=b.ssh('cat '+remote+'/exit',capture_output=True,text=True,timeout=20)
   if z.returncode==0:s['returncode']=int(z.stdout.strip());break
  else:raise TimeoutError('observation deadline; board run may still be live; do not restart')
  dest=R/'results';dest.mkdir(exist_ok=False)
  # Keep partial evidence even when board exits before all variants exist.
  names=['events.jsonl','resources.jsonl','status.json','process.log']+[v+suffix for v in cfg['variants'] for suffix in ['.log','-dumps']]
  z=b.ssh('tar --ignore-failed-read -czf - -C '+remote+' '+' '.join(names),capture_output=True,timeout=90)
  (R/'results.tar.gz').write_bytes(z.stdout);(R/'archive-stderr.txt').write_bytes(z.stderr)
  with tarfile.open(R/'results.tar.gz') as t:
   for m in t.getmembers():assert not m.issym() and not m.islnk() and '..' not in Path(m.name).parts
   t.extractall(dest)
  z=locked('cat /sys/kernel/debug/viplite/mem_profile; cat /proc/meminfo',capture_output=True,text=True)
  (dest/'post-health.txt').write_text(z.stdout)
  s.update(status='complete' if s['returncode']==0 else 'failed',ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
