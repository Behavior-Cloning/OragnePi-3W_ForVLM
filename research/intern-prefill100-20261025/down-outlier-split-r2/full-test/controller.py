from pathlib import Path
import sys,json,shlex,hashlib,time,os,traceback
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent.parent));from controller_down_outputfp16_tmpfs import b,locked
remote='/tmp/opi-outlier-full-c448-r1';state=R/'controller.json';assert not state.exists()
s={'status':'preparing','pid':os.getpid(),'remote':remote,'started':time.time()}
def save():state.write_text(json.dumps(s,indent=2))
save()
try:
 with b.hostlock():
  locked('mkdir '+remote,check=True)
  files={n:R/n for n in ['board_matrix.py','resident_worker_v1.py','inputs.json','context_rag.py','fixture_validation.py','config.json']}
  files.update({'outlier32.nb':R.parent/'full-c448-m32/export_nbg_unify/network_binary.nb','outlier8.nb':R.parent/'full-c448-m8/export_nbg_unify/network_binary.nb'})
  expected={}
  for n,p in files.items():
   expected[n]=hashlib.sha256(p.read_bytes()).hexdigest();b.scp(str(p),b.HOST+':'+remote+'/'+n,check=True,timeout=120)
  code='from pathlib import Path;import hashlib,json;p=Path('+repr(remote)+');print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in '+repr(list(files))+'}))'
  z=locked('python3 -c '+shlex.quote(code),capture_output=True,text=True,check=True);assert json.loads(z.stdout)==expected
  (R/'deployment-sha.json').write_text(json.dumps(expected,indent=2))
  locked('python3 -c '+shlex.quote("from pathlib import Path;r=Path('/tmp/opi-outlier-full-c448-r1');base=Path('/tmp/opi-c448-qualified-r3/original');g=r/'graphs';g.mkdir();(g/'tail8').mkdir();[(g/('layer'+str(i)+'.nb')).symlink_to(base/('layer'+str(i)+'.nb')) for i in list(range(0,24,2))+[24,25,26,27]];(g/'embeddings.f32').symlink_to(base/'embeddings.f32');[(g/('prefill-layer'+str(i)+'.nb')).symlink_to(r/'outlier32.nb' if i==2 else base/('prefill-layer'+str(i)+'.nb')) for i in range(0,24,2)];[(g/'tail8'/('prefill-layer'+str(i)+'.nb')).symlink_to(r/'outlier8.nb' if i==2 else base/'tail8'/('prefill-layer'+str(i)+'.nb')) for i in range(0,24,2)]"),check=True)
  command='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 1000 /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+remote+'/board_matrix.py'
  wrapper=command+' >'+remote+'/process.log 2>&1; result=$?; echo "$result" >'+remote+'/exit'
  z=b.ssh('nohup sh -c '+shlex.quote(wrapper)+' </dev/null >/dev/null 2>&1 & echo $!',capture_output=True,text=True,check=True,timeout=20)
  s.update(status='running',board_wrapper_pid=int(z.stdout.strip()));save()
  for i in range(110):
   time.sleep(10)
   z=b.ssh('cat '+remote+'/exit',capture_output=True,text=True,timeout=20)
   if z.returncode==0:s['returncode']=int(z.stdout.strip());break
  else:raise TimeoutError('observation deadline; board run may still be live')
  dest=R/'results';dest.mkdir(exist_ok=False)
  z=b.ssh('tar -czf - -C '+remote+' events.jsonl resources.jsonl status.json process.log original.log outlier8.log original-repeat.log original-dumps outlier8-dumps original-repeat-dumps',capture_output=True,timeout=60)
  (R/'results.tar.gz').write_bytes(z.stdout)
  import tarfile
  with tarfile.open(R/'results.tar.gz') as t:
   for m in t.getmembers():assert not m.issym() and not m.islnk() and '..' not in Path(m.name).parts
   t.extractall(dest)
  z=locked('cat /sys/kernel/debug/viplite/mem_profile; cat /proc/meminfo',capture_output=True,text=True)
  (dest/'post-health.txt').write_text(z.stdout)
  s.update(status='complete' if s['returncode']==0 else 'failed',ended=time.time());save()
except BaseException:s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
