from pathlib import Path
import sys,json,time,shlex,hashlib,tarfile,traceback
P=Path(__file__).resolve().parent;sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
run='first-chunk-grouped-r1';remote='/opt/opi-vlm/intern-prefill100-20261025/'+run;output='/dev/shm/'+run
status=P/'controller.json';assert not status.exists(),'unique run required'
s={'status':'starting','run':run,'remote':remote,'output':output,'started':time.time()}
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2));t.replace(status)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def locked(cmd,**kw):return b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=kw.pop('timeout',40),**kw)
def recover():
 packet=locked('tar -czf - -C '+shlex.quote(output)+' .',capture_output=True,timeout=180,check=True)
 arc=P/(run+'.tar.gz');arc.write_bytes(packet.stdout);dest=P/'results'/run;dest.mkdir(parents=True,exist_ok=False)
 with tarfile.open(arc) as tar:
  for m in tar.getmembers():
   path=(dest/m.name).resolve();assert path==dest.resolve() or dest.resolve() in path.parents
   assert not m.issym() and not m.islnk()
  tar.extractall(dest)
 return dest
save()
try:
 with b.hostlock():
  before=locked('df -B1 --output=avail /dev/shm /opt/opi-vlm; cat /proc/meminfo',capture_output=True,text=True,check=True);s['health_before']=before.stdout
  lines=before.stdout.splitlines();assert int(lines[1])>700*1024**2 and int(lines[2])>1024**3
  available=int(next(x.split()[1] for x in lines if x.startswith('MemAvailable:')))*1024;assert available>2*1024**3
  locked('mkdir '+shlex.quote(remote)+' && test ! -e '+shlex.quote(output),check=True)
  for name,f in {'resident-first-chunk-grouped':P/'runtime/resident-first-chunk-grouped','board_first.py':P/'board_first.py','resident_worker_v1.py':P/'resident_worker_v1.py','config.json':P/'config.json'}.items():
   locked('cat > '+shlex.quote(remote+'/'+name),input=f.read_bytes(),check=True)
   result=locked('sha256sum '+shlex.quote(remote+'/'+name),capture_output=True,text=True,check=True);assert result.stdout.split()[0]==sha(f)
  locked('chmod 755 '+shlex.quote(remote+'/resident-first-chunk-grouped'),check=True)
  command='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 15 1200 /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+shlex.quote(remote+'/board_first.py')+' '+shlex.quote(output)
  shell=command+' > '+shlex.quote(remote+'/process.log')+' 2>&1; rc=$?; printf "%s" "$rc" > '+shlex.quote(remote+'/exit')
  b.ssh('nohup sh -c '+shlex.quote(shell)+' </dev/null >/dev/null 2>&1 &',check=True,timeout=20)
  s.update(status='running');save();deadline=time.monotonic()+1320
  while True:
   time.sleep(15);z=b.ssh('cat '+shlex.quote(remote+'/exit'),capture_output=True,text=True,timeout=20)
   if z.returncode==0:s['returncode']=int(z.stdout.strip());break
   if time.monotonic()>deadline:raise TimeoutError('deadline, no automatic restart')
  s['process_log']=locked('cat '+shlex.quote(remote+'/process.log'),capture_output=True,text=True,check=True).stdout
  dest=recover();s.update(status='complete' if s['returncode']==0 and json.loads((dest/'meta.json').read_text())['status']=='complete' else 'failed',result=str(dest),ended=time.time());save()
except BaseException:
 s.update(status='failed',error=traceback.format_exc(),ended=time.time());save();raise
