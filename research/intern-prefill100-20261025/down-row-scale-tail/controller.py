"""Stage one 30 MB candidate in tmpfs; reuse SHA-frozen originals, double lock."""
from pathlib import Path
import argparse,json,sys,time,shlex,tarfile,traceback
from contract import sha,validate_aba,REMOTE
P=Path(__file__).resolve().parent;R=P.parent
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
run='down-row-scale-tail-r1';status=P/(run+'-controller.json');assert not status.exists(),'Never duplicate run'
cfg=json.loads((P/'config.json').read_text());validate_aba(cfg);remote=REMOTE;out=remote+'/result'
sources={n:P/n for n in ['board_fulli16.py','resident_worker_v1.py','context_rag.py','contract.py','inputs.json','config.json','baseline-config.json','static-audit.json','controller.py']}
sources['candidate.nb']=R/'graphs/down-row-scale-c512-m32-l2/export_nbg_unify/network_binary.nb'
sources['tail-candidate.nb']=R/'graphs/down-row-scale-c512-m8-l2/export_nbg_unify/network_binary.nb'
assert sha(sources['tail-candidate.nb'])==cfg['tail_candidate']['sha256']
sources['tail-static-audit.json']=P/'tail-static-audit.json'
sources['resident-prefixsafe512']=P/'runtime/resident-prefixsafe512'
assert sha(sources['resident-prefixsafe512'])==cfg['variants'][0]['expected_binary_sha256']
assert sources['resident-prefixsafe512'].stat().st_mode & 0o111, 'Runtime must be executable in tar'
sources['atoms.json']=R.parent/'home-companion-atoms128-v2-20261022/atoms.json'
assert sha(sources['candidate.nb'])==cfg['candidate']['sha256'] and sha(sources['inputs.json'])==cfg['expected_fixture_sha256']
hashes={n:sha(f) for n,f in sources.items()};need=sum(f.stat().st_size for f in sources.values())
s={'status':'prepared','run':run,'remote':remote,'started':time.time(),'source_sha256':hashes,'upload_bytes':need,'scope':cfg['scope']}
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2));t.replace(status)
def locked(cmd,**kw):return b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=kw.pop('timeout',40),**kw)
def verify():
 for n,h in hashes.items():assert sha(sources[n])==h,n
 validate_aba(cfg)
def recover():
 locked('mkdir -p '+shlex.quote(out)+' && cp '+shlex.quote(remote+'/process.log')+' '+shlex.quote(out+'/process.log'),check=True)
 packet=locked('tar -czf - -C '+shlex.quote(out)+' .',capture_output=True,timeout=180,check=True)
 arc=P/(run+'.tar.gz');arc.write_bytes(packet.stdout);dest=P/'results'/run;dest.mkdir(parents=True,exist_ok=False)
 with tarfile.open(arc) as tar:
  for m in tar.getmembers():
   path=(dest/m.name).resolve();assert path==dest.resolve() or dest.resolve() in path.parents
   assert not m.issym() and not m.islnk()
  tar.extractall(dest)
 s['archive_sha256']=sha(arc);s['archive_bytes']=arc.stat().st_size
 return dest
verify()
if not args.execute:
 print(json.dumps({'prepared':True,'source_sha256':hashes,'upload_bytes':need,'board_operations':0}));sys.exit(0)
save()
try:
 bundle=P/(run+'-upload.tar')
 with tarfile.open(bundle,'w',dereference=True) as tar:
  for n,f in sources.items():tar.add(f,arcname=n,recursive=False)
 s['bundle_sha256']=sha(bundle)
 with b.hostlock():
  verify();mount=locked('findmnt -no FSTYPE,OPTIONS /dev/shm',capture_output=True,text=True,check=True).stdout.split()
  assert mount[0]=='tmpfs' and 'noexec' not in ''.join(mount[1:]).split(',')
  health=locked('df -B1 --output=avail /dev/shm; cat /proc/meminfo',capture_output=True,text=True,check=True).stdout;s['health_before']=health
  lines=health.splitlines();free=int(lines[1]);mem=int(next(x.split()[1] for x in lines if x.startswith('MemAvailable:')))*1024
  assert free>need+200*1024**2 and mem>need+2*1024**3,(free,mem,need)
  locked('mkdir '+shlex.quote(remote),check=True);s.update(status='uploading',remote_claimed=True);save()
  with bundle.open('rb') as f:locked('tar -xf - -C '+shlex.quote(remote),stdin=f,timeout=300,check=True)
  locked('sha256sum -c -',input=''.join(h+'  '+remote+'/'+n+'\n' for n,h in hashes.items()).encode(),capture_output=True,check=True)
  locked('test -x '+shlex.quote(remote+'/resident-prefixsafe512'),check=True)
  locked('mkdir '+shlex.quote(remote+'/candidate'),check=True)
  locked(' && '.join('ln -s '+shlex.quote(t)+' '+shlex.quote(remote+'/candidate/'+n) for n,t in cfg['overlay_contract'].items()),check=True)
  locked('mkdir '+shlex.quote(remote+'/tail-candidate'),check=True)
  locked(' && '.join('ln -s '+shlex.quote(t)+' '+shlex.quote(remote+'/tail-candidate/'+n) for n,t in cfg['tail_overlay_contract'].items()),check=True)
  verify()
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 15 3600 env PYTHONDONTWRITEBYTECODE=1 /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+shlex.quote(remote+'/board_fulli16.py')+' --config '+shlex.quote(remote+'/config.json')+' --out '+shlex.quote(out)
  shell=cmd+' > '+shlex.quote(remote+'/process.log')+' 2>&1; rc=$?; printf "%s" "$rc" > '+shlex.quote(remote+'/exit')
  b.ssh('nohup sh -c '+shlex.quote(shell)+' </dev/null >/dev/null 2>&1 &',check=True,timeout=20)
  s.update(status='running');save();deadline=time.monotonic()+3750
  while True:
   time.sleep(15)
   try:z=b.ssh('cat '+shlex.quote(remote+'/exit'),capture_output=True,text=True,timeout=20)
   except Exception:
    s['last_observation_error']=traceback.format_exc();save()
    if time.monotonic()>deadline:raise TimeoutError('Observation deadline; no automatic restart')
    continue
   if z.returncode==0:s['returncode']=int(z.stdout.strip());break
   if time.monotonic()>deadline:raise TimeoutError('Observation deadline; no automatic restart')
  dest=recover();meta=json.loads((dest/'meta.json').read_text()) if (dest/'meta.json').exists() else None
  s.update(status='complete' if s['returncode']==0 and meta and meta['status']=='complete' else 'failed',result=str(dest),meta=meta,ended=time.time());save()
except BaseException:
 s.update(status='failed',error=traceback.format_exc(),ended=time.time(),recovery_pending=bool(s.get('remote_claimed')));save();raise
