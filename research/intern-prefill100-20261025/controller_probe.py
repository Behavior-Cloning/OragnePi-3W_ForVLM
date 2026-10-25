#!/usr/bin/env python3
"""Root-owned Windows controller; both locks; immutable unique probe run."""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess,sys,tarfile,time,traceback
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('run');a=ap.parse_args();assert a.run.replace('-','').replace('_','').isalnum()
status=R/(a.run+'-controller.json');assert not status.exists(),'unique run required'
s={'run':a.run,'status':'starting','started':time.time()}
def save():
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(status)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def locked(cmd,**kw):return b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),timeout=kw.pop('timeout',30),**kw)
def upload(src,dst):
 data=Path(src).read_bytes();locked('cat > '+shlex.quote(dst+'.part')+' && mv '+shlex.quote(dst+'.part')+' '+shlex.quote(dst),input=data,timeout=180,check=True)
remote='/opt/opi-vlm/intern-prefill100-20261025/runs/'+a.run;output=remote+'/result';exitfile=remote+'/exit';logfile=remote+'/process.log'
save()
try:
 manifest=json.loads((R/'graphs/deployment-manifest.json').read_text());canonical=json.loads((R/'graphs/canonical-fixture-status.json').read_text())
 assert canonical['status']=='complete' and len(manifest['cases'])==10 and len(canonical['cases'])==10
 cb={c['id']:c for c in canonical['cases']};rows={c['id']:c for c in manifest['cases']}
 binary=R/'runtime/tensor-probe-prefill100';binary_sha=sha(binary)
 assert binary_sha=='eed89e71c0030ea15fb90d490611b6d6efdcf88a530906fdc57a1c42498e4da3'
 sequence=[];sources={'probe':binary,'board_probe_suite.py':R/'board_probe_suite.py'}
 for layer in [0,2]:
  for idx,m in enumerate([16,64,128,32,256,16]):
   name=f'c512-m{m}-l{layer}';row=rows[name];inp=R/'fixtures-canonical'/name/'probe-input.f32'
   assert sha(row['nb'])==row['nb_sha256'] and sha(inp)==cb[name]['input_sha256']
   assert sha(inp.with_name('probe-reference.f32'))==cb[name]['reference_sha256']
   sources[name+'.nb']=Path(row['nb']);sources[name+'.input.f32']=inp
   sequence.append({'run_id':name+('-repeat' if idx==5 else ''),'query_tokens':m,'first_layer':layer,'capacity':512,'nb':remote+'/'+name+'.nb','nb_sha256':row['nb_sha256'],'input':remote+'/'+name+'.input.f32','input_sha256':cb[name]['input_sha256'],'reference_sha256':cb[name]['reference_sha256']})
 cfg={'scope':'single fused2 same-input API only; no whole-model TTFT or background audio','binary':remote+'/probe','binary_sha256':binary_sha,'warmup':3,'repeats':30,'sequence':sequence,'canonical_status_sha256':sha(R/'graphs/canonical-fixture-status.json'),'source_sha256':{str(p.relative_to(R)):sha(p) for p in [R/'controller_probe.py',R/'board_probe_suite.py']}}
 frozen=R/(a.run+'-config.json');frozen.write_text(json.dumps(cfg,indent=2));sources['config.json']=frozen
 with b.hostlock():
  p=locked('df -B1 --output=avail /opt/opi-vlm',capture_output=True,text=True,check=True)
  free=int(p.stdout.splitlines()[-1]);need=sum(x.stat().st_size for x in sources.values())
  assert free>need+450*1024*1024,(free,need)
  locked('mkdir -p '+shlex.quote(str(Path(remote).parent))+' && mkdir '+shlex.quote(remote),check=True)
  s.update(status='uploading',bytes=need,remote_claimed=True);save()
  for name,src in sources.items():upload(src,remote+'/'+name)
  locked('chmod 755 '+shlex.quote(remote+'/probe'),check=True)
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 1200 python3 -u '+shlex.quote(remote+'/board_probe_suite.py')+' --config '+shlex.quote(remote+'/config.json')+' --out '+shlex.quote(output)
  script=cmd+' >'+shlex.quote(logfile)+' 2>&1; rc=$?; printf "%s" "$rc" >'+shlex.quote(exitfile)
  b.ssh('nohup sh -c '+shlex.quote(script)+' </dev/null >/dev/null 2>&1 &',check=True,timeout=20)
  s.update(status='running');save();deadline=time.monotonic()+1320
  while True:
   time.sleep(10)
   p=b.ssh('cat '+shlex.quote(exitfile),timeout=20,capture_output=True,text=True)
   if p.returncode==0:s['returncode']=int(p.stdout.strip());break
   if time.monotonic()>deadline:raise TimeoutError('remote deadline; do not auto-restart')
  locked('mkdir -p '+shlex.quote(output)+' && cp '+shlex.quote(logfile)+' '+shlex.quote(output+'/process.log'),check=True)
  # Stream archive over SSH so board does not require an additional tar copy.
  p=locked('tar -czf - -C '+shlex.quote(output)+' .',timeout=90,capture_output=True,check=True)
  archive=R/(a.run+'.tar.gz');archive.write_bytes(p.stdout)
  dest=R/'results'/a.run;dest.mkdir(parents=True,exist_ok=False)
  with tarfile.open(archive) as tar:
   for m in tar.getmembers():
    q=(dest/m.name).resolve();assert q==dest.resolve() or dest.resolve() in q.parents;assert not m.issym() and not m.islnk()
   tar.extractall(dest)
  meta=dest/'meta.json';s['meta']=json.loads(meta.read_text()) if meta.exists() else None
  s.update(status='complete' if s.get('returncode')==0 and s['meta'] and s['meta']['status']=='complete' else 'failed',ended=time.time());save()
except BaseException:
 original=traceback.format_exc();s.update(status='failed',ended=time.time(),error=original,recovery_pending=bool(s.get('remote_claimed')));save()
 if s.get('remote_claimed'):
  try:
   # Acquire both locks before observing or copying partial data. If the remote
   # benchmark is still running the short flock timeout leaves recovery pending.
   with b.hostlock():
    locked('test -d '+shlex.quote(remote)+' && mkdir -p '+shlex.quote(output)+' && test -f '+shlex.quote(logfile)+' && cp '+shlex.quote(logfile)+' '+shlex.quote(output+'/process.log'),check=True)
    packet=locked('tar -czf - -C '+shlex.quote(output)+' .',timeout=90,capture_output=True,check=True)
    partial=R/(a.run+'-partial-'+str(int(time.time()))+'.tar.gz');partial.write_bytes(packet.stdout)
    s.update(recovery_pending=False,partial_archive=str(partial),recovery_note='Remote lock acquired; partial evidence recovered. Original failure retained; no auto-restart.')
  except BaseException:s['recovery_error']=traceback.format_exc()
  save()
 raise
