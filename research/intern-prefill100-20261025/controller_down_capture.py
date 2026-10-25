#!/usr/bin/env python3
"""Noninterference A/B/A for diagnostic down capture; no optimization claim."""
from pathlib import Path
import argparse,json,shlex,subprocess,sys,time,traceback
from controller_down_outputfp16_tmpfs import sha,locked,upload,extract,b
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
run='down-capture-c512-m32-aba-r1';status=R/(run+'-controller.json');assert not status.exists(),'Never duplicate a run'
remote='/dev/shm/opi-prefill100-probe-'+run;out=remote+'/result';exitfile=remote+'/exit';logfile=remote+'/process.log'
state={'run':run,'status':'preparing','started':time.time(),'remote':remote,'scope':'two-layer diagnostic outputs, canonical FP32 history128, not captured full NPU prefix'}
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(status)
D=R/'graphs/down-capture-c512-m32-l2';m=json.loads((D/'conversion.json').read_text());assert json.loads((D/'status.json').read_text())['state']=='complete'
assert m['normal_output_indices']==[0,1,2,3,4] and m['capture_output_indices']==[5,6]
S=R/'graphs/cases/c512-m32-l2';fixture=R/'fixtures-canonical/c512-m32-l2';fm=json.loads((fixture/'probe-fixture.json').read_text());assert fm['history']==128
B=R/'runtime/down-capture-probe';pb=json.loads((B/'build.json').read_text());binary=B/'tensor-probe-capture';assert sha(binary)==pb['binary_sha256']
for n,h in pb['source_sha256'].items():assert sha(B/'source'/n)==h
sources={'probe':binary,'board_probe_down_capture.py':R/'board_probe_down_capture.py','original.nb':S/'export_nbg_unify/network_binary.nb','capture.nb':D/'export_nbg_unify/network_binary.nb','input.f32':fixture/'probe-input.f32'}
assert sha(sources['capture.nb'])==m['nb_sha256'];assert sha(sources['original.nb'])==m['source_hashes']['export_nbg_unify/network_binary.nb'];assert sha(sources['input.f32'])==fm['input_sha256']
seq=[]
for name,nb,extra in [('original-before','original.nb',0),('capture','capture.nb',2*4864),('original-after','original.nb',0)]:
 seq.append({'run_id':name,'capacity':512,'query_tokens':32,'first_layer':2,'nb':remote+'/'+nb,'nb_sha256':sha(sources[nb]),'input':remote+'/input.f32','input_sha256':fm['input_sha256'],'expected_output_bytes':4*32*(896+4*128+extra)})
cfg={'run':run,'warmup':3,'repeats':30,'binary':remote+'/probe','binary_sha256':pb['binary_sha256'],'sequence':seq,'source_sha256':{str(p.relative_to(R)):sha(p) for p in [Path(__file__),R/'board_probe_down_capture.py',R/'controller_down_outputfp16_tmpfs.py',R/'graphs/build_down_capture.py',D/'conversion.json',D/'source-audit.json',B/'build.json',fixture/'probe-fixture.json']},'reserve_bytes':1024**3+100*1024**2,'staging_kind':'tmpfs','explicit_root_disk_artifact_bytes':0,'no_full_model_claim':True,'no_speedup_claim':True}
frozen=R/(run+'-config.json');frozen.write_text(json.dumps(cfg,indent=2)+'\n');sources['config.json']=frozen;sourcehash={n:sha(p) for n,p in sources.items()};need=sum(p.stat().st_size for p in sources.values());state.update(upload_bytes=need,source_sha256=sourcehash)
def verify():
 for n,h in sourcehash.items():assert sha(sources[n])==h,n
 for n,h in cfg['source_sha256'].items():assert sha(R/n)==h,n
 for n,h in pb['source_sha256'].items():assert sha(B/'source'/n)==h,n
verify()
if not a.execute:print(json.dumps({'prepared':True,'board_operations':0,'upload_bytes':need,'source_sha256':cfg['source_sha256']}));sys.exit(0)
save()
try:
 with b.hostlock():
  verify();m=locked('findmnt -no FSTYPE,OPTIONS /dev/shm',capture_output=True,text=True,check=True).stdout.split();assert m[0]=='tmpfs' and 'noexec' not in ''.join(m[1:]).split(',')
  z=locked('df -B1 --output=avail /dev/shm && cat /proc/meminfo',capture_output=True,text=True,check=True).stdout.splitlines();free=int(z[1]);mem=int(next(l.split()[1] for l in z if l.startswith('MemAvailable:')))*1024
  assert free>need+cfg['reserve_bytes'] and mem>need+cfg['reserve_bytes']+256*1024**2,(free,mem,need)
  locked('mkdir '+shlex.quote(remote),check=True);state.update(status='uploading',remote_claimed=True,tmpfs_free_before=free,mem_available_before=mem);save()
  for n,f in sources.items():verify();upload(f,remote+'/'+n)
  verify();locked('sha256sum -c -',input=''.join(h+'  '+remote+'/'+n+'\n' for n,h in sourcehash.items()).encode(),capture_output=True,check=True)
  locked('chmod 755 '+shlex.quote(remote+'/probe'),check=True)
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 600 python3 -u '+shlex.quote(remote+'/board_probe_down_capture.py')+' --config '+shlex.quote(remote+'/config.json')+' --out '+shlex.quote(out)
  script=cmd+' >'+shlex.quote(logfile)+' 2>&1; rc=$?; printf "%s" "$rc" >'+shlex.quote(exitfile)
  b.ssh('nohup sh -c '+shlex.quote(script)+' </dev/null >/dev/null 2>&1 &',timeout=20,check=True);state.update(status='running');save();deadline=time.monotonic()+720
  while True:
   time.sleep(5);z=b.ssh('cat '+shlex.quote(exitfile),timeout=20,capture_output=True,text=True)
   if z.returncode==0:state['returncode']=int(z.stdout.strip());break
   if time.monotonic()>deadline:raise TimeoutError('Observation deadline; do not restart')
  locked('mkdir -p '+shlex.quote(out)+' && cp '+shlex.quote(logfile)+' '+shlex.quote(out+'/process.log'),check=True)
  packet=locked('tar -czf - -C '+shlex.quote(out)+' .',timeout=90,capture_output=True,check=True);archive=R/(run+'.tar.gz');archive.write_bytes(packet.stdout);dest=R/'results'/run;extract(archive,dest)
  meta=json.loads((dest/'meta.json').read_text()) if (dest/'meta.json').exists() else None
  ok=state.get('returncode')==0 and meta and meta['status']=='complete';state.update(status='complete' if ok else 'failed',meta=meta,archive_sha256=sha(archive),ended=time.time(),scratch_preserved=True);save()
except BaseException:
 state.update(status='failed',error=traceback.format_exc(),ended=time.time(),recovery_pending=bool(state.get('remote_claimed')));save()
 if state.get('remote_claimed'):
  try:
   with b.hostlock():
    packet=locked('tar -czf - -C '+shlex.quote(remote)+' .',timeout=90,capture_output=True,check=True);f=R/(run+'-partial.tar.gz');f.write_bytes(packet.stdout);state.update(recovery_pending=False,partial_archive=str(f));save()
  except BaseException:state['recovery_error']=traceback.format_exc();save()
 raise
