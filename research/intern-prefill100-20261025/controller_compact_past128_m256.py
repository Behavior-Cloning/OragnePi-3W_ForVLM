#!/usr/bin/env python3
"""M256 all-active-history-preserving past128 geometry and softmax comparison."""
from pathlib import Path
import argparse,json,shlex,subprocess,sys,time,traceback
from controller_down_outputfp16_tmpfs import sha,locked,upload,extract,b
R=Path(__file__).resolve().parent
PIN='18b29578182bf1067512008ad914897a7639bcdba6d89dff2bc4bee173f4c1e4'
a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');a=a.parse_args()
run='compact-past128-m256-abcd-a-r1';status=R/(run+'-controller.json');assert not status.exists(),'Never duplicate a run'
remote='/dev/shm/opi-prefill100-probe-'+run;out=remote+'/result';exitfile=remote+'/exit';logfile=remote+'/process.log'
state={'run':run,'status':'preparing','started':time.time(),'remote':remote,'scope':'two-layer M256, C512 plain/row224 then C128 plain/row224 and C512 plain repeat; not full TTFT'}
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(status)
oldfile=R/'single-block-c512-r1-config.json';old=json.loads(oldfile.read_text());row=next(x for x in old['sequence'] if x['run_id']=='c512-m256-l0')
nb=R/'graphs/cases/c512-m256-l0/export_nbg_unify/network_binary.nb';inp=R/'fixtures-canonical/c512-m256-l0/probe-input.f32';binary=R/'runtime/compact-probe/tensor-probe-compact'
assert row['capacity']==512 and row['first_layer']==0 and row['query_tokens']==256
assert sha(nb)==row['nb_sha256'] and sha(inp)==row['input_sha256'] and sha(binary)==PIN and old['binary_sha256']=='eed89e71c0030ea15fb90d490611b6d6efdcf88a530906fdc57a1c42498e4da3'
compact=R/'graphs/compact-past128-m256';candidate_status=compact/'status.json';cs=json.loads(candidate_status.read_text());assert cs['state']=='complete' and len(cs['cases'])==2
variants={x['variant']:x for x in cs['cases']}
for variant,x in variants.items():
 assert x['state']=='compiled' and x['payload_byte_equal'] and x['quantize_byte_equal'] and x['port_order_verified_from_C']
 assert sha(compact/variant/'export_nbg_unify/network_binary.nb')==x['nb_sha256']
input128=R/'fixtures-compact/c128-m256-l0/probe-input.f32';fixture_meta=R/'fixtures-compact/c128-m256-l0/probe-fixture.json';fm=json.loads(fixture_meta.read_text());assert sha(input128)==fm['input_sha256'] and fm['source_input_sha256']==sha(inp) and fm['history']==128 and fm['same_all_active_KV'] and fm['same_hidden_and_rope']
probe_build=R/'runtime/compact-probe/build.json';pb=json.loads(probe_build.read_text());assert pb['binary_sha256']==PIN
for n,h in pb['candidate_source'].items():assert sha(R/'runtime/compact-probe/source'/n)==h
oldrow=R/'graphs/candidate-softmax-row224-m256/export_nbg_unify/network_binary.nb';assert sha(oldrow)=='0588fb60e18097dc0c6c3e544c8bf11be0c544d84b8a82422e4a67ac9cbebcb7'
sources={'original.nb':nb,'row224-c512.nb':oldrow,'plain-c128.nb':compact/'plain/export_nbg_unify/network_binary.nb','row224-c128.nb':compact/'row224/export_nbg_unify/network_binary.nb','input512.f32':inp,'input128.f32':input128,'probe':binary,'board_probe_compact.py':R/'board_probe_compact.py'}
sequence=[]
for name,filename,capacity in [('original-before','original.nb',512),('row224-c512','row224-c512.nb',512),('plain-c128','plain-c128.nb',128),('row224-c128','row224-c128.nb',128),('original-after','original.nb',512)]:
 inputname='input'+str(capacity)+'.f32'
 sequence.append({**row,'run_id':name,'capacity':capacity,'nb':remote+'/'+filename,'nb_sha256':sha(sources[filename]),'input':remote+'/'+inputname,'input_sha256':sha(sources[inputname])})
cfg={'run':run,'warmup':3,'repeats':30,'candidate_status_sha256':sha(candidate_status),'candidate_status_path':str(candidate_status),'binary':remote+'/probe','binary_sha256':PIN,'sequence':sequence,'capacity':512,'query_rows':256,'history_positions':128,'source_config_sha256':sha(oldfile),'source_sha256':{str(q.relative_to(R)):sha(q) for q in [Path(__file__),R/'board_probe_compact.py',R/'controller_down_outputfp16_tmpfs.py',R/'graphs/build_compact_past128_m256.py',R/'derive_compact_fixture.py',fixture_meta,probe_build]},'reserve_bytes':1024**3+100*1024**2,'staging_kind':'tmpfs','explicit_root_disk_artifact_bytes':0,'same_effective_input_for_variants':True,'input_layout_differs_only_in_zero_masked_past':True,'no_full_model_claim':True}
frozen=R/(run+'-config.json');frozen.write_text(json.dumps(cfg,indent=2)+'\n');sources['config.json']=frozen;sourcehash={n:sha(f) for n,f in sources.items()};need=sum(f.stat().st_size for f in sources.values());state.update(upload_bytes=need,source_sha256=sourcehash)
def verify():
 for n,h in sourcehash.items():assert sha(sources[n])==h,n
 for n,h in cfg['source_sha256'].items():assert sha(R/n)==h,n
 assert sha(oldfile)==cfg['source_config_sha256']
 assert sha(candidate_status)==cfg['candidate_status_sha256']
 for n,h in pb['candidate_source'].items():assert sha(R/'runtime/compact-probe/source'/n)==h
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
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 600 python3 -u '+shlex.quote(remote+'/board_probe_compact.py')+' --config '+shlex.quote(remote+'/config.json')+' --out '+shlex.quote(out)
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
