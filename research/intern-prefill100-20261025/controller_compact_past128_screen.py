#!/usr/bin/env python3
"""Compact past128 M64/M128/L0 and M256/L2 per-case A/B/A."""
from pathlib import Path
import argparse,json,shlex,subprocess,sys,time,traceback
from controller_down_outputfp16_tmpfs import sha,locked,upload,extract,b
R=Path(__file__).resolve().parent
PIN='18b29578182bf1067512008ad914897a7639bcdba6d89dff2bc4bee173f4c1e4'
a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');a=a.parse_args()
run='compact-past128-screen-aba-r1';status=R/(run+'-controller.json');assert not status.exists(),'Never duplicate a run'
remote='/dev/shm/opi-prefill100-probe-'+run;out=remote+'/result';exitfile=remote+'/exit';logfile=remote+'/process.log'
state={'run':run,'status':'preparing','started':time.time(),'remote':remote,'scope':'three compact-past pairings, each original/compact/original, two-layer only; not full TTFT'}
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(status)
oldfile=R/'single-block-c512-r1-config.json';old=json.loads(oldfile.read_text());oldrows={x['run_id']:x for x in old['sequence']}
binary=R/'runtime/compact-probe/tensor-probe-compact';assert sha(binary)==PIN
compact=R/'graphs/compact-past128-screen';candidate_status=compact/'status.json';cs=json.loads(candidate_status.read_text());assert cs['state']=='complete' and len(cs['cases'])==3
variants={x['variant']:x for x in cs['cases']}
probe_build=R/'runtime/compact-probe/build.json';pb=json.loads(probe_build.read_text());assert pb['binary_sha256']==PIN
for n,h in pb['candidate_source'].items():assert sha(R/'runtime/compact-probe/source'/n)==h
sources={'probe':binary,'board_probe_compact.py':R/'board_probe_compact.py'};sequence=[];metahashes={}
for tokens,first in [(64,0),(128,0),(256,2)]:
 key=f'm{tokens}-l{first}';row=oldrows['c512-'+key];x=variants[key];assert x['capacity']==128 and x['query_tokens']==tokens and x['first_layer']==first
 assert x['state']=='compiled' and x['payload_byte_equal'] and x['quantize_byte_equal'] and x['port_order_verified_from_C']
 original=R/'graphs/cases'/('c512-'+key)/'export_nbg_unify/network_binary.nb';candidate=compact/key/'export_nbg_unify/network_binary.nb'
 inp512=R/'fixtures-canonical'/('c512-'+key)/'probe-input.f32';inp128=R/'fixtures-compact'/('c128-'+key)/'probe-input.f32';meta=R/'fixtures-compact'/('c128-'+key)/'probe-fixture.json';fm=json.loads(meta.read_text());metahashes[str(meta.relative_to(R))]=sha(meta)
 assert sha(original)==row['nb_sha256'] and sha(candidate)==x['nb_sha256'] and sha(inp512)==row['input_sha256']==fm['source_input_sha256'] and sha(inp128)==fm['input_sha256']
 assert fm['history']==128 and fm['query_tokens']==tokens and fm['first_layer']==first and fm['same_all_active_KV'] and fm['same_hidden_and_rope']
 for cap,nb,inp in [(512,original,inp512),(128,candidate,inp128)]:sources[key+f'-c{cap}.nb']=nb;sources[key+f'-c{cap}.f32']=inp
 for label,cap in [('before',512),('compact',128),('after',512)]:
  nbname=key+f'-c{cap}.nb';inname=key+f'-c{cap}.f32';sequence.append({**row,'run_id':key+'-'+label,'capacity':cap,'nb':remote+'/'+nbname,'nb_sha256':sha(sources[nbname]),'input':remote+'/'+inname,'input_sha256':sha(sources[inname])})
cfg={'run':run,'warmup':3,'repeats':30,'candidate_status_sha256':sha(candidate_status),'candidate_status_path':str(candidate_status),'binary':remote+'/probe','binary_sha256':PIN,'sequence':sequence,'past_capacities':[128,512],'query_rows':[64,128,256],'history_positions':128,'source_config_sha256':sha(oldfile),'source_sha256':{str(q.relative_to(R)):sha(q) for q in [Path(__file__),R/'board_probe_compact.py',R/'controller_down_outputfp16_tmpfs.py',R/'graphs/build_compact_past128_screen.py',R/'derive_compact_screen_fixtures.py',probe_build]},'reserve_bytes':1024**3+100*1024**2,'staging_kind':'tmpfs','explicit_root_disk_artifact_bytes':0,'fixture_metadata_sha256':metahashes,'same_effective_input_within_each_pair':True,'input_layout_differs_only_in_zero_masked_past':True,'no_full_model_claim':True}
frozen=R/(run+'-config.json');frozen.write_text(json.dumps(cfg,indent=2)+'\n');sources['config.json']=frozen;sourcehash={n:sha(f) for n,f in sources.items()};need=sum(f.stat().st_size for f in sources.values());state.update(upload_bytes=need,source_sha256=sourcehash)
def verify():
 for n,h in sourcehash.items():assert sha(sources[n])==h,n
 for n,h in cfg['source_sha256'].items():assert sha(R/n)==h,n
 assert sha(oldfile)==cfg['source_config_sha256']
 assert sha(candidate_status)==cfg['candidate_status_sha256']
 for n,h in metahashes.items():assert sha(R/n)==h
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
