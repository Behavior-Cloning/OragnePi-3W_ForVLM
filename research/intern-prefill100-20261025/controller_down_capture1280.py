#!/usr/bin/env python3
"""Replay frozen actual-NPU upstream inputs through original/capture C1280 blocks."""
from pathlib import Path
import argparse,json,shlex,subprocess,sys,time,traceback,tarfile,re
from controller_down_outputfp16_tmpfs import sha,locked,upload,extract,b
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
run='down-capture-c1280-replay-r1';status=R/(run+'-controller.json');assert not status.exists(),'Never duplicate a run'
remote='/dev/shm/opi-prefill100-probe-'+run;out=remote+'/result';exitfile=remote+'/exit';logfile=remote+'/process.log'
state={'run':run,'status':'preparing','started':time.time(),'remote':remote,'scope':'34 actual original-NPU input windows; diagnostic no speedup/whole-model claim'}
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(status)
D=R/'graphs/down-capture-c1280-m16-l2';m=json.loads((D/'conversion.json').read_text());assert json.loads((D/'status.json').read_text())['state']=='complete'
assert m['normal_output_indices']==[0,1,2,3,4] and m['capture_output_indices']==[5,6,7,8]
S=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-t16-cap1-productwide/l2');B=R/'runtime/down-capture1280-probe';pb=json.loads((B/'build.json').read_text());binary=B/'tensor-probe-capture1280';assert sha(binary)==pb['binary_sha256']
for n,h in pb['source_sha256'].items():assert sha(B/'source'/n)==h
sources={'probe':binary,'board_probe_down_capture1280.py':R/'board_probe_down_capture1280.py','original.nb':S/'export_nbg_unify/network_binary.nb','capture.nb':D/'export_nbg_unify/network_binary.nb'}
assert sha(sources['capture.nb'])==m['nb_sha256'];assert sha(sources['original.nb'])==m['source_hashes']['export_nbg_unify/network_binary.nb']=='48b0b3a0c45b1280a6bc818620b3e34117d9c20629b53bb4c7d899a202b3e7d6'
old=R/'actual-npu-range-diagnostic/results/actual-npu-range-r1';events=[json.loads(x) for x in (old/'events.jsonl').read_text().splitlines()];es=[x for x in events if x['event']=='result' and x['capture'] and x['variant']=='c1280-original'];assert len(es)==2 and all(x['capture_control_exact'] for x in es)
seq=[];fixtures=[]
for e in es:
 frozen={x['name']:x for x in e['captures']};folder=old/e['name']/'io'
 for f in sorted(folder.glob('*-input.f32')):
  match=re.fullmatch(r'p(\d+)-n16-v(\d+)-input.f32',f.name);assert match;pos,valid=map(int,match.groups());assert valid in [8,16];o=f.with_name(f.name.replace('-input','-output'));assert sha(f)==frozen[f.name]['sha256'] and sha(o)==frozen[o.name]['sha256'];key=e['case']+'-'+f.stem.removesuffix('-input');name=key+'.f32';sources[name]=f
  rec={'key':key,'case':e['case'],'position':pos,'valid_rows':valid,'input_path':str(f),'input_sha256':sha(f),'original_output_path':str(o),'original_output_sha256':sha(o),'source_event':e['name']};fixtures.append(rec)
  phases=[('original','original.nb',0),('capture','capture.nb',2*4864+2*896)]
  if pos==0:phases.append(('original-after','original.nb',0))
  for label,nb,extra in phases:seq.append({**rec,'run_id':key+'-'+label,'variant':label,'capacity':1280,'query_tokens':16,'first_layer':2,'nb':remote+'/'+nb,'nb_sha256':sha(sources[nb]),'input':remote+'/'+name,'expected_output_bytes':4*16*(896+4*128+extra)})
assert len(fixtures)==34 and len(seq)==70 and sum(x['valid_rows'] for x in fixtures)==528
cfg={'run':run,'warmup':1,'repeats':2,'binary':remote+'/probe','binary_sha256':pb['binary_sha256'],'sequence':seq,'fixtures':fixtures,'source_sha256':{str(p.relative_to(R)):sha(p) for p in [Path(__file__),R/'board_probe_down_capture1280.py',R/'controller_down_outputfp16_tmpfs.py',R/'graphs/build_down_capture1280.py',D/'conversion.json',D/'source-audit.json',B/'build.json',old/'events.jsonl']},'reserve_bytes':1024**3+100*1024**2,'staging_kind':'tmpfs','explicit_root_disk_artifact_bytes':0,'no_full_model_claim':True,'no_speedup_claim':True}
frozen=R/(run+'-config.json');frozen.write_text(json.dumps(cfg,indent=2)+'\n');sources['config.json']=frozen;sourcehash={n:sha(p) for n,p in sources.items()};need=sum(p.stat().st_size for p in sources.values());state.update(upload_bytes=need,source_sha256=sourcehash)
def verify():
 for n,h in sourcehash.items():assert sha(sources[n])==h,n
 for n,h in cfg['source_sha256'].items():assert sha(R/n)==h,n
 for n,h in pb['source_sha256'].items():assert sha(B/'source'/n)==h,n
 for row in fixtures:assert sha(Path(row['original_output_path']))==row['original_output_sha256']
verify()
if not a.execute:print(json.dumps({'prepared':True,'windows':len(fixtures),'runs':len(seq),'upload_raw_bytes':need,'board_operations':0}));sys.exit(0)
save()
archive_in=R/(run+'-upload.tar.gz')
try:
 with tarfile.open(archive_in,'w:gz',dereference=True) as tar:
  for n,f in sources.items():tar.add(f,arcname=n,recursive=False)
 with tarfile.open(archive_in) as tar:
  members=tar.getmembers();assert len(members)==len(sources) and {m.name for m in members}==set(sources) and all(m.isfile() for m in members)
 archive_sha=sha(archive_in);state.update(upload_archive_sha256=archive_sha,upload_archive_bytes=archive_in.stat().st_size);save()
 with b.hostlock():
  verify();m=locked('findmnt -no FSTYPE,OPTIONS /dev/shm',capture_output=True,text=True,check=True).stdout.split();assert m[0]=='tmpfs' and 'noexec' not in ''.join(m[1:]).split(',')
  z=locked('df -B1 --output=avail /dev/shm && cat /proc/meminfo',capture_output=True,text=True,check=True).stdout.splitlines();free=int(z[1]);mem=int(next(l.split()[1] for l in z if l.startswith('MemAvailable:')))*1024
  assert free>need+archive_in.stat().st_size+cfg['reserve_bytes'] and mem>need+archive_in.stat().st_size+cfg['reserve_bytes']+256*1024**2
  locked('mkdir '+shlex.quote(remote),check=True);state.update(status='uploading',remote_claimed=True);save();upload(archive_in,remote+'/upload.tar.gz');verify()
  locked('sha256sum -c -',input=(archive_sha+'  '+remote+'/upload.tar.gz\n').encode(),capture_output=True,check=True);locked('tar -xzf '+shlex.quote(remote+'/upload.tar.gz')+' -C '+shlex.quote(remote),check=True)
  locked('sha256sum -c -',input=''.join(h+'  '+remote+'/'+n+'\n' for n,h in sourcehash.items()).encode(),capture_output=True,check=True);locked('chmod 755 '+shlex.quote(remote+'/probe'),check=True)
  cmd='flock -w 5 /run/opi-npu-kv-benchmark.lock timeout -k 10 900 python3 -u '+shlex.quote(remote+'/board_probe_down_capture1280.py')+' --config '+shlex.quote(remote+'/config.json')+' --out '+shlex.quote(out)
  script=cmd+' >'+shlex.quote(logfile)+' 2>&1; rc=$?; printf "%s" "$rc" >'+shlex.quote(exitfile)
  b.ssh('nohup sh -c '+shlex.quote(script)+' </dev/null >/dev/null 2>&1 &',timeout=20,check=True);state.update(status='running');save();deadline=time.monotonic()+1020
  while True:
   time.sleep(5);z=b.ssh('cat '+shlex.quote(exitfile),timeout=20,capture_output=True,text=True)
   if z.returncode==0:state['returncode']=int(z.stdout.strip());break
   if time.monotonic()>deadline:raise TimeoutError('Observation deadline; do not restart')
  locked('mkdir -p '+shlex.quote(out)+' && cp '+shlex.quote(logfile)+' '+shlex.quote(out+'/process.log'),check=True)
  packet=locked('tar -czf - -C '+shlex.quote(out)+' .',timeout=90,capture_output=True,check=True);archive=R/(run+'.tar.gz');archive.write_bytes(packet.stdout);dest=R/'results'/run;extract(archive,dest)
  meta=json.loads((dest/'meta.json').read_text()) if (dest/'meta.json').exists() else None;ok=state.get('returncode')==0 and meta and meta['status']=='complete';state.update(status='complete' if ok else 'failed',meta=meta,archive_sha256=sha(archive),ended=time.time(),scratch_preserved=True);save()
except BaseException:
 state.update(status='failed',error=traceback.format_exc(),ended=time.time(),recovery_pending=bool(state.get('remote_claimed')));save();raise
